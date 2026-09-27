"""Classifiers: Jev (primary), heuristic rules (fallback + research baseline), and the
circuit-breaking wrapper that combines them.

Every classifier returns the vendor-neutral :class:`~jev_triage.models.Classification`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Noul,
    Question,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from .models import SEVERITIES, SUBSYSTEMS, ChoiceResult, Classification, TriageInput
from .preprocess import build_state
from .questions import NOUL_QUESTIONS, build_questions

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class ClassifierError(Exception):
    """Base error. ``reason`` is a low-cardinality label used in metrics."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class ClassifierUnavailable(ClassifierError):
    """Transient failure (network, timeout, 429/5xx after retries)."""


class ClassifierMisconfigured(ClassifierError):
    """Permanent failure that retries will not fix (bad key, invalid request)."""


# --------------------------------------------------------------------------------------
# Protocol + helpers
# --------------------------------------------------------------------------------------


@runtime_checkable
class Classifier(Protocol):
    name: str

    async def classify(self, event: TriageInput) -> Classification: ...

    async def aclose(self) -> None: ...


def normalized_confidence(probabilities: dict[str, float]) -> float:
    """Peak-based confidence in [0, 1]: 1 when all mass is on one option, 0 when uniform.

    This mirrors the definition TypeSafe documents for Choice answers,
    ``(n * p_max - 1) / (n - 1)``, so heuristic and Jev confidences are on the
    same scale and the same routing thresholds apply to both.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    peak = max(probabilities.values())
    return max(0.0, min(1.0, (n * peak - 1.0) / (n - 1.0)))


def _choice_from_probs(probs: dict[str, float]) -> ChoiceResult:
    total = sum(probs.values()) or 1.0
    norm = {k: v / total for k, v in probs.items()}
    label = max(norm, key=norm.__getitem__)
    return ChoiceResult(label=label, confidence=normalized_confidence(norm), probabilities=norm)


# --------------------------------------------------------------------------------------
# Jev classifier
# --------------------------------------------------------------------------------------


class JevClassifier:
    """Calls TypeSafe's System One endpoint once per event with all questions."""

    name = "jev"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jev-latest",
        base_url: str | None = None,
        timeout_s: float = 5.0,
        max_retries: int = 2,
        retry_budget_s: float = 10.0,
        max_state_chars: int = 6000,
        extra_nouls: Mapping[str, Noul] | None = None,
        client: AsyncTypeSafeClient | None = None,
    ) -> None:
        # `client` injection exists for tests (httpx2.MockTransport) and for
        # sharing one connection pool between several classifier instances.
        self._client = client or AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout_s,
            retry=RetryPolicy(
                max_retries=max_retries,
                timeout=retry_budget_s,
                # 529 = TypeSafe "overloaded"; covered by the 5xx range.
                http_statuses={408, 429, *range(500, 600)},
            ),
        )
        self._model = model
        self._max_state_chars = max_state_chars
        self._questions: dict[str, Question] = build_questions()
        # Experiments (see research/) may add yes/no questions; answers land in `nouls`.
        extra = dict(extra_nouls or {})
        clash = set(extra) & set(self._questions)
        if clash:
            raise ValueError(f"extra_nouls would overwrite built-in questions: {sorted(clash)}")
        self._questions.update(extra)
        self._noul_keys: tuple[str, ...] = (*NOUL_QUESTIONS, *extra)

    async def classify(self, event: TriageInput) -> Classification:
        state = build_state(event, max_chars=self._max_state_chars)
        started = time.perf_counter()
        try:
            resp = await self._client.system_one(state=state, questions=self._questions)
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as exc:
            raise ClassifierMisconfigured(f"TypeSafe auth failed: {exc}", reason="auth") from exc
        except TypeSafeUnprocessableEntityError as exc:
            # Our request is malformed; log the body because it names the bad field.
            log.error("TypeSafe rejected request (422): %s", getattr(exc, "body", exc))
            raise ClassifierMisconfigured("TypeSafe rejected request", reason="invalid") from exc
        except TypeSafeRateLimitError as exc:
            raise ClassifierUnavailable("rate limited after retries", reason="rate_limit") from exc
        except TypeSafeAPIConnectionError as exc:  # includes timeouts
            raise ClassifierUnavailable(f"connection error: {exc}", reason="connection") from exc
        except TypeSafeAPIError as exc:
            reason = "server" if (getattr(exc, "status", 0) or 0) >= 500 else "http"
            raise ClassifierUnavailable(f"API error {exc.status}", reason=reason) from exc
        except TypeSafeError as exc:  # client-side validation etc.
            raise ClassifierMisconfigured(str(exc), reason="sdk") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._to_classification(resp, latency_ms)

    def _to_classification(self, resp: SystemOneResponse, latency_ms: float) -> Classification:
        try:
            sev = resp.choices["severity"]
            sub = resp.choices["subsystem"]
            nouls = {k: float(resp.nouls[k].noul) for k in self._noul_keys}
        except KeyError as exc:
            # Should be impossible (schema is guaranteed), but never trust the wire.
            raise ClassifierUnavailable(f"missing answer {exc}", reason="bad_response") from exc
        return Classification(
            severity=ChoiceResult(
                label=sev.choice, confidence=sev.confidence, probabilities=dict(sev.probabilities)
            ),
            subsystem=ChoiceResult(
                label=sub.choice, confidence=sub.confidence, probabilities=dict(sub.probabilities)
            ),
            nouls=nouls,
            classifier=self.name,
            model_version=resp.model,
            latency_ms=latency_ms,
            input_tokens=resp.usage.input_tokens if resp.usage else None,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------------------
# Heuristic classifier (fallback and research baseline)
# --------------------------------------------------------------------------------------

_I = re.IGNORECASE
_SUBSYSTEM_RULES: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(
            r"no space left|disk (full|usage)|filesystem|inode|i/o error|ext4|xfs|"
            r"raid|smart|mount",
            _I,
        ),
        "storage",
    ),
    (re.compile(r"out of memory|oom|killed process|swap|cannot allocate|memory", _I), "memory"),
    (re.compile(r"load average|cpu|throttl|soft lockup|high load", _I), "cpu_load"),
    (re.compile(r"nxdomain|dns|resolv|name or service not known|servfail", _I), "dns"),
    (
        re.compile(
            r"sshd|failed password|authentication fail|invalid user|sudo|pam_|"
            r"permission denied|unauthori[sz]ed|login",
            _I,
        ),
        "auth_access",
    ),
    (re.compile(r"postgres|mysql|mariadb|mongo|redis|deadlock|replication|sql", _I), "database"),
    (
        re.compile(r"nginx|apache|http/\d|\b5\d\d\b|upstream|gateway|traceback|exception", _I),
        "web_app",
    ),
    (
        re.compile(
            r"link (is )?down|connection (refused|reset|timed out)|packet loss|"
            r"unreachable|eth\d|tls handshake",
            _I,
        ),
        "network",
    ),
    (
        re.compile(r"kernel|segfault|panic|mce|ecc|temperature|firmware|driver", _I),
        "kernel_hardware",
    ),
    (re.compile(r"cron|systemd.*timer|backup|job (failed|succeeded)", _I), "scheduler_jobs"),
]
_SEV_RULES: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(
            r"panic|data loss|corrupt|service (is )?down|emerg|kernel bug|"
            r"no space left",
            _I,
        ),
        "critical",
    ),
    (
        re.compile(
            r"\bcrit(ical)?\b|oom|killed process|fatal|segfault|replication (broken|"
            r"stopped)|i/o error",
            _I,
        ),
        "high",
    ),
    (re.compile(r"\berr(or)?\b|fail(ed|ure)?|refused|timed? ?out|denied|5\d\d", _I), "medium"),
    (re.compile(r"\bwarn(ing)?\b|retry|slow|deprecated", _I), "low"),
]
_LEVEL_HINT: Final[dict[str, str]] = {
    "emerg": "critical",
    "alert": "critical",
    "crit": "critical",
    "critical": "critical",
    "fatal": "critical",
    "err": "medium",
    "error": "medium",
    "warning": "low",
    "warn": "low",
    "notice": "info",
    "info": "info",
    "debug": "info",
}


class HeuristicClassifier:
    """Deterministic keyword rules. Cheap, explainable and fully offline.

    Output probabilities are *not* calibrated; they encode rule strength so the
    same routing code works. In the paper this is the "rules" baseline.
    """

    name = "heuristic"

    def __init__(self, *, degraded: bool = False) -> None:
        self._degraded = degraded

    async def classify(self, event: TriageInput) -> Classification:
        started = time.perf_counter()
        # Same inputs Jev gets (title, message, enrichment context) for a fair baseline.
        text = "\n".join([event.title, event.message, *event.context_lines])

        sub_probs = dict.fromkeys(SUBSYSTEMS, 0.02)
        for pattern, label in _SUBSYSTEM_RULES:
            if pattern.search(text):
                sub_probs[label] += 0.6
                break
        else:
            sub_probs["other"] += 0.3

        sev_probs = dict.fromkeys(SEVERITIES, 0.03)
        matched = next((lbl for pat, lbl in _SEV_RULES if pat.search(text)), None)
        hint = _LEVEL_HINT.get((event.level or "").lower())
        if matched:
            sev_probs[matched] += 0.55
        if hint:
            sev_probs[hint] += 0.35
        if not matched and not hint:
            sev_probs["info"] += 0.3

        sev = _choice_from_probs(sev_probs)
        is_bad = sev.label in {"critical", "high", "medium"}
        nouls = {
            "user_impact": 0.7 if sev.label in {"critical", "high"} else 0.15,
            "actionable": 0.75 if is_bad else 0.2,
            "transient": 0.6 if re.search(r"retry|retrying|temporar|recovered", text, _I) else 0.2,
            "security_relevant": 0.8 if sub_probs["auth_access"] > 0.5 else 0.05,
        }
        return Classification(
            severity=sev,
            subsystem=_choice_from_probs(sub_probs),
            nouls=nouls,
            classifier=self.name,
            model_version="rules-v1",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            degraded=self._degraded,
        )

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# Circuit breaker + fallback
# --------------------------------------------------------------------------------------


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Minimal async-safe circuit breaker (closed -> open -> half-open -> closed)."""

    def __init__(self, failure_threshold: int, cooldown_s: float) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._state = _BreakerState()
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        opened = self._state.opened_at
        return opened is not None and (time.monotonic() - opened) < self._cooldown

    async def allow(self) -> bool:
        """True if a call may proceed. After cooldown one trial call is let through."""
        async with self._lock:
            if self._state.opened_at is None:
                return True
            if time.monotonic() - self._state.opened_at >= self._cooldown:
                # Half-open: re-arm the timer so concurrent callers keep falling back
                # while this single trial call is in flight.
                self._state.opened_at = time.monotonic()
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            if self._state.opened_at is not None:
                log.info("circuit breaker closed: primary classifier recovered")
            self._state = _BreakerState()

    async def record_failure(self) -> None:
        async with self._lock:
            self._state.consecutive_failures += 1
            if self._state.consecutive_failures >= self._threshold:
                if self._state.opened_at is None:
                    log.warning(
                        "circuit breaker opened after %d consecutive failures",
                        self._state.consecutive_failures,
                    )
                self._state.opened_at = time.monotonic()


class FallbackClassifier:
    """Primary classifier guarded by a circuit breaker, with a degraded fallback.

    Triage must never go dark because a third-party API is down: if Jev fails,
    the event is still classified by rules and marked ``degraded=True`` so the
    router can be more conservative and dashboards can show it.
    """

    def __init__(
        self,
        primary: Classifier,
        fallback: Classifier,
        breaker: CircuitBreaker,
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.breaker = breaker
        self.name = primary.name
        self._on_error = on_error

    async def classify(self, event: TriageInput) -> Classification:
        if await self.breaker.allow():
            try:
                result = await self.primary.classify(event)
            except ClassifierError as exc:
                await self.breaker.record_failure()
                if self._on_error:
                    self._on_error(exc.reason)
                level = (
                    logging.ERROR if isinstance(exc, ClassifierMisconfigured) else logging.WARNING
                )
                log.log(
                    level, "primary classifier failed (%s): %s; using fallback", exc.reason, exc
                )
            else:
                await self.breaker.record_success()
                return result
        fb = await self.fallback.classify(event)
        return fb.model_copy(update={"degraded": True})

    async def aclose(self) -> None:
        await asyncio.gather(self.primary.aclose(), self.fallback.aclose())
