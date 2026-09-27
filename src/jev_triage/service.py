"""Triage pipeline: normalise -> de-duplicate -> enrich -> classify -> route -> deliver."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import metrics
from .classifier import (
    CircuitBreaker,
    Classifier,
    FallbackClassifier,
    HeuristicClassifier,
    JevClassifier,
)
from .config import ClassifierBackend, Settings
from .enrich import LokiEnricher
from .models import AlertmanagerWebhook, LogRecord, TriageDecision, TriageInput
from .preprocess import DedupCache, from_alert, from_log
from .router import RoutingPolicy, decide
from .sinks import LogSink, Sink, WebhookSink

log = logging.getLogger(__name__)

# Per-message cap before building state; the state builder applies the global cap.
_MAX_MESSAGE_CHARS = 4000


@dataclass
class BatchResult:
    decisions: list[TriageDecision] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)  # ids of earlier decisions reused
    failed: int = 0
    skipped: int = 0  # e.g. resolved alerts


class TriageService:
    def __init__(
        self,
        settings: Settings,
        classifier: Classifier,
        *,
        policy: RoutingPolicy | None = None,
        enricher: LokiEnricher | None = None,
        sinks: Sequence[Sink] = (),
    ) -> None:
        self.settings = settings
        self.classifier = classifier
        self.policy = policy or RoutingPolicy.from_settings(settings)
        self.enricher = enricher
        self.sinks = list(sinks) or [LogSink()]
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._dedup = DedupCache(settings.dedup_ttl_s, settings.dedup_max_entries)
        # Bounded, insertion-ordered buffer of recent decisions for the API/UI.
        self._recent: OrderedDict[str, TriageDecision] = OrderedDict()

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, s: Settings) -> TriageService:
        heuristic = HeuristicClassifier()
        classifier: Classifier
        if s.backend is ClassifierBackend.JEV:
            assert s.typesafe_api_key is not None  # guaranteed by Settings validation
            breaker = CircuitBreaker(s.breaker_failure_threshold, s.breaker_cooldown_s)
            classifier = FallbackClassifier(
                primary=JevClassifier(
                    api_key=s.typesafe_api_key.get_secret_value(),
                    model=s.typesafe_model,
                    base_url=s.typesafe_base_url,
                    timeout_s=s.request_timeout_s,
                    max_retries=s.max_retries,
                    retry_budget_s=s.retry_budget_s,
                    max_state_chars=s.max_state_chars,
                ),
                fallback=heuristic,
                breaker=breaker,
                on_error=lambda reason: metrics.CLASSIFIER_ERRORS.labels(reason).inc(),
            )
        else:
            classifier = heuristic

        enricher = (
            LokiEnricher(
                s.loki_url,
                lookback_s=s.loki_lookback_s,
                max_lines=s.max_context_lines,
                timeout_s=s.loki_timeout_s,
                redact_pii=s.redact_pii,
            )
            if s.loki_url
            else None
        )
        sinks: list[Sink] = [LogSink()]
        if s.webhook_url:
            sinks.append(WebhookSink(s.webhook_url, timeout_s=s.webhook_timeout_s))
        return cls(s, classifier, enricher=enricher, sinks=sinks)

    # ------------------------------------------------------------------ public API
    async def triage_logs(self, records: Iterable[LogRecord]) -> BatchResult:
        events = [
            from_log(r, redact_pii=self.settings.redact_pii, max_message_chars=_MAX_MESSAGE_CHARS)
            for r in records
        ]
        return await self._run_batch(events)

    async def triage_alertmanager(self, payload: AlertmanagerWebhook) -> BatchResult:
        firing = [a for a in payload.alerts if a.status == "firing"]
        events = [
            from_alert(a, redact_pii=self.settings.redact_pii, max_message_chars=_MAX_MESSAGE_CHARS)
            for a in firing
        ]
        result = await self._run_batch(events)
        # Resolved notifications are acknowledged but not re-classified.
        result.skipped += len(payload.alerts) - len(firing)
        return result

    def recent(self, limit: int = 50) -> list[TriageDecision]:
        items = list(self._recent.values())
        return items[-limit:][::-1]  # newest first

    def get(self, decision_id: str) -> TriageDecision | None:
        return self._recent.get(decision_id)

    @property
    def breaker_open(self) -> bool:
        return isinstance(self.classifier, FallbackClassifier) and self.classifier.breaker.is_open

    async def aclose(self) -> None:
        closers = [self.classifier.aclose(), *(s.aclose() for s in self.sinks)]
        if self.enricher:
            closers.append(self.enricher.aclose())
        await asyncio.gather(*closers, return_exceptions=True)

    # ------------------------------------------------------------------ internals
    async def _run_batch(self, events: list[TriageInput]) -> BatchResult:
        result = BatchResult()
        todo: list[TriageInput] = []
        seen_in_batch: set[str] = set()
        in_batch_dups: list[str] = []  # fingerprints repeated within this batch
        for ev in events:
            metrics.EVENTS.labels(ev.kind).inc()
            prior = self._dedup.get(ev.fingerprint)
            if prior is not None:
                metrics.DEDUPED.labels(ev.kind).inc()
                result.duplicates.append(prior)
                continue
            if ev.fingerprint in seen_in_batch:
                metrics.DEDUPED.labels(ev.kind).inc()
                in_batch_dups.append(ev.fingerprint)
                continue
            seen_in_batch.add(ev.fingerprint)
            todo.append(ev)

        outcomes = await asyncio.gather(*(self._process(ev) for ev in todo), return_exceptions=True)
        for ev, out in zip(todo, outcomes, strict=True):
            if isinstance(out, BaseException):
                # One bad event must not sink the whole batch.
                result.failed += 1
                log.error(
                    "triage failed for event", exc_info=out, extra={"fingerprint": ev.fingerprint}
                )
            else:
                result.decisions.append(out)
        # Point in-batch repeats at the decision made for their first occurrence.
        by_fp = {d.input.fingerprint: d.id for d in result.decisions}
        result.duplicates.extend(by_fp[fp] for fp in in_batch_dups if fp in by_fp)
        return result

    async def _process(self, event: TriageInput) -> TriageDecision:
        async with self._sem:
            metrics.IN_FLIGHT.inc()
            try:
                if self.enricher is not None:
                    event = await self.enricher.enrich(event)
                classification = await self.classifier.classify(event)
            finally:
                metrics.IN_FLIGHT.dec()

        route, reasons = decide(classification, self.policy)
        decision = TriageDecision(
            id=uuid.uuid4().hex[:16],
            received_at=datetime.now(timezone.utc),
            input=event,
            classification=classification,
            route=route,
            reasons=reasons,
            needs_attention=route.value in {"page", "ticket", "human_review"},
        )
        self._record(decision)
        await self._deliver(decision)
        return decision

    def _record(self, d: TriageDecision) -> None:
        c = d.classification
        metrics.DECISIONS.labels(
            d.route.value, c.severity.label, c.classifier, str(c.degraded).lower()
        ).inc()
        metrics.CLASSIFIER_LATENCY.labels(c.classifier).observe(c.latency_ms / 1000.0)
        metrics.CONFIDENCE.labels("severity", c.classifier).observe(c.severity.confidence)
        metrics.CONFIDENCE.labels("subsystem", c.classifier).observe(c.subsystem.confidence)
        if c.input_tokens:
            metrics.INPUT_TOKENS.inc(c.input_tokens)
        metrics.BREAKER_OPEN.set(1 if self.breaker_open else 0)

        self._dedup.put(d.input.fingerprint, d.id)
        self._recent[d.id] = d
        while len(self._recent) > self.settings.decision_buffer_size:
            self._recent.popitem(last=False)

    async def _deliver(self, d: TriageDecision) -> None:
        results = await asyncio.gather(*(s.send(d) for s in self.sinks), return_exceptions=True)
        for sink, res in zip(self.sinks, results, strict=True):
            if isinstance(res, BaseException):
                metrics.SINK_ERRORS.labels(sink.name).inc()
                log.error("sink %s raised", sink.name, exc_info=res)
