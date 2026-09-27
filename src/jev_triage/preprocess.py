"""Pre-processing: normalisation, redaction, fingerprinting, de-duplication, state building.

Everything here is deterministic and runs *before* the model call. Two goals:

1. **Privacy.** Log lines regularly contain credentials, tokens and personal
   data. They are redacted before anything leaves the host.
2. **Accuracy and cost.** Jev loses accuracy when the state is padded with
   irrelevant detail, so we send only the event plus a bounded window of
   context, and we skip exact repeats entirely (de-duplication).
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Final

from .models import AlertmanagerAlert, LogRecord, TriageInput

# --------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------
# Order matters: specific secrets first, generic patterns last.
_REDACTIONS: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "<PRIVATE_KEY>",
    ),
    (re.compile(r"(?i)\b(authorization:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1<TOKEN>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "<JWT>"),
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "<AWS_KEY_ID>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "<GITHUB_TOKEN>"),
    (
        re.compile(r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\s*[=:]\s*\S+"),
        r"\1=<REDACTED>",
    ),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s:/@]+:[^\s@/]+@"), r"\1<USER>:<PASS>@"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
]


def redact(text: str) -> str:
    """Remove credentials and personal data from a log line.

    IP addresses and hostnames are intentionally *kept*: they are usually the
    most useful triage signal and are internal infrastructure identifiers.
    """
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


# --------------------------------------------------------------------------------------
# Templating / fingerprinting
# --------------------------------------------------------------------------------------
# Variable parts that make otherwise-identical lines look different.
_TEMPLATE_RULES: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<UUID>",
    ),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
        ),
        "<TS>",
    ),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
]
_WS = re.compile(r"\s+")


def template_of(message: str) -> str:
    """Collapse variable tokens so repeated events share one template."""
    for pattern, repl in _TEMPLATE_RULES:
        message = pattern.sub(repl, message)
    return _WS.sub(" ", message).strip()


def fingerprint(*parts: str | None) -> str:
    """Stable short hash over the given parts (None-safe)."""
    h = hashlib.blake2b(digest_size=12)
    for part in parts:
        h.update((part or "").encode("utf-8", "replace"))
        h.update(b"\x1f")  # unit separator avoids ambiguous concatenation
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Normalisation into TriageInput
# --------------------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + " …[truncated]"


def from_log(record: LogRecord, *, redact_pii: bool, max_message_chars: int) -> TriageInput:
    message = redact(record.message) if redact_pii else record.message
    message = _clip(message, max_message_chars)
    return TriageInput(
        kind="log",
        title=_clip(message.splitlines()[0] if message else "", 200),
        message=message,
        timestamp=record.timestamp,
        host=record.host,
        service=record.service,
        level=record.level,
        labels=record.labels,
        fingerprint=fingerprint("log", record.host, record.service, template_of(message)),
    )


def from_alert(
    alert: AlertmanagerAlert, *, redact_pii: bool, max_message_chars: int
) -> TriageInput:
    labels = dict(alert.labels)
    ann = alert.annotations
    title = labels.get("alertname", "unnamed-alert")
    body_parts = [ann.get("summary", ""), ann.get("description", "")]
    message = "\n".join(p for p in body_parts if p) or title
    if redact_pii:
        message = redact(message)
    host = labels.get("instance") or labels.get("host") or labels.get("node")
    service = labels.get("job") or labels.get("service")
    return TriageInput(
        kind="alert",
        title=title,
        message=_clip(message, max_message_chars),
        timestamp=alert.startsAt or datetime.now(timezone.utc),
        host=host,
        service=service,
        # Alertmanager's own "severity" label is kept as a *hint* in labels; the
        # model makes an independent judgment (the research compares both).
        level=labels.get("severity"),
        labels=labels,
        # Alertmanager already fingerprints alerts; reuse it when present.
        fingerprint=alert.fingerprint or fingerprint("alert", title, host, service),
    )


def build_state(event: TriageInput, *, max_chars: int) -> dict[str, Any]:
    """Build the JSON ``state`` for Jev.

    Only fields that help the decision are included; empty fields are dropped.
    Questions refer to ``event`` by name (in backticks), which is the pattern
    TypeSafe recommends for structured state.
    """
    ev: dict[str, Any] = {
        "source": "alertmanager alert" if event.kind == "alert" else "log line",
        "title": event.title if event.kind == "alert" else None,
        "message": event.message,
        "host": event.host,
        "service": event.service,
        "reported_level": event.level,
    }
    if event.kind == "alert":
        # Drop labels that are noise for the model (IDs, prometheus internals).
        skip = {"alertname", "instance", "job", "severity", "__name__", "prometheus"}
        extra = {k: v for k, v in event.labels.items() if k not in skip}
        if extra:
            ev["labels"] = extra
    state: dict[str, Any] = {"event": {k: v for k, v in ev.items() if v}}

    if event.context_lines:
        state["recent_logs_same_source"] = list(event.context_lines)

    # Enforce the size budget by trimming context first, then the message.
    while _approx_size(state) > max_chars and state.get("recent_logs_same_source"):
        state["recent_logs_same_source"].pop(0)  # drop oldest
    if not state.get("recent_logs_same_source"):
        state.pop("recent_logs_same_source", None)
    overflow = _approx_size(state) - max_chars
    if overflow > 0:
        msg = state["event"]["message"]
        state["event"]["message"] = _clip(msg, max(200, len(msg) - overflow))
    return state


def _approx_size(obj: Any) -> int:
    # repr() is a cheap, stable upper-bound proxy for serialized JSON length.
    return len(repr(obj))


# --------------------------------------------------------------------------------------
# De-duplication
# --------------------------------------------------------------------------------------


class DedupCache:
    """Thread-safe TTL + LRU set of recently seen fingerprints.

    A repeated fingerprint inside the TTL returns the *previous* decision id so
    the caller can reuse it instead of paying for another model call.
    """

    def __init__(self, ttl_s: float, max_entries: int) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if now - ts > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
