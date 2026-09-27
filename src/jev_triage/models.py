"""Domain models shared by the API, the triage pipeline and the research harness.

The classifier layer deliberately returns :class:`Classification` (plain
probabilities) instead of SDK objects, so routing logic, tests and the
evaluation harness never depend on a specific model vendor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Label spaces. Keep these small and closed: Jev answers Choice questions over them,
# and the research harness evaluates against exactly the same label sets.
# --------------------------------------------------------------------------------------

SEVERITIES: tuple[str, ...] = ("critical", "high", "medium", "low", "info")

SUBSYSTEMS: tuple[str, ...] = (
    "storage",
    "memory",
    "cpu_load",
    "network",
    "dns",
    "auth_access",
    "database",
    "web_app",
    "kernel_hardware",
    "scheduler_jobs",
    "other",
)


class Route(str, Enum):
    """What the service does with an event after triage."""

    PAGE = "page"  # Wake someone up now
    TICKET = "ticket"  # Needs action, but not urgent
    NOTIFY = "notify"  # FYI / digest
    SUPPRESS = "suppress"  # Known noise, drop (still counted in metrics)
    HUMAN_REVIEW = "human_review"  # Model is unsure; a person decides


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


class LogRecord(BaseModel):
    """A single log line pushed to ``POST /v1/logs``."""

    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1, max_length=20_000)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    host: str | None = Field(default=None, max_length=255)
    service: str | None = Field(default=None, max_length=255)
    level: str | None = Field(default=None, max_length=32)
    labels: dict[str, str] = Field(default_factory=dict)


class LogBatch(BaseModel):
    records: list[LogRecord] = Field(min_length=1)


class AlertmanagerAlert(BaseModel):
    """One alert inside an Alertmanager webhook (payload version 4)."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["firing", "resolved"]
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    startsAt: datetime | None = None
    endsAt: datetime | None = None
    generatorURL: str | None = None
    fingerprint: str | None = None


class AlertmanagerWebhook(BaseModel):
    """Alertmanager ``webhook_config`` payload."""

    model_config = ConfigDict(extra="ignore")

    version: str = "4"
    status: Literal["firing", "resolved"]
    receiver: str = ""
    groupKey: str | None = None
    groupLabels: dict[str, str] = Field(default_factory=dict)
    commonLabels: dict[str, str] = Field(default_factory=dict)
    commonAnnotations: dict[str, str] = Field(default_factory=dict)
    externalURL: str | None = None
    alerts: list[AlertmanagerAlert] = Field(default_factory=list)


class TriageInput(BaseModel):
    """Normalised event that the classifier sees, regardless of where it came from."""

    kind: Literal["log", "alert"]
    title: str
    message: str
    timestamp: datetime
    host: str | None = None
    service: str | None = None
    level: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    context_lines: list[str] = Field(default_factory=list)
    fingerprint: str = ""


# --------------------------------------------------------------------------------------
# Classifier output (vendor-neutral)
# --------------------------------------------------------------------------------------


class ChoiceResult(BaseModel):
    """A categorical answer with its full distribution."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]

    def p(self, label: str) -> float:
        return self.probabilities.get(label, 0.0)


class Classification(BaseModel):
    """Everything a classifier says about one event.

    ``nouls`` holds yes/no probabilities keyed by question name
    (``user_impact``, ``actionable``, ``transient``, ``security_relevant``).
    """

    model_config = ConfigDict(frozen=True)

    severity: ChoiceResult
    subsystem: ChoiceResult
    nouls: dict[str, float]
    classifier: str
    model_version: str | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    degraded: bool = False  # True when produced by the fallback classifier


class TriageDecision(BaseModel):
    """Final, routed decision returned by the API and sent to sinks."""

    id: str
    received_at: datetime
    input: TriageInput
    classification: Classification
    route: Route
    reasons: list[str]
    needs_attention: bool

    def summary(self) -> dict[str, Any]:
        """Compact representation for logs and webhooks (no raw context lines)."""
        c = self.classification
        return {
            "id": self.id,
            "route": self.route.value,
            "severity": c.severity.label,
            "severity_confidence": round(c.severity.confidence, 3),
            "subsystem": c.subsystem.label,
            "subsystem_confidence": round(c.subsystem.confidence, 3),
            "signals": {k: round(v, 3) for k, v in c.nouls.items()},
            "reasons": self.reasons,
            "title": self.input.title,
            "host": self.input.host,
            "service": self.input.service,
            "classifier": c.classifier,
            "degraded": c.degraded,
            "latency_ms": round(c.latency_ms, 1),
        }
