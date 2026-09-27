"""Runtime configuration.

All settings come from environment variables prefixed with ``TRIAGE_`` (or a
``.env`` file), so the same image runs unchanged in Docker Compose, Kubernetes
or on a bare VM. The TypeSafe API key is the only exception: it is read from
``TYPESAFE_API_KEY`` because that is the variable the official SDK expects.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClassifierBackend(str, Enum):
    """Which classifier makes the triage decision."""

    JEV = "jev"  # TypeSafe Jev, with the heuristic classifier as a fallback
    HEURISTIC = "heuristic"  # Regex/keyword rules only (offline mode, CI, baseline)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535)
    log_level: str = "INFO"
    log_json: bool = True

    # --- Classifier ------------------------------------------------------------
    backend: ClassifierBackend = ClassifierBackend.JEV
    typesafe_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="TYPESAFE_API_KEY",
        description="API key for TypeSafe. Required when backend=jev.",
    )
    typesafe_model: str = "jev-latest"
    typesafe_base_url: str | None = None
    request_timeout_s: float = Field(5.0, gt=0, description="Per-attempt HTTP timeout.")
    max_retries: int = Field(2, ge=0, le=10)
    retry_budget_s: float = Field(10.0, gt=0, description="Total time budget incl. retries.")
    max_concurrency: int = Field(16, ge=1, le=512, description="Parallel in-flight Jev calls.")

    # Circuit breaker: after N consecutive failures, skip Jev for `cooldown` seconds
    # and use the heuristic classifier directly. Keeps triage alive during an outage.
    breaker_failure_threshold: int = Field(5, ge=1)
    breaker_cooldown_s: float = Field(30.0, gt=0)

    # --- Routing policy (see router.py for how these are combined) -------------
    review_min_confidence: float = Field(0.50, ge=0, le=1)
    auto_min_confidence: float = Field(0.80, ge=0, le=1)
    page_risk_threshold: float = Field(0.60, ge=0, le=1)
    impact_threshold: float = Field(0.50, ge=0, le=1)
    actionable_threshold: float = Field(0.50, ge=0, le=1)
    suppress_transient_threshold: float = Field(0.75, ge=0, le=1)
    routine_max_actionable: float = Field(
        0.20, ge=0, le=1, description="Info events below this P(actionable) are routine noise."
    )

    # --- Pre-processing -------------------------------------------------------
    max_state_chars: int = Field(6000, ge=500, description="Hard cap on state size sent to Jev.")
    max_context_lines: int = Field(20, ge=0, le=200)
    redact_pii: bool = True
    dedup_ttl_s: float = Field(300.0, ge=0, description="0 disables de-duplication.")
    dedup_max_entries: int = Field(10_000, ge=100)
    max_batch_size: int = Field(500, ge=1, description="Max records per /v1/logs request.")

    # --- Enrichment -----------------------------------------------------------
    loki_url: str | None = Field(default=None, description="e.g. http://loki:3100")
    loki_lookback_s: int = Field(300, ge=10)
    loki_timeout_s: float = Field(2.0, gt=0)

    # --- Notification sinks ---------------------------------------------------
    webhook_url: str | None = Field(
        default=None, description="Generic JSON webhook for PAGE/TICKET/HUMAN_REVIEW routes."
    )
    webhook_timeout_s: float = Field(3.0, gt=0)
    decision_buffer_size: int = Field(500, ge=10, description="Recent decisions kept in memory.")

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {v}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.review_min_confidence > self.auto_min_confidence:
            raise ValueError("review_min_confidence must be <= auto_min_confidence")
        if self.backend is ClassifierBackend.JEV and self.typesafe_api_key is None:
            raise ValueError(
                "TYPESAFE_API_KEY is required when TRIAGE_BACKEND=jev "
                "(set TRIAGE_BACKEND=heuristic to run offline)"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached so env is parsed once)."""
    return Settings()  # type: ignore[call-arg]
