from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_triage.config import ClassifierBackend, Settings
from jev_triage.models import SEVERITIES, SUBSYSTEMS


def fake_jev_body(
    severity: dict[str, float] | None = None,
    subsystem: dict[str, float] | None = None,
    nouls: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A well-formed /v1/systemone response body (shape copied from the API reference)."""
    sev = {k: 0.0 for k in SEVERITIES} | (severity or {"info": 1.0})
    sub = {k: 0.0 for k in SUBSYSTEMS} | (subsystem or {"other": 1.0})
    n = {"user_impact": 0.1, "actionable": 0.1, "transient": 0.1, "security_relevant": 0.0}
    n |= nouls or {}

    def conf(p: dict[str, float]) -> float:
        k = len(p)
        return max(0.0, (k * max(p.values()) - 1) / (k - 1))

    answers: dict[str, Any] = {
        "severity": {
            "type": "choice",
            "choice": max(sev, key=sev.get),
            "confidence": conf(sev),
            "probabilities": sev,
        },
        "subsystem": {
            "type": "choice",
            "choice": max(sub, key=sub.get),
            "confidence": conf(sub),
            "probabilities": sub,
        },
    }
    for key, value in n.items():
        answers[key] = {"type": "noul", "noul": value}
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 321, "output_tokens": 40},
    }


def mock_client(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )


def json_response(body: dict[str, Any], status: int = 200) -> httpx2.Response:
    return httpx2.Response(
        status,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-typesafe-request-id": "req_test"},
    )


@pytest.fixture
def heuristic_settings() -> Settings:
    return Settings(backend=ClassifierBackend.HEURISTIC, log_json=False, dedup_ttl_s=60)
