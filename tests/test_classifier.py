from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx2
import pytest

from jev_triage.classifier import (
    CircuitBreaker,
    ClassifierMisconfigured,
    ClassifierUnavailable,
    FallbackClassifier,
    HeuristicClassifier,
    JevClassifier,
)
from jev_triage.models import TriageInput

from .conftest import fake_jev_body, json_response, mock_client


def event(msg: str = "kernel: Out of memory: Killed process 1234 (java)") -> TriageInput:
    return TriageInput(
        kind="log",
        title=msg,
        message=msg,
        timestamp=datetime.now(timezone.utc),
        host="app1",
        service="kernel",
    )


def jev_with(handler) -> JevClassifier:  # type: ignore[no-untyped-def]
    return JevClassifier(api_key="test-key", client=mock_client(handler))


class TestJevClassifier:
    async def test_request_shape_and_parsing(self) -> None:
        seen: dict = {}

        def handler(req: httpx2.Request) -> httpx2.Response:
            seen["url"] = str(req.url)
            seen["auth"] = req.headers.get("authorization")
            seen["body"] = json.loads(req.content)
            return json_response(
                fake_jev_body(
                    severity={"high": 0.9, "critical": 0.1},
                    subsystem={"memory": 0.95, "kernel_hardware": 0.05},
                    nouls={"actionable": 0.88},
                )
            )

        c = await jev_with(handler).classify(event())
        # One request carrying every question and a structured state.
        assert seen["url"].endswith("/v1/systemone")
        assert seen["auth"] == "Bearer test-key"
        body = seen["body"]
        assert set(body["questions"]) == {
            "severity",
            "subsystem",
            "user_impact",
            "actionable",
            "transient",
            "security_relevant",
        }
        assert body["state"]["event"]["host"] == "app1"
        # Parsed into vendor-neutral types.
        assert c.severity.label == "high" and c.subsystem.label == "memory"
        assert c.nouls["actionable"] == pytest.approx(0.88)
        assert c.model_version == "jev-1.13.0" and c.input_tokens == 321
        assert c.classifier == "jev" and not c.degraded

    @pytest.mark.parametrize(
        ("status", "exc", "reason"),
        [
            (401, ClassifierMisconfigured, "auth"),
            (403, ClassifierMisconfigured, "auth"),
            (422, ClassifierMisconfigured, "invalid"),
            (429, ClassifierUnavailable, "rate_limit"),
            (500, ClassifierUnavailable, "server"),
            (529, ClassifierUnavailable, "server"),
        ],
    )
    async def test_http_errors_are_mapped(self, status: int, exc: type, reason: str) -> None:
        def handler(req: httpx2.Request) -> httpx2.Response:
            return json_response({"detail": "nope"}, status=status)

        with pytest.raises(exc) as info:
            await jev_with(handler).classify(event())
        assert info.value.reason == reason

    async def test_connection_error_is_unavailable(self) -> None:
        def handler(req: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("boom", request=req)

        with pytest.raises(ClassifierUnavailable) as info:
            await jev_with(handler).classify(event())
        assert info.value.reason == "connection"


class TestHeuristic:
    @pytest.mark.parametrize(
        ("msg", "subsystem", "severities"),
        [
            ("kernel: Out of memory: Killed process 1234 (java)", "memory", {"high", "critical"}),
            ("sshd[22]: Failed password for root from 1.2.3.4", "auth_access", {"medium"}),
            ("write failed: No space left on device", "storage", {"critical"}),
            ("CRON[99]: (root) CMD (run-parts /etc/cron.hourly)", "scheduler_jobs", {"info"}),
        ],
    )
    async def test_rules(self, msg: str, subsystem: str, severities: set[str]) -> None:
        c = await HeuristicClassifier().classify(event(msg))
        assert c.subsystem.label == subsystem
        assert c.severity.label in severities
        assert abs(sum(c.severity.probabilities.values()) - 1) < 1e-9


class TestFallback:
    async def test_falls_back_and_opens_breaker(self) -> None:
        calls = {"n": 0}
        errors: list[str] = []

        def handler(req: httpx2.Request) -> httpx2.Response:
            calls["n"] += 1
            return json_response({}, status=503)

        fb = FallbackClassifier(
            jev_with(handler),
            HeuristicClassifier(),
            CircuitBreaker(2, cooldown_s=60),
            on_error=errors.append,
        )
        for _ in range(5):
            c = await fb.classify(event())
            assert c.degraded and c.classifier == "heuristic"
        # Two real attempts, then the breaker short-circuits the remaining three.
        assert calls["n"] == 2 and errors == ["server", "server"]
        assert fb.breaker.is_open

    async def test_recovers_after_cooldown(self) -> None:
        state = {"fail": True}

        def handler(req: httpx2.Request) -> httpx2.Response:
            return json_response({}, 500) if state["fail"] else json_response(fake_jev_body())

        breaker = CircuitBreaker(1, cooldown_s=0.01)
        fb = FallbackClassifier(jev_with(handler), HeuristicClassifier(), breaker)
        assert (await fb.classify(event())).degraded
        state["fail"] = False
        import asyncio

        await asyncio.sleep(0.02)
        c = await fb.classify(event())
        assert not c.degraded and c.classifier == "jev" and not breaker.is_open


async def test_extra_nouls_are_sent_and_returned() -> None:
    from typesafe_sdk import Noul

    def handler(req: httpx2.Request) -> httpx2.Response:
        assert "is_anomaly" in json.loads(req.content)["questions"]
        return json_response(fake_jev_body(nouls={"is_anomaly": 0.93}))

    clf = JevClassifier(
        api_key="k", client=mock_client(handler), extra_nouls={"is_anomaly": Noul(instructions="?")}
    )
    assert (await clf.classify(event())).nouls["is_anomaly"] == pytest.approx(0.93)
    with pytest.raises(ValueError, match="overwrite"):
        JevClassifier(
            api_key="k",
            client=mock_client(handler),
            extra_nouls={"actionable": Noul(instructions="?")},
        )
