from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from jev_triage.api import create_app
from jev_triage.config import Settings
from jev_triage.service import TriageService


def client(settings: Settings) -> TestClient:
    service = TriageService.from_settings(settings)
    return TestClient(create_app(settings, service))


def am_payload(*alerts: dict[str, Any]) -> dict[str, Any]:
    return {"version": "4", "status": "firing", "receiver": "triage", "alerts": list(alerts)}


def alert(name: str, summary: str, status: str = "firing", fp: str | None = None) -> dict:
    return {
        "status": status,
        "labels": {"alertname": name, "instance": "web1:9100", "job": "node"},
        "annotations": {"summary": summary},
        "startsAt": "2026-09-27T08:00:00Z",
        "fingerprint": fp or name,
    }


def test_health_and_metrics(heuristic_settings: Settings) -> None:
    with client(heuristic_settings) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").json()["status"] == "ok"
        assert "triage_events_total" in c.get("/metrics").text


def test_alertmanager_webhook(heuristic_settings: Settings) -> None:
    with client(heuristic_settings) as c:
        r = c.post(
            "/v1/alertmanager",
            json=am_payload(
                alert("DiskFull", "No space left on device on /var"),
                alert("Old", "resolved one", status="resolved"),
            ),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["processed"] == 1 and body["skipped"] == 1
        d = body["decisions"][0]
        assert d["subsystem"] == "storage" and d["route"] in {"page", "ticket", "human_review"}

        # Same alert again -> de-duplicated, pointing at the first decision.
        again = c.post(
            "/v1/alertmanager",
            json=am_payload(alert("DiskFull", "No space left on device on /var")),
        ).json()
        assert again["processed"] == 0 and again["duplicate_of"] == [d["id"]]

        detail = c.get(f"/v1/decisions/{d['id']}").json()
        assert "probabilities" in detail["classification"]["severity"]


def test_logs_batch_dedups_within_batch(heuristic_settings: Settings) -> None:
    with client(heuristic_settings) as c:
        recs = [
            {
                "message": f"sshd: Failed password for root from 10.0.0.{i} port 22",
                "host": "bastion",
                "service": "sshd",
            }
            for i in range(5)
        ]
        body = c.post("/v1/logs", json={"records": recs}).json()
        assert body["processed"] == 1  # same template -> one model call
        first = body["decisions"][0]["id"]
        assert body["deduplicated"] == 4 and body["duplicate_of"] == [first] * 4
        assert body["decisions"][0]["subsystem"] == "auth_access"
        assert len(c.get("/v1/decisions").json()) == 1


def test_validation_and_limits(heuristic_settings: Settings) -> None:
    s = heuristic_settings.model_copy(update={"max_batch_size": 2})
    with client(s) as c:
        assert c.post("/v1/logs", json={"records": []}).status_code == 422
        too_many = {"records": [{"message": f"m{i}"} for i in range(3)]}
        assert c.post("/v1/logs", json=too_many).status_code == 413
        assert c.get("/v1/decisions/nope").status_code == 404


def test_jev_backend_requires_key() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="TYPESAFE_API_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]
