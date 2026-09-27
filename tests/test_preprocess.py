from __future__ import annotations

import time
from datetime import datetime, timezone

from jev_triage.models import AlertmanagerAlert, LogRecord, TriageInput
from jev_triage.preprocess import (
    DedupCache,
    build_state,
    from_alert,
    from_log,
    redact,
    template_of,
)


class TestRedact:
    def test_secrets_are_removed(self) -> None:
        line = (
            "login ok user=bob password=hunter2 token: abc.def "
            "Authorization: Bearer sk-live-123 key=AKIAABCDEFGHIJKLMNOP mail bob@corp.com "
            "db=postgres://admin:s3cret@db1:5432/app"
        )
        out = redact(line)
        for secret in (
            "hunter2",
            "abc.def",
            "sk-live-123",
            "AKIAABCDEFGHIJKLMNOP",
            "bob@corp.com",
            "s3cret",
        ):
            assert secret not in out, secret
        assert "db1:5432" in out  # infrastructure identifiers are kept

    def test_ips_are_kept(self) -> None:
        assert "10.0.0.5" in redact("Failed password for root from 10.0.0.5 port 22")


class TestTemplate:
    def test_variable_tokens_collapse(self) -> None:
        a = template_of("Failed password for root from 10.0.0.5 port 51234 ssh2")
        b = template_of("Failed password for root from 192.168.1.9 port 40000 ssh2")
        assert a == b

    def test_uuid_and_hex(self) -> None:
        t = template_of("req 3f2b8c1e-1a2b-4c3d-9e8f-0123456789ab at 0xdeadbeef")
        assert "<UUID>" in t and "<HEX>" in t


class TestNormalise:
    def test_log_fingerprint_ignores_numbers(self) -> None:
        r1 = LogRecord(message="disk usage 91% on /var", host="web1", service="node")
        r2 = LogRecord(message="disk usage 97% on /var", host="web1", service="node")
        f1 = from_log(r1, redact_pii=True, max_message_chars=1000).fingerprint
        f2 = from_log(r2, redact_pii=True, max_message_chars=1000).fingerprint
        assert f1 == f2

    def test_alert_uses_alertmanager_fingerprint(self) -> None:
        a = AlertmanagerAlert(
            status="firing",
            labels={"alertname": "DiskFull", "instance": "web1:9100", "severity": "warning"},
            annotations={"summary": "Disk almost full"},
            fingerprint="abc123",
        )
        ev = from_alert(a, redact_pii=True, max_message_chars=1000)
        assert ev.fingerprint == "abc123"
        assert ev.title == "DiskFull" and ev.host == "web1:9100" and ev.level == "warning"


class TestBuildState:
    def _event(self, n_context: int, msg: str = "boom") -> TriageInput:
        return TriageInput(
            kind="alert",
            title="X",
            message=msg,
            timestamp=datetime.now(timezone.utc),
            host="h",
            context_lines=[f"line {i} " + "x" * 100 for i in range(n_context)],
        )

    def test_empty_fields_dropped(self) -> None:
        ev = TriageInput(kind="log", title="t", message="m", timestamp=datetime.now(timezone.utc))
        state = build_state(ev, max_chars=6000)
        assert state == {"event": {"source": "log line", "message": "m"}}

    def test_context_trimmed_oldest_first(self) -> None:
        state = build_state(self._event(50), max_chars=2000)
        ctx = state["recent_logs_same_source"]
        assert len(repr(state)) <= 2000
        assert ctx[-1].startswith("line 49")  # newest kept

    def test_huge_message_clipped(self) -> None:
        state = build_state(self._event(0, msg="y" * 20_000), max_chars=1000)
        assert len(repr(state)) <= 1100
        assert state["event"]["message"].endswith("[truncated]")


class TestDedup:
    def test_ttl_expiry(self) -> None:
        cache = DedupCache(ttl_s=0.05, max_entries=10)
        cache.put("k", "d1")
        assert cache.get("k") == "d1"
        time.sleep(0.06)
        assert cache.get("k") is None

    def test_lru_bound(self) -> None:
        cache = DedupCache(ttl_s=60, max_entries=100)
        for i in range(150):
            cache.put(str(i), str(i))
        assert len(cache) == 100 and cache.get("0") is None and cache.get("149") == "149"

    def test_disabled(self) -> None:
        cache = DedupCache(ttl_s=0, max_entries=100)
        cache.put("k", "v")
        assert cache.get("k") is None
