"""Synthetic sysadmin log generator for the demo stack.

Writes syslog-style lines ("<RFC3339> <host> <service>[pid]: <message>") to
LOG_DIR/<host>.log so Grafana Alloy can tail them into Loki. Scenarios mix
real problems (disk full, OOM, SSH brute force, 502 bursts, DNS failures)
with the routine noise that makes alert fatigue a real problem.

Optionally (TRIAGE_URL set) also pushes WARNING+ lines straight to the triage
service's /v1/logs endpoint in small batches, exercising the second ingest path.

Standard library only, so it runs on a bare python:3.12-slim image.
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOG_DIR", "/var/log/demo"))
RATE = float(os.environ.get("LINES_PER_SECOND", "4"))
TRIAGE_URL = os.environ.get("TRIAGE_URL", "").rstrip("/")
PUSH_INTERVAL_S = float(os.environ.get("PUSH_INTERVAL_S", "10"))
SEED = os.environ.get("SEED")

HOSTS = ["web1", "web2", "db1", "bastion", "worker1"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s loggen %(levelname)s %(message)s")
log = logging.getLogger("loggen")

# (weight, host-pool, service, level, message template)
ROUTINE = [
    (
        30,
        ["web1", "web2"],
        "nginx",
        "info",
        '{ip} - - "GET /api/items/{n} HTTP/1.1" 200 {size} "-" "curl/8.5"',
    ),
    (10, HOSTS, "CRON", "info", "(root) CMD (run-parts /etc/cron.hourly)"),
    (8, HOSTS, "systemd", "info", "Started Session {n} of User deploy."),
    (6, ["db1"], "postgres", "info", "checkpoint complete: wrote {n} buffers"),
    (5, ["bastion"], "sshd", "info", "Accepted publickey for deploy from 10.0.4.{o} port {port}"),
    (4, ["worker1"], "app", "warning", "retrying job {n} after transient error (attempt 1/5)"),
    (3, HOSTS, "systemd-timesyncd", "info", "Synchronized to time server 10.0.0.1:123"),
]
INCIDENTS = [
    # (weight, host-pool, service, level, template, burst_size)
    (
        3,
        ["bastion"],
        "sshd",
        "warning",
        "Failed password for invalid user {user} from 185.220.{o}.{o2} port {port} ssh2",
        12,
    ),
    (
        2,
        ["web1", "web2", "db1"],
        "kernel",
        "err",
        "Out of memory: Killed process {n} (java) total-vm:{size}kB",
        1,
    ),
    (
        2,
        ["db1", "worker1"],
        "app",
        "crit",
        "write /var/lib/data/seg-{n}.dat failed: No space left on device",
        3,
    ),
    (
        2,
        ["web1", "web2"],
        "nginx",
        "err",
        '{ip} - - "POST /api/checkout HTTP/1.1" 502 157 "-" "Mozilla/5.0"',
        15,
    ),
    (
        2,
        ["web1", "worker1"],
        "systemd-resolved",
        "warning",
        "Transaction {n} for <api.payments.internal IN A> scope dns on eth0 failed: SERVFAIL",
        5,
    ),
    (
        1,
        ["db1"],
        "postgres",
        "err",
        "could not receive data from WAL stream: replication terminated by primary server",
        1,
    ),
    (
        1,
        ["web2"],
        "kernel",
        "warning",
        "EXT4-fs warning (device sda1): ext4_dx_add_entry: Directory index full!",
        1,
    ),
]
USERS = ["admin", "oracle", "test", "ubuntu", "postgres", "git"]


def render(template: str) -> str:
    return template.format(
        ip=f"10.0.{random.randint(0, 9)}.{random.randint(2, 254)}",
        n=random.randint(1000, 99999),
        size=random.randint(200, 90000),
        o=random.randint(1, 254),
        o2=random.randint(1, 254),
        port=random.randint(30000, 65000),
        user=random.choice(USERS),
    )


class Writer:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._files = {h: open(LOG_DIR / f"{h}.log", "a", buffering=1) for h in HOSTS}
        self._pending: list[dict] = []
        self._last_push = time.monotonic()

    def emit(self, host: str, service: str, level: str, message: str) -> None:
        now = datetime.now(timezone.utc)
        pid = random.randint(100, 32000)
        self._files[host].write(
            f"{now.isoformat(timespec='milliseconds')} {host} {service}[{pid}]: {message}\n"
        )
        if TRIAGE_URL and level in {"warning", "err", "crit"}:
            self._pending.append(
                {
                    "message": message,
                    "timestamp": now.isoformat(),
                    "host": host,
                    "service": service,
                    "level": level,
                }
            )

    def maybe_push(self) -> None:
        if not TRIAGE_URL or not self._pending:
            return
        if time.monotonic() - self._last_push < PUSH_INTERVAL_S:
            return
        batch, self._pending = self._pending[:200], self._pending[200:]
        self._last_push = time.monotonic()
        req = urllib.request.Request(
            f"{TRIAGE_URL}/v1/logs",
            data=json.dumps({"records": batch}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
                log.info(
                    "pushed %d lines -> processed=%s dedup=%s",
                    len(batch),
                    body.get("processed"),
                    body.get("deduplicated"),
                )
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            # Triage may still be starting; drop the batch rather than grow unbounded.
            log.warning("push to triage failed (%s); dropped %d lines", exc, len(batch))

    def close(self) -> None:
        for f in self._files.values():
            f.close()


def main() -> None:
    if SEED:
        random.seed(int(SEED))
    writer = Writer()
    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("writing to %s at ~%.1f lines/s (push=%s)", LOG_DIR, RATE, TRIAGE_URL or "off")

    r_weights = [r[0] for r in ROUTINE]
    i_weights = [i[0] for i in INCIDENTS]
    while running:
        if random.random() < 0.01:  # ~1 incident per 100 ticks
            _, hosts, svc, lvl, tpl, burst = random.choices(INCIDENTS, i_weights)[0]
            host = random.choice(hosts)
            for _ in range(burst):
                writer.emit(host, svc, lvl, render(tpl))
        else:
            _, hosts, svc, lvl, tpl = random.choices(ROUTINE, r_weights)[0]
            writer.emit(random.choice(hosts), svc, lvl, render(tpl))
        writer.maybe_push()
        time.sleep(random.expovariate(RATE))
    writer.close()
    log.info("stopped")
    sys.exit(0)


if __name__ == "__main__":
    main()
