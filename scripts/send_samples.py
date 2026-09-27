"""Push a handful of representative log lines to a running service and print the decisions.

python scripts/send_samples.py [--url http://localhost:8080]
"""

import argparse
import json
import urllib.request

SAMPLES = [
    (
        "db1",
        "postgres",
        "could not receive data from WAL stream: replication terminated by primary server",
    ),
    ("db1", "app", "write /var/lib/data/seg-7781.dat failed: No space left on device"),
    ("bastion", "sshd", "Failed password for invalid user oracle from 185.220.101.4 port 51234"),
    ("web1", "nginx", '10.0.3.7 - - "POST /api/checkout HTTP/1.1" 502 157 "-" "Mozilla/5.0"'),
    (
        "web1",
        "systemd-resolved",
        "Transaction 4411 for <api.payments.internal IN A> failed: SERVFAIL",
    ),
    ("worker1", "app", "retrying job 55123 after transient error (attempt 1/5)"),
    ("web2", "CRON", "(root) CMD (run-parts /etc/cron.hourly)"),
    ("web2", "systemd", "Started Session 812 of User deploy."),
    # Same template as the SSH line above: de-duplicated, no second model call.
    ("bastion", "sshd", "Failed password for invalid user oracle from 45.13.2.9 port 40000"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    args = parser.parse_args()

    body = {"records": [{"host": h, "service": s, "message": m} for h, s, m in SAMPLES]}
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/logs",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    print(
        f"processed: {result['processed']} | deduplicated: {result['deduplicated']} "
        f"| failed: {result['failed']}\n"
    )
    print(
        f"{'route':<13} {'severity':<9} {'conf':>5}  {'subsystem':<15} {'conf':>5}  {'ms':>5}  msg"
    )
    print("-" * 100)
    for d in result["decisions"]:
        print(
            f"{d['route']:<13} {d['severity']:<9} {d['severity_confidence']:>5.2f}  "
            f"{d['subsystem']:<15} {d['subsystem_confidence']:>5.2f}  {d['latency_ms']:>5.0f}  "
            f"{d['title'][:45]}"
        )
        if d["degraded"]:
            print("   ! degraded: rule-based fallback was used instead of Jev")


if __name__ == "__main__":
    main()
