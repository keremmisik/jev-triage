"""Summarise the recent decisions of a running service (e.g. the Docker Compose demo).

python scripts/show_decisions.py [--url http://localhost:8080]
"""

import argparse
import collections
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    url = f"{args.url.rstrip('/')}/v1/decisions?limit={args.limit}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        decisions = json.loads(resp.read())

    print(f"decisions: {len(decisions)}")
    print("by route :", dict(collections.Counter(d["route"] for d in decisions)))
    print("degraded :", sum(d["degraded"] for d in decisions))

    # Alert titles are alert names (no spaces); log titles are the log line itself.
    alerts = [d for d in decisions if d["title"] and " " not in d["title"]]
    print(f"\nAlertmanager alerts ({len(alerts)}):")
    print(f"{'route':<13} {'severity':<9} {'conf':>5}  {'subsystem':<15} {'ms':>5}  alert")
    print("-" * 80)
    for d in alerts:
        print(
            f"{d['route']:<13} {d['severity']:<9} {d['severity_confidence']:>5.2f}  "
            f"{d['subsystem']:<15} {d['latency_ms']:>5.0f}  {d['title']} @ {d['host']}"
        )


if __name__ == "__main__":
    main()
