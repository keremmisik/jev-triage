# jev-triage

**Confidence-gated log and alert triage for sysadmins, powered by [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).**

Alert fatigue is a solved problem on paper and an unsolved one on call. `jev-triage` sits between
Alertmanager / your log pipeline and the humans, and for every event asks a *System One* model a
handful of small, typed questions: *how severe? which subsystem? is anyone impacted? does someone
need to act? is it transient?* The answers come back as **calibrated probabilities in ~100 ms**, and
plain Python code, not a prompt, decides whether to **page**, open a **ticket**, **notify**,
**suppress**, or hand the event to a **human** because the model is unsure.

```
loggen ─► files ─► Alloy ─► Loki ──(LogQL rules)──► Alertmanager ──webhook──► jev-triage ─► page / ticket / notify / suppress / human_review
node-exporter ─► Prometheus ─(rules)─────────────►      ▲                        │   │
                                                         │      Loki context ◄────┘   └─► /metrics ─► Prometheus ─► Grafana
```

## Why this design

| Principle | Where |
|---|---|
| **Atomic questions, policy in code.** The model answers one judgment per question; thresholds live in `router.py`, are named, and are unit-tested. | `questions.py`, `router.py` |
| **Confidence is a routing axis.** Low confidence goes to a human; paging uses the *probability mass* on critical+high, so a split 35%/30% verdict still pages (safety bias). | `router.py` |
| **Arithmetic stays out of the model.** Counts and windows are computed by LogQL/PromQL; Jev receives "14 failed logins in 2m" and judges what that means. | `deploy/loki/rules` |
| **Never go dark.** A circuit breaker falls back to deterministic rules if the API is down; results are marked `degraded` and never auto-suppressed. | `classifier.py` |
| **Privacy first.** Passwords, tokens, JWTs, cloud keys, URL credentials and e-mail addresses are redacted before anything leaves the host. | `preprocess.py` |
| **Pay once per problem.** Events are fingerprinted by *template* (numbers, IPs, UUIDs stripped) and de-duplicated with a TTL cache. | `preprocess.py` |
| **Small, focused state.** Only the event plus a bounded window of same-source Loki lines is sent, since irrelevant context costs accuracy. | `enrich.py`, `build_state()` |

## Quick start

### Full demo stack (Docker Compose)

```bash
cp .env.example .env          # add TYPESAFE_API_KEY (or set TRIAGE_BACKEND=heuristic)
docker compose up -d --build
```

| Service | URL |
|---|---|
| Grafana dashboard | http://localhost:3000 (admin / `GF_ADMIN_PASSWORD`) |
| Triage API docs | http://localhost:8080/docs |
| Recent decisions | http://localhost:8080/v1/decisions |
| Prometheus / Alertmanager | http://localhost:9090 / http://localhost:9093 |

A synthetic log generator produces routine noise plus incidents (SSH brute force, OOM kills,
ENOSPC, nginx 502 bursts, DNS SERVFAIL, lost replication, and a deliberately noisy "job retry"
alert). Within ~2 minutes alerts flow through Alertmanager into triage.

### Service only

```bash
pip install -e ".[dev]"
export TYPESAFE_API_KEY=...            # or: export TRIAGE_BACKEND=heuristic
jev-triage                             # http://localhost:8080

curl -s localhost:8080/v1/logs -H 'content-type: application/json' -d '{
  "records": [{"host": "db1", "service": "postgres",
               "message": "could not receive data from WAL stream: replication terminated by primary server"}]
}' | jq
```

### Helper scripts

```bash
python scripts/smoke_jev.py        # one real Jev call using the key in .env
python scripts/send_samples.py     # push representative log lines, print decisions
python scripts/show_decisions.py   # summarise recent decisions (e.g. of the demo stack)
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/alertmanager` | Alertmanager webhook receiver (resolved alerts are acknowledged, not re-classified) |
| POST | `/v1/logs` | Push up to `TRIAGE_MAX_BATCH_SIZE` log records |
| GET | `/v1/decisions` | Recent decisions, newest first |
| GET | `/v1/decisions/{id}` | Full decision: probabilities, reasons, enrichment context |
| GET | `/healthz`, `/readyz` | Liveness / readiness (`degraded` while the breaker is open) |
| GET | `/metrics` | Prometheus metrics |

## Configuration

All settings are environment variables with the `TRIAGE_` prefix (see `src/jev_triage/config.py`
for the full list and `.env.example` for the common ones). The most important:

| Variable | Default | Meaning |
|---|---|---|
| `TYPESAFE_API_KEY` | none | Required for `TRIAGE_BACKEND=jev` |
| `TRIAGE_BACKEND` | `jev` | `jev` (with rule fallback) or `heuristic` (offline) |
| `TRIAGE_REVIEW_MIN_CONFIDENCE` | `0.50` | Below this → `human_review` |
| `TRIAGE_AUTO_MIN_CONFIDENCE` | `0.80` | Required to auto-suppress |
| `TRIAGE_PAGE_RISK_THRESHOLD` | `0.60` | P(critical)+P(high) needed to page |
| `TRIAGE_LOKI_URL` | none | Enables context enrichment for alerts |
| `TRIAGE_WEBHOOK_URL` | none | Receives page / ticket / human_review decisions |
| `TRIAGE_DEDUP_TTL_S` | `300` | Template de-duplication window (0 = off) |

## Development

```bash
pip install -e ".[dev]"
pytest              # 53 tests; Jev is mocked at the HTTP layer, no API key needed
ruff check . && mypy src
```

## Research

`research/` contains an independent evaluation of Jev on public system logs (Loghub BGL and
Thunderbird) and human-annotated triage labels: accuracy, **calibration** (ECE, Brier,
reliability diagrams), **selective automation** (risk–coverage, AURC, coverage at fixed risk),
latency and cost. See [`research/RESEARCH_PLAN.md`](research/RESEARCH_PLAN.md) and
`jev-eval --help`.

## Limitations

* Jev is an early-access, text-only model with the known weak spots TypeSafe documents for
  jev-1.13 (literal reading, numbers and dates, adversarial content). Log lines can be
  attacker-controlled, so treat model output as advisory for security-relevant events.
* Heuristic fallback probabilities are rule strengths, not calibrated probabilities.
* The demo configs are for local use only (no auth, no TLS, single-node Loki).

## License

MIT
