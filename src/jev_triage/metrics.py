"""Prometheus metrics. Label sets are kept low-cardinality on purpose
(no hosts, no fingerprints) so the service is safe to scrape at scale."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

EVENTS = Counter("triage_events_total", "Events received, before de-duplication.", ["kind"])
DEDUPED = Counter(
    "triage_deduplicated_total",
    "Events skipped because an identical one was seen recently.",
    ["kind"],
)
DECISIONS = Counter(
    "triage_decisions_total", "Routed decisions.", ["route", "severity", "classifier", "degraded"]
)
CLASSIFIER_LATENCY = Histogram(
    "triage_classifier_latency_seconds",
    "Wall time of one classifier call.",
    ["classifier"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5),
)
CONFIDENCE = Histogram(
    "triage_confidence",
    "Model confidence per question.",
    ["question", "classifier"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
CLASSIFIER_ERRORS = Counter(
    "triage_classifier_errors_total", "Primary classifier failures by reason.", ["reason"]
)
INPUT_TOKENS = Counter("triage_input_tokens_total", "Input tokens billed by TypeSafe.")
BREAKER_OPEN = Gauge("triage_circuit_breaker_open", "1 while the Jev circuit breaker is open.")
ENRICH_ERRORS = Counter("triage_enrichment_errors_total", "Loki context lookups that failed.")
SINK_ERRORS = Counter("triage_sink_errors_total", "Notification deliveries that failed.", ["sink"])
IN_FLIGHT = Gauge("triage_in_flight", "Events currently being classified.")
