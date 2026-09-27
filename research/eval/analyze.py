"""Turn result JSONL files into the paper's tables.

For each classifier it reports, where labels exist:

* **Anomaly (Loghub labels)**: AUROC, average precision, F1 at p=0.5, binary
  ECE and Brier score of P(anomaly).
* **Severity / subsystem (human labels)**: accuracy, macro-F1, ECE, Brier,
  AURC, and *coverage at a fixed risk*, i.e. the share of events that could be
  auto-routed while keeping errors at or below ``max_risk``.
* **Cost and speed**: latency p50/p95/p99, mean input tokens, USD per 1k events.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jev_triage.models import SEVERITIES, SUBSYSTEMS

from .calibration import (
    auroc,
    average_precision,
    binary_calibration_report,
    calibration_report,
    coverage_at_risk,
    risk_coverage,
)


def load_rows(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Rows grouped by classifier (latest successful row per id wins)."""
    by_clf: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    errors: dict[str, int] = defaultdict(int)
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error"):
                    errors[row["classifier"]] += 1
                    continue
                name = row["classifier"] + (
                    f"@{row['model_version']}"
                    if row.get("model_version") and row["classifier"] == "jev"
                    else ""
                )
                by_clf[name][row["id"]] = row
    for clf, n in errors.items():
        print(f"note: {n} failed rows for {clf} ignored (re-run to retry them)")
    return {k: list(v.values()) for k, v in by_clf.items()}


def anomaly_score(row: dict[str, Any]) -> float:
    """P(anomaly): the dedicated Noul when present, else severity mass (rules baseline)."""
    if "is_anomaly" in (row.get("nouls") or {}):
        return float(row["nouls"]["is_anomaly"])
    sev = row["severity"]
    return float(sev.get("critical", 0) + sev.get("high", 0) + sev.get("medium", 0))


def _macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue  # class absent from both labels and predictions
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def _choice_metrics(
    rows: list[dict[str, Any]], key: str, space: tuple[str, ...], max_risk: float
) -> dict[str, Any] | None:
    rows = [r for r in rows if r["labels"].get(key) in space]
    if not rows:
        return None
    idx = {lbl: i for i, lbl in enumerate(space)}
    probs = np.array([[r[key].get(lbl, 0.0) for lbl in space] for r in rows])
    y = np.array([idx[r["labels"][key]] for r in rows])
    rep = calibration_report(probs, y)
    pred = probs.argmax(axis=1)
    # Selective prediction on the model's *own* confidence field (what the router uses).
    rc = risk_coverage(np.array([r[f"{key}_conf"] for r in rows]), pred == y)
    cov, thr = coverage_at_risk(rc, max_risk)
    return {
        "n": rep.n,
        "accuracy": rep.accuracy,
        "macro_f1": _macro_f1(y, pred, len(space)),
        "ece": rep.ece,
        "brier": rep.brier,
        "aurc": rc.aurc,
        f"coverage@risk<={max_risk}": cov,
        "threshold": thr,
        "_bins": rep.bins,
    }


def _anomaly_metrics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [r for r in rows if r["labels"].get("anomaly") in (0, 1)]
    if not rows:
        return None
    s = np.array([anomaly_score(r) for r in rows])
    y = np.array([r["labels"]["anomaly"] for r in rows], dtype=int)
    pred = s >= 0.5
    tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    rep = binary_calibration_report(s, y)
    return {
        "n": len(rows),
        "positives": int(y.sum()),
        "auroc": auroc(s, y),
        "avg_precision": average_precision(s, y),
        "f1@0.5": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "ece": rep.ece,
        "brier": rep.brier,
        "_bins": rep.bins,
    }


def _cost_speed(rows: list[dict[str, Any]], price_per_mtok: float) -> dict[str, Any]:
    lat = np.array([r.get("latency_ms") or 0.0 for r in rows])
    toks = [r["input_tokens"] for r in rows if r.get("input_tokens")]
    mean_tok = float(np.mean(toks)) if toks else 0.0
    return {
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "mean_input_tokens": mean_tok,
        "usd_per_1k_events": mean_tok * 1000 * price_per_mtok / 1e6,
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def analyze(
    paths: list[Path], *, out_prefix: str | None, price_per_mtok: float, max_risk: float
) -> dict[str, Any]:
    groups = load_rows(paths)
    report: dict[str, Any] = {}
    reliability: list[list[Any]] = []
    md: list[str] = []

    for clf, rows in sorted(groups.items()):
        entry: dict[str, Any] = {"n": len(rows), "cost_speed": _cost_speed(rows, price_per_mtok)}
        for task, metrics in (
            ("anomaly", _anomaly_metrics(rows)),
            ("severity", _choice_metrics(rows, "severity", SEVERITIES, max_risk)),
            ("subsystem", _choice_metrics(rows, "subsystem", SUBSYSTEMS, max_risk)),
        ):
            if metrics is None:
                continue
            for b in metrics.pop("_bins"):
                reliability.append(
                    [clf, task, b.lower, b.upper, b.count, b.mean_confidence, b.accuracy]
                )
            entry[task] = metrics
        report[clf] = entry

    # Markdown: one table per task, classifiers as rows.
    for task in ("anomaly", "severity", "subsystem", "cost_speed"):
        rows = [(c, e[task]) for c, e in report.items() if task in e]
        if not rows:
            continue
        cols = list(rows[0][1].keys())
        md.append(f"### {task}\n")
        md.append("| classifier | " + " | ".join(cols) + " |")
        md.append("|---" * (len(cols) + 1) + "|")
        md.extend(f"| {c} | " + " | ".join(_fmt(m[k]) for k in cols) + " |" for c, m in rows)
        md.append("")
    text = "\n".join(md)
    print(text)

    if out_prefix:
        prefix = Path(out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".md").write_text(text, encoding="utf-8")
        prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
        with prefix.with_name(prefix.name + "_reliability.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "classifier",
                    "task",
                    "bin_lower",
                    "bin_upper",
                    "count",
                    "mean_confidence",
                    "accuracy",
                ]
            )
            w.writerows(reliability)
    return report
