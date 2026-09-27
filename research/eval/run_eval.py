"""Evaluation CLI.

Typical workflow::

    # 1. Freeze a template-diverse, balanced sample (identical input for all systems)
    jev-eval sample --dataset bgl --raw data/BGL.log --n 2000 --out data/bgl_2k.jsonl

    # 2. (Optional) export for human annotation, label it, then merge back
    jev-eval export-annotation --sample data/bgl_2k.jsonl --out data/bgl_2k_annotate.csv
    jev-eval merge-annotation --sample data/bgl_2k.jsonl --csv data/bgl_2k_annotated.csv \
        --out data/bgl_2k_labeled.jsonl

    # 3. Run each system; runs are resumable (already-scored ids are skipped)
    jev-eval run --sample data/bgl_2k.jsonl --classifier heuristic --out results/bgl_rules.jsonl
    jev-eval run --sample data/bgl_2k.jsonl --classifier jev --out results/bgl_jev.jsonl

    # 4. Metrics tables (+ reliability CSV for plots)
    jev-eval analyze results/bgl_*.jsonl --out results/bgl_report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typesafe_sdk import Noul

from jev_triage.classifier import (
    Classifier,
    ClassifierError,
    HeuristicClassifier,
    JevClassifier,
)
from jev_triage.models import TriageInput

from . import datasets

log = logging.getLogger("jev_eval")

# Experiment-only question used for the Loghub anomaly task. Kept out of the
# production question set so the service's behaviour is unaffected.
ANOMALY_NOUL = {
    "is_anomaly": Noul(
        instructions=(
            "Does the `event` report a failure, error, or abnormal condition, rather than "
            "normal operation?"
        ),
        criteria={
            "true": "Something is wrong or abnormal (fault, error, exception, degraded state).",
            "false": "Routine, expected, or purely informational message.",
        },
    )
}


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def build_classifier(name: str, model: str) -> Classifier:
    if name == "heuristic":
        return HeuristicClassifier()
    if name == "jev":
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            sys.exit("TYPESAFE_API_KEY is not set")
        # No fallback in evaluation: a failed call is recorded as an error,
        # never silently replaced by another system's answer.
        return JevClassifier(
            api_key=key,
            model=model,
            extra_nouls=ANOMALY_NOUL,
            max_retries=4,
            retry_budget_s=60.0,
            timeout_s=15.0,
        )
    raise SystemExit(f"unknown classifier {name!r}")


def to_input(ex: datasets.Example) -> TriageInput:
    return TriageInput(
        kind="log",
        title=ex.message[:200],
        message=ex.message,
        timestamp=datetime.now(timezone.utc),
        host=ex.host,
        service=ex.service,
        level=ex.level,
        fingerprint=ex.id,
    )


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line after a crash
            if row.get("error") is None:
                done.add(row["id"])
    return done


async def run(args: argparse.Namespace) -> None:
    examples = datasets.read_jsonl(Path(args.sample))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = [e for e in examples if e.id not in done][: args.limit or None]
    log.info("%d examples, %d already scored, %d to run", len(examples), len(done), len(todo))
    if not todo:
        return

    clf = build_classifier(args.classifier, args.model)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters = {"ok": 0, "err": 0}
    started = time.perf_counter()

    async def one(ex: datasets.Example, fh: Any) -> None:
        async with sem:
            row: dict[str, Any] = {"id": ex.id, "labels": ex.labels, "classifier": clf.name}
            try:
                c = await clf.classify(to_input(ex))
                row.update(
                    model_version=c.model_version,
                    latency_ms=c.latency_ms,
                    input_tokens=c.input_tokens,
                    severity=c.severity.probabilities,
                    severity_conf=c.severity.confidence,
                    subsystem=c.subsystem.probabilities,
                    subsystem_conf=c.subsystem.confidence,
                    nouls=c.nouls,
                    error=None,
                )
                counters["ok"] += 1
            except ClassifierError as exc:
                row["error"] = f"{exc.reason}: {exc}"
                counters["err"] += 1
            async with write_lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()  # results survive interruption; rerun resumes

    # Small local appends; blocking I/O here is negligible next to network latency.
    with out.open("a", encoding="utf-8") as fh:  # noqa: ASYNC230
        try:
            await asyncio.gather(*(one(e, fh) for e in todo))
        finally:
            await clf.aclose()
    elapsed = time.perf_counter() - started
    log.info(
        "done: %d ok, %d errors in %.1fs (%.1f ex/s)",
        counters["ok"],
        counters["err"],
        elapsed,
        len(todo) / max(elapsed, 1e-9),
    )
    if counters["err"]:
        log.warning("errors are recorded in the output and retried on the next run")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        prog="jev-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="freeze a diverse, balanced sample from raw Loghub logs")
    s.add_argument("--dataset", choices=sorted(datasets.PARSERS), required=True)
    s.add_argument("--raw", required=True, help="path to the raw .log file")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument(
        "--positive-ratio",
        type=float,
        default=0.5,
        help="anomaly share; use -1 to keep the natural ratio",
    )
    s.add_argument("--per-template", type=int, default=3)
    s.add_argument("--max-lines", type=int, default=None, help="read only the first N lines")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--out", required=True)

    e = sub.add_parser("export-annotation", help="write a CSV for human labelling")
    e.add_argument("--sample", required=True)
    e.add_argument("--out", required=True)

    m = sub.add_parser("merge-annotation", help="attach labelled CSV to a frozen sample")
    m.add_argument("--sample", required=True)
    m.add_argument("--csv", required=True)
    m.add_argument("--out", required=True)

    r = sub.add_parser("run", help="score a frozen sample with one classifier")
    r.add_argument("--sample", required=True)
    r.add_argument("--classifier", choices=["jev", "heuristic"], required=True)
    r.add_argument("--model", default="jev-latest")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--limit", type=int, default=0, help="score at most N new examples")
    r.add_argument("--out", required=True)

    a = sub.add_parser("analyze", help="metrics tables from one or more result files")
    a.add_argument("results", nargs="+")
    a.add_argument("--out", default=None, help="prefix for .md/.json/reliability .csv output")
    a.add_argument(
        "--price-per-mtok",
        type=float,
        default=0.042,
        help="USD per million input tokens (TypeSafe list price, Sep 2026)",
    )
    a.add_argument("--max-risk", type=float, default=0.02)

    args = p.parse_args(argv)
    if args.cmd == "sample":
        ratio = None if args.positive_ratio < 0 else args.positive_ratio
        exs = datasets.diverse_sample(
            datasets.iter_raw(Path(args.raw), args.dataset, args.max_lines),
            args.n,
            positive_ratio=ratio,
            per_template=args.per_template,
            seed=args.seed,
        )
        n = datasets.write_jsonl(exs, Path(args.out))
        pos = sum(e.labels.get("anomaly", 0) for e in exs)
        log.info("wrote %d examples (%d anomalies) to %s", n, pos, args.out)
    elif args.cmd == "export-annotation":
        n = datasets.export_for_annotation(datasets.read_jsonl(Path(args.sample)), Path(args.out))
        log.info("wrote %d rows to %s", n, args.out)
    elif args.cmd == "merge-annotation":
        merged = datasets.merge_annotations(datasets.read_jsonl(Path(args.sample)), Path(args.csv))
        n = datasets.write_jsonl(merged, Path(args.out))
        log.info("wrote %d labelled examples to %s", n, args.out)
    elif args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "analyze":
        from .analyze import analyze

        analyze(
            [Path(x) for x in args.results],
            out_prefix=args.out,
            price_per_mtok=args.price_per_mtok,
            max_risk=args.max_risk,
        )


if __name__ == "__main__":
    main()
