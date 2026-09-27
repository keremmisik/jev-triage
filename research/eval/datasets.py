"""Dataset loading and sampling for the evaluation.

Sources
-------
* **Loghub BGL / Thunderbird** (Zhu et al., ISSRE 2023): line-level anomaly
  labels. The first token of each raw line is ``-`` for normal lines and an
  alert category (e.g. ``KERNDTLB``) otherwise. Download the raw logs from the
  Zenodo links in https://github.com/logpai/loghub; they are free for research
  use and must be cited.
* **Annotated CSV**: our own severity/subsystem/actionable labels on a sample
  (see ``export_for_annotation``), because no public log dataset has
  operator-style triage labels.

Samples are frozen to JSONL so every classifier sees the identical items.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jev_triage.preprocess import template_of

ANNOTATION_COLUMNS = ("severity", "subsystem", "actionable", "user_impact", "annotator", "notes")


@dataclass
class Example:
    id: str
    message: str
    host: str | None = None
    service: str | None = None
    level: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    template: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            **{k: d.get(k) for k in ("id", "message", "host", "service", "level")},
            labels=d.get("labels") or {},
            template=d.get("template") or "",
        )


# --------------------------------------------------------------------------------------
# Raw Loghub parsers
# --------------------------------------------------------------------------------------

# BGL: LABEL TS DATE NODE TIME NODE TYPE COMPONENT LEVEL CONTENT...
_BGL = re.compile(r"^(\S+) (\d+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (.*)$")
# Thunderbird: LABEL TS DATE HOST MON DAY TIME HOST/HOST SERVICE[PID]: CONTENT
_TBIRD = re.compile(
    r"^(\S+) (\d+) (\S+) (\S+) (\w{3}\s+\d+ [\d:]+) (\S+) ([^:\[\s]+)(?:\[\d+\])?:? ?(.*)$"
)


def parse_bgl(line: str, idx: int) -> Example | None:
    m = _BGL.match(line.rstrip("\n"))
    if not m:
        return None
    label, _ts, _date, node, _time, _node2, _typ, component, level, content = m.groups()
    return Example(
        id=f"bgl-{idx}",
        message=content,
        host=node,
        service=component.lower(),
        level=level.lower(),
        labels={"anomaly": int(label != "-"), "alert_category": None if label == "-" else label},
    )


def parse_thunderbird(line: str, idx: int) -> Example | None:
    m = _TBIRD.match(line.rstrip("\n"))
    if not m:
        return None
    label, _ts, _date, host, _when, _hh, service, content = m.groups()
    return Example(
        id=f"tbird-{idx}",
        message=content or service,
        host=host,
        service=service.lower(),
        labels={"anomaly": int(label != "-"), "alert_category": None if label == "-" else label},
    )


PARSERS = {"bgl": parse_bgl, "thunderbird": parse_thunderbird}


def iter_raw(path: Path, dataset: str, max_lines: int | None = None) -> Iterator[Example]:
    parse = PARSERS[dataset]
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            ex = parse(line, i)
            if ex is not None:
                yield ex


# --------------------------------------------------------------------------------------
# Template-diverse, class-balanced sampling
# --------------------------------------------------------------------------------------


def diverse_sample(
    examples: Iterable[Example],
    n: int,
    *,
    positive_ratio: float | None = 0.5,
    per_template: int = 3,
    seed: int = 13,
) -> list[Example]:
    """Sample ``n`` examples with at most ``per_template`` per log template.

    Raw system logs are dominated by a handful of templates; uniform sampling
    would mostly measure how well a model handles the same five messages.
    Capping per template (reservoir sampling within each template) measures
    breadth instead. ``positive_ratio`` rebalances anomalies vs. normal lines
    (None keeps the natural ratio of the de-duplicated pool).
    """
    rng = random.Random(seed)
    pools: dict[str, list[Example]] = {}
    seen: dict[str, int] = {}
    for ex in examples:
        key = f"{ex.labels.get('anomaly')}|{template_of(ex.message)}"
        ex.template = key.split("|", 1)[1]
        seen[key] = seen.get(key, 0) + 1
        bucket = pools.setdefault(key, [])
        if len(bucket) < per_template:
            bucket.append(ex)
        else:  # reservoir replacement keeps a uniform sample within the template
            j = rng.randrange(seen[key])
            if j < per_template:
                bucket[j] = ex

    pos = [e for k, b in pools.items() if k.startswith("1|") for e in b]
    neg = [e for k, b in pools.items() if not k.startswith("1|") for e in b]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if positive_ratio is None:
        out = pos + neg
        rng.shuffle(out)
        return out[:n]
    n_pos = min(len(pos), round(n * positive_ratio))
    n_neg = min(len(neg), n - n_pos)
    out = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------------------
# Frozen samples and human annotation
# --------------------------------------------------------------------------------------


def write_jsonl(examples: Iterable[Example], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as fh:
        return [Example.from_dict(json.loads(line)) for line in fh if line.strip()]


def export_for_annotation(examples: Iterable[Example], path: Path) -> int:
    """CSV for human labelling. Existing labels (e.g. BGL anomaly) are *hidden*
    so annotators are not anchored by them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "host", "service", "level", "message", *ANNOTATION_COLUMNS])
        for ex in examples:
            w.writerow(
                [
                    ex.id,
                    ex.host or "",
                    ex.service or "",
                    ex.level or "",
                    ex.message,
                    *([""] * len(ANNOTATION_COLUMNS)),
                ]
            )
            n += 1
    return n


def merge_annotations(examples: list[Example], csv_path: Path) -> list[Example]:
    """Attach annotated labels to a frozen sample (rows without a severity are skipped)."""
    by_id = {e.id: e for e in examples}
    out: list[Example] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ex = by_id.get(row["id"])
            if ex is None or not row.get("severity"):
                continue
            for col in ("severity", "subsystem"):
                if row.get(col):
                    ex.labels[col] = row[col].strip().lower()
            for col in ("actionable", "user_impact"):
                if row.get(col):
                    ex.labels[col] = int(row[col].strip().lower() in {"1", "yes", "true", "y"})
            out.append(ex)
    return out
