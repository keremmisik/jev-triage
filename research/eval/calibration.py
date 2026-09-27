"""Calibration and selective-prediction metrics (NumPy only, no sklearn).

Definitions follow the standard literature so numbers are comparable:

* ECE / MCE: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
  (equal-width binning over the top-label confidence).
* Brier score: mean squared error between the probability vector and the one-hot label.
* Risk-coverage curve / AURC: Geifman & El-Yaniv, "Selective Classification for
  Deep Neural Networks", NeurIPS 2017. This is the metric that matches how
  jev-triage uses confidence: "act automatically only above threshold t".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

F = NDArray[np.float64]


def _as_probs(probs: ArrayLike) -> F:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probs must be 2-D (n_samples, n_classes)")
    if np.any(p < -1e-9):
        raise ValueError("probabilities must be non-negative")
    s = p.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("each row must have positive mass")
    return p / s  # renormalise small numeric drift from the API


def _as_labels(labels: ArrayLike, n_classes: int) -> NDArray[np.int64]:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("labels must be 1-D")
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError("label index out of range")
    return y


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    accuracy: float
    ece: float
    mce: float
    brier: float
    nll: float
    bins: list[ReliabilityBin]


def reliability_bins(
    confidence: F, correct: NDArray[np.bool_], n_bins: int
) -> list[ReliabilityBin]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Right-closed bins; the first bin also includes 0.
        mask = (
            (confidence > lo) & (confidence <= hi) if i else (confidence >= lo) & (confidence <= hi)
        )
        cnt = int(mask.sum())
        out.append(
            ReliabilityBin(
                lower=float(lo),
                upper=float(hi),
                count=cnt,
                mean_confidence=float(confidence[mask].mean()) if cnt else float("nan"),
                accuracy=float(correct[mask].mean()) if cnt else float("nan"),
            )
        )
    return out


def calibration_report(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> CalibrationReport:
    """Top-label calibration for a multi-class classifier.

    Note: ``confidence`` here is the *max probability*, the textbook quantity.
    TypeSafe's own ``confidence`` field is a different, peak-normalised
    statistic; the analysis script reports both.
    """
    p = _as_probs(probs)
    y = _as_labels(labels, p.shape[1])
    if len(y) != len(p):
        raise ValueError("probs and labels length mismatch")
    n = len(y)
    if n == 0:
        raise ValueError("empty input")

    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    bins = reliability_bins(conf, correct, n_bins)
    gaps = [(b.count, abs(b.accuracy - b.mean_confidence)) for b in bins if b.count]
    ece = sum(c * g for c, g in gaps) / n
    mce = max((g for _, g in gaps), default=0.0)

    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0))))
    return CalibrationReport(
        n=n,
        accuracy=float(correct.mean()),
        ece=float(ece),
        mce=float(mce),
        brier=brier,
        nll=nll,
        bins=bins,
    )


def binary_calibration_report(
    p_pos: ArrayLike, labels: ArrayLike, n_bins: int = 15
) -> CalibrationReport:
    """Calibration for a yes/no probability (e.g. a Jev Noul)."""
    p1 = np.clip(np.asarray(p_pos, dtype=np.float64), 0.0, 1.0)
    return calibration_report(np.stack([1.0 - p1, p1], axis=1), labels, n_bins)


# --------------------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskCoverage:
    coverage: F  # fraction of samples the system acts on, descending confidence
    risk: F  # error rate among the samples it acts on
    thresholds: F  # confidence threshold that yields each coverage level
    aurc: float


def risk_coverage(confidence: ArrayLike, correct: ArrayLike) -> RiskCoverage:
    """Risk-coverage curve. Lower AURC = confidence ranks errors below successes better."""
    c = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if c.shape != ok.shape or c.ndim != 1 or c.size == 0:
        raise ValueError("confidence and correct must be equal-length, non-empty 1-D arrays")
    order = np.argsort(-c, kind="stable")
    errors = np.cumsum(~ok[order])
    k = np.arange(1, len(c) + 1)
    risk = errors / k
    coverage = k / len(c)
    return RiskCoverage(coverage=coverage, risk=risk, thresholds=c[order], aurc=float(risk.mean()))


def selective_at_coverage(rc: RiskCoverage, target_coverage: float) -> tuple[float, float]:
    """(risk, threshold) when acting on the most-confident ``target_coverage`` share."""
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    i = int(np.searchsorted(rc.coverage, target_coverage, side="left"))
    i = min(i, len(rc.coverage) - 1)
    return float(rc.risk[i]), float(rc.thresholds[i])


def coverage_at_risk(rc: RiskCoverage, max_risk: float) -> tuple[float, float]:
    """Largest coverage whose error rate is <= ``max_risk`` -> (coverage, threshold).

    This answers the operational question: "if we tolerate 2% wrong automatic
    decisions, what share of events can skip a human?"
    """
    ok = np.nonzero(rc.risk <= max_risk)[0]
    if ok.size == 0:
        return 0.0, float("inf")
    i = int(ok[-1])
    return float(rc.coverage[i]), float(rc.thresholds[i])


# --------------------------------------------------------------------------------------
# Ranking metrics for binary tasks
# --------------------------------------------------------------------------------------


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under ROC via the Mann-Whitney U statistic (ties get half credit)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    # Average ranks for ties.
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the precision-recall curve (step-wise, as in sklearn)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    # Evaluate only at the last index of each distinct score (threshold).
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s) - 1]
    precision = tp[last] / (last + 1)
    recall = tp[last] / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))
