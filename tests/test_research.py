from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.eval import datasets
from research.eval.calibration import (
    auroc,
    average_precision,
    binary_calibration_report,
    calibration_report,
    coverage_at_risk,
    risk_coverage,
    selective_at_coverage,
)
from research.eval.run_eval import main as eval_main


class TestCalibration:
    def test_perfectly_calibrated_bins(self) -> None:
        # 10 samples at conf 0.8, 8 correct -> gap 0 in that bin.
        probs = np.array([[0.8, 0.2]] * 10)
        labels = np.array([0] * 8 + [1] * 2)
        rep = calibration_report(probs, labels, n_bins=10)
        assert rep.ece == pytest.approx(0.0, abs=1e-12)
        assert rep.accuracy == pytest.approx(0.8)

    def test_overconfident(self) -> None:
        probs = np.array([[0.99, 0.01]] * 10)
        labels = np.array([0] * 5 + [1] * 5)
        rep = calibration_report(probs, labels)
        assert rep.ece == pytest.approx(0.49)
        assert rep.mce == pytest.approx(0.49)
        # Brier for p=(.99,.01): correct -> 2*0.01^2, wrong -> .99^2+.99^2
        assert rep.brier == pytest.approx((5 * 2 * 0.0001 + 5 * 2 * 0.9801) / 10)

    def test_binary_wrapper(self) -> None:
        rep = binary_calibration_report([0.9, 0.1, 0.8, 0.3], [1, 0, 1, 0])
        assert rep.accuracy == 1.0 and rep.n == 4

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError):
            calibration_report(np.array([0.5, 0.5]), [0])
        with pytest.raises(ValueError):
            calibration_report(np.array([[0.5, 0.5]]), [2])

    def test_risk_coverage(self) -> None:
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        correct = np.array([True, True, False, True])
        rc = risk_coverage(conf, correct)
        assert rc.risk.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
        assert rc.aurc == pytest.approx((0 + 0 + 1 / 3 + 1 / 4) / 4)
        assert coverage_at_risk(rc, 0.0) == (0.5, 0.8)
        assert selective_at_coverage(rc, 0.75)[0] == pytest.approx(1 / 3)

    def test_ranking_metrics_match_sklearn(self) -> None:
        skm = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        s = np.round(rng.random(500) * 0.6 + y * 0.3, 2)  # rounding creates ties
        assert auroc(s, y) == pytest.approx(skm.roc_auc_score(y, s))
        assert average_precision(s, y) == pytest.approx(skm.average_precision_score(y, s))


BGL_LINES = [
    "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 "
    "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
    "KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 "
    "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt",
    "APPREAD 1117869872 2005.06.04 R04-M1-N4-I:J18-U11 2005-06-04-00.24.32.432192 "
    "R04-M1-N4-I:J18-U11 RAS APP FATAL ciod: failed to read message prefix on control stream "
    "(CioStream socket to 172.16.96.116:33569",
    "- 1117838573 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.53.276129 "
    "R02-M1-N0-C:J12-U11 RAS KERNEL INFO generating core.2275",
]


class TestDatasets:
    def test_parse_bgl(self) -> None:
        ex = datasets.parse_bgl(BGL_LINES[1], 1)
        assert ex is not None
        assert ex.labels == {"anomaly": 1, "alert_category": "KERNDTLB"}
        assert ex.service == "kernel" and ex.level == "fatal"
        assert ex.message == "data TLB error interrupt"

    def test_parse_thunderbird(self) -> None:
        ex = datasets.parse_thunderbird(
            "- 1131566461 2005.11.09 dn228 Nov 10 00:01:01 dn228/dn228 "
            "crond(pam_unix)[2915]: session closed for user root",
            0,
        )
        assert ex is not None and ex.host == "dn228" and ex.service == "crond(pam_unix)"
        assert ex.message == "session closed for user root" and ex.labels["anomaly"] == 0

    def test_diverse_sample_caps_templates(self) -> None:
        exs = [
            datasets.Example(id=str(i), message=f"cache error {i}", labels={"anomaly": 0})
            for i in range(100)
        ]
        exs += [
            datasets.Example(id=f"p{i}", message=f"fatal {i}", labels={"anomaly": 1})
            for i in range(100)
        ]
        out = datasets.diverse_sample(exs, 10, per_template=2, positive_ratio=0.5)
        # Only 2 distinct templates x 2 each are available.
        assert len(out) == 4 and sum(e.labels["anomaly"] for e in out) == 2


def test_cli_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "BGL.log"
    raw.write_text("\n".join(BGL_LINES * 3) + "\n")
    sample, results = tmp_path / "s.jsonl", tmp_path / "r.jsonl"
    eval_main(["sample", "--dataset", "bgl", "--raw", str(raw), "--n", "10", "--out", str(sample)])
    assert len(datasets.read_jsonl(sample)) >= 3

    csv_path = tmp_path / "ann.csv"
    eval_main(["export-annotation", "--sample", str(sample), "--out", str(csv_path)])
    header = csv_path.read_text().splitlines()[0]
    assert "anomaly" not in header  # labels hidden from annotators

    eval_main(["run", "--sample", str(sample), "--classifier", "heuristic", "--out", str(results)])
    n_first = len(results.read_text().splitlines())
    eval_main(
        ["run", "--sample", str(sample), "--classifier", "heuristic", "--out", str(results)]
    )  # resumable: nothing new to do
    assert len(results.read_text().splitlines()) == n_first

    prefix = tmp_path / "report"
    eval_main(["analyze", str(results), "--out", str(prefix)])
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert "anomaly" in report["heuristic"]
    assert (tmp_path / "report_reliability.csv").exists()
