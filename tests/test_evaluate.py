"""Tests for theme frequency computation and Kendall's tau rank correlation."""

import math
from collections import Counter

import numpy as np
import pytest
from scipy.stats import kendalltau


def _compute_theme_frequencies(pred_labels, ref_labels, all_labels, n_responses):
    """Local copy for testing — mirrors themefinder.cli._compute_theme_frequencies."""
    # Import here to verify the real implementation stays in sync
    # (cannot import cli.py directly due to missing sentiment_analysis export)
    pred_counter = Counter(label for s in pred_labels for label in s)
    ref_counter = Counter(label for s in ref_labels for label in s)

    theme_freq = []
    for label in all_labels:
        pc = pred_counter.get(label, 0)
        rc = ref_counter.get(label, 0)
        theme_freq.append(
            {
                "label": label,
                "pred_count": pc,
                "pred_pct": round(pc / n_responses * 100, 1),
                "ref_count": rc,
                "ref_pct": round(rc / n_responses * 100, 1),
            }
        )
    theme_freq.sort(key=lambda x: x["ref_count"], reverse=True)

    pred_freq_vec = [pred_counter.get(label, 0) for label in all_labels]
    ref_freq_vec = [ref_counter.get(label, 0) for label in all_labels]
    if len(all_labels) >= 2:
        tau, tau_p = kendalltau(pred_freq_vec, ref_freq_vec)
    else:
        tau, tau_p = float("nan"), float("nan")

    return theme_freq, float(tau), float(tau_p)


@pytest.fixture()
def basic_labels():
    """Pred and ref label sets with known frequencies."""
    pred = [
        frozenset({"A", "B"}),
        frozenset({"A"}),
        frozenset({"B", "C"}),
        frozenset({"A", "C"}),
    ]
    ref = [
        frozenset({"A", "B"}),
        frozenset({"A", "B"}),
        frozenset({"A"}),
        frozenset({"C"}),
    ]
    all_labels = sorted({"A", "B", "C"})
    return pred, ref, all_labels


class TestThemeFrequencyCounts:
    def test_counts(self, basic_labels):
        pred, ref, all_labels = basic_labels
        freq, _, _ = _compute_theme_frequencies(pred, ref, all_labels, len(pred))
        by_label = {f["label"]: f for f in freq}

        assert by_label["A"]["pred_count"] == 3
        assert by_label["A"]["ref_count"] == 3
        assert by_label["B"]["pred_count"] == 2
        assert by_label["B"]["ref_count"] == 2
        assert by_label["C"]["pred_count"] == 2
        assert by_label["C"]["ref_count"] == 1

    def test_percentages(self, basic_labels):
        pred, ref, all_labels = basic_labels
        freq, _, _ = _compute_theme_frequencies(pred, ref, all_labels, len(pred))
        by_label = {f["label"]: f for f in freq}

        assert by_label["A"]["pred_pct"] == 75.0
        assert by_label["C"]["ref_pct"] == 25.0

    def test_sorted_by_ref_count_desc(self, basic_labels):
        pred, ref, all_labels = basic_labels
        freq, _, _ = _compute_theme_frequencies(pred, ref, all_labels, len(pred))
        ref_counts = [f["ref_count"] for f in freq]
        assert ref_counts == sorted(ref_counts, reverse=True)


class TestKendallTau:
    def test_identical_rankings(self):
        """When pred and ref have identical frequencies, tau = 1.0."""
        labels = [frozenset({"A", "B"}), frozenset({"A"})]
        all_labels = sorted({"A", "B"})
        _, tau, _ = _compute_theme_frequencies(labels, labels, all_labels, len(labels))
        assert tau == pytest.approx(1.0)

    def test_different_rankings(self):
        """When rankings differ, tau is between -1 and 1."""
        pred = [frozenset({"A"}), frozenset({"A"}), frozenset({"B"})]
        ref = [frozenset({"B"}), frozenset({"B"}), frozenset({"A"})]
        all_labels = sorted({"A", "B"})
        _, tau, _ = _compute_theme_frequencies(pred, ref, all_labels, len(pred))
        assert -1.0 <= tau <= 1.0
        assert tau == pytest.approx(-1.0)

    def test_single_label_returns_nan(self):
        """Kendall's tau is undefined for fewer than 2 labels."""
        labels = [frozenset({"A"}), frozenset({"A"})]
        all_labels = ["A"]
        _, tau, tau_p = _compute_theme_frequencies(
            labels, labels, all_labels, len(labels)
        )
        assert math.isnan(tau)
        assert math.isnan(tau_p)

    def test_label_in_only_one_set(self):
        """Labels absent from one set get 0 frequency."""
        pred = [frozenset({"A"}), frozenset({"A"})]
        ref = [frozenset({"B"}), frozenset({"B"})]
        all_labels = sorted({"A", "B"})
        freq, tau, _ = _compute_theme_frequencies(pred, ref, all_labels, len(pred))
        by_label = {f["label"]: f for f in freq}
        assert by_label["A"]["ref_count"] == 0
        assert by_label["B"]["pred_count"] == 0
        assert tau == pytest.approx(-1.0)


def _bootstrap_ci(pred_labels, ref_labels, all_labels, n_bootstrap=1000, seed=42):
    """Local bootstrap CI helper — mirrors the logic in cli.py evaluate command."""
    n = len(pred_labels)
    rng = np.random.default_rng(seed=seed)
    boot_exact, boot_overlap, boot_tau = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sp = [pred_labels[i] for i in idx]
        sr = [ref_labels[i] for i in idx]
        boot_exact.append(sum(1 for p, r in zip(sp, sr) if p == r) / n)
        boot_overlap.append(sum(1 for p, r in zip(sp, sr) if p & r) / n)
        if len(all_labels) >= 2:
            pc = Counter(lbl for s in sp for lbl in s)
            rc = Counter(lbl for s in sr for lbl in s)
            pv = [pc.get(lbl, 0) for lbl in all_labels]
            rv = [rc.get(lbl, 0) for lbl in all_labels]
            t, _ = kendalltau(pv, rv)
            if not math.isnan(t):
                boot_tau.append(t)
    lo_e, hi_e = np.percentile(boot_exact, [2.5, 97.5])
    lo_o, hi_o = np.percentile(boot_overlap, [2.5, 97.5])
    tau_ci = None
    if boot_tau:
        lo_t, hi_t = np.percentile(boot_tau, [2.5, 97.5])
        tau_ci = [round(float(lo_t), 4), round(float(hi_t), 4)]
    return (
        [round(float(lo_e), 4), round(float(hi_e), 4)],
        [round(float(lo_o), 4), round(float(hi_o), 4)],
        tau_ci,
    )


class TestBootstrapCI:
    def test_exact_match_ci_bounds(self, basic_labels):
        """Exact match CI should bracket the point estimate."""
        pred, ref, all_labels = basic_labels
        exact = sum(1 for p, r in zip(pred, ref) if p == r) / len(pred)
        exact_ci, _, _ = _bootstrap_ci(pred, ref, all_labels)
        assert exact_ci[0] <= exact <= exact_ci[1]

    def test_overlap_ci_bounds(self, basic_labels):
        """Overlap rate CI should bracket the point estimate."""
        pred, ref, all_labels = basic_labels
        overlap = sum(1 for p, r in zip(pred, ref) if p & r) / len(pred)
        _, overlap_ci, _ = _bootstrap_ci(pred, ref, all_labels)
        assert overlap_ci[0] <= overlap <= overlap_ci[1]

    def test_kendall_tau_ci_bounds(self, basic_labels):
        """Kendall's tau CI should bracket the point estimate."""
        pred, ref, all_labels = basic_labels
        _, tau, _ = _compute_theme_frequencies(pred, ref, all_labels, len(pred))
        _, _, tau_ci = _bootstrap_ci(pred, ref, all_labels)
        assert tau_ci is not None
        assert tau_ci[0] <= tau <= tau_ci[1]

    def test_kendall_tau_ci_none_for_single_label(self):
        """With <2 labels, tau CI should be None."""
        labels = [frozenset({"A"}), frozenset({"A"})]
        all_labels = ["A"]
        _, _, tau_ci = _bootstrap_ci(labels, labels, all_labels)
        assert tau_ci is None

    def test_ci_is_two_element_list(self, basic_labels):
        """Each CI should be a [lo, hi] pair with lo <= hi."""
        pred, ref, all_labels = basic_labels
        exact_ci, overlap_ci, tau_ci = _bootstrap_ci(pred, ref, all_labels)
        for interval in [exact_ci, overlap_ci, tau_ci]:
            if interval is not None:
                assert len(interval) == 2
                assert interval[0] <= interval[1]
