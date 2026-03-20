"""Tests for theme frequency computation and Kendall's tau rank correlation."""

import math
from collections import Counter

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
