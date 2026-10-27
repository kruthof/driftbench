"""Tests for drift_bench.analysis.camera_ready.constraint_count.

Synthetic briefs. No network, no API keys.
"""

from __future__ import annotations

import numpy as np

from drift_bench.analysis.camera_ready import constraint_count as cc


BRIEF = {
    "id": "test_01",
    "domain": "energy systems",  # 2 words
    "objective": "Assess whether a thing reduces another thing.",  # 7 words
    "hard_constraints": ["must use at least 3 sites", "exclude pilots"],  # 6 + 2
    "success_criteria": ["reports a confidence interval"],  # 4
    "plausible_directions": ["difference in differences", "matching", "iv"],  # 3 + 1 + 1
    "banned_moves": ["add rooftop solar", "include interviews", "x"],  # 3 + 2 + 1
}


def test_brief_features_counts_and_word_counts():
    f = cc.brief_features(BRIEF)
    assert f["brief_id"] == "test_01"
    assert f["domain"] == "energy systems"
    assert f["n_hard_constraints"] == 2
    assert f["n_banned_moves"] == 3
    assert f["n_success_criteria"] == 1
    assert f["n_plausible_directions"] == 3
    assert f["objective_words"] == 7
    assert f["hard_constraint_words"] == 8
    assert f["banned_move_words"] == 6
    assert f["brief_words"] == 7 + 2 + 8 + 4 + 5 + 6


def test_brief_features_tolerates_missing_lists():
    f = cc.brief_features({"id": "empty", "objective": "one two"})
    assert f["n_hard_constraints"] == 0
    assert f["n_banned_moves"] == 0
    assert f["hard_constraint_words"] == 0
    assert f["objective_words"] == 2
    assert f["brief_words"] == 2


def test_brief_table_sorted_with_log_words():
    t = cc.brief_table({"b": dict(BRIEF, id="b"), "a": dict(BRIEF, id="a")})
    assert list(t["brief_id"]) == ["a", "b"]
    assert t["log_brief_words"].iloc[0] == np.log(t["brief_words"].iloc[0])
