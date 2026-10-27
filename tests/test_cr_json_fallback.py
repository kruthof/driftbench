"""The judge JSON fallback must take the first complete object, not a greedy span."""

import json

import pytest

from drift_bench.judges.judge import first_json_object


def test_trailing_prose_after_object_is_ignored():
    text = '{"a": 1, "checks": [{"x": "y"}]}\n\nNote: I also considered {more}.'
    assert first_json_object(text) == {"a": 1, "checks": [{"x": "y"}]}


def test_prose_before_object_is_skipped():
    text = 'Here is the result:\n{"a": 2}'
    assert first_json_object(text) == {"a": 2}


def test_no_object_raises():
    with pytest.raises(ValueError):
        first_json_object("no json here {broken")


def test_first_valid_object_wins_over_later_one():
    text = '{"first": true} {"second": true}'
    assert first_json_object(text) == {"first": True}


def test_round_trip_on_clean_json():
    obj = {"objective_fidelity": 4, "constraint_checks": [{"constraint": "c", "status": "satisfied"}]}
    assert first_json_object(json.dumps(obj)) == obj
