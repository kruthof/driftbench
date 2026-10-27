"""Tests for the mechanism hand-check LLM annotator (X9). Synthetic data only, no network."""

import json
import math
import socket

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import mechanism_handcheck as MH
from drift_bench.analysis.camera_ready.typology_annotate import codebook_sha256


# --- label parsing -----------------------------------------------------------


def test_parse_valid_answer():
    content = '{"appears": "yes", "verdict": "compliant", "departs": "yes", "rationale": "Ticks it but runs 4 arms."}'
    p = MH.parse_response(content)
    assert p["parse_ok"] and p["consistent"]
    assert (p["appears"], p["verdict"], p["departs"]) == ("yes", "compliant", "yes")
    assert p["rationale"] == "Ticks it but runs 4 arms."


def test_parse_tolerates_prose_and_spelling_variants():
    content = 'Here: {"appears": "No Checklist", "verdict": "Absent", "departs": "Unclear", "rationale": "cut off"} done'
    p = MH.parse_response(content)
    assert p["parse_ok"]
    assert (p["appears"], p["verdict"], p["departs"]) == ("no_checklist", "absent", "unclear")
    p2 = MH.parse_response('{"appears": "yes", "verdict": "flagged non-compliant", "departs": "no", "rationale": "x"}')
    assert p2["verdict"] == "flagged_noncompliant" and p2["parse_ok"]


def test_parse_rejects_bad_json_and_unknown_labels():
    assert not MH.parse_response("no json here")["parse_ok"]
    assert not MH.parse_response('{"appears": "yes", ')["parse_ok"]
    p = MH.parse_response('{"appears": "maybe", "verdict": "compliant", "departs": "yes", "rationale": "x"}')
    assert not p["parse_ok"] and p["appears"] == "" and "appears" in p["parse_error"]


def test_parse_flags_appears_verdict_rule_break():
    p = MH.parse_response('{"appears": "no", "verdict": "compliant", "departs": "yes", "rationale": "x"}')
    assert p["parse_ok"] and not p["consistent"]
    assert MH.is_consistent("yes", "hedged")
    assert MH.is_consistent("no_checklist", "absent")
    assert not MH.is_consistent("yes", "absent")


# --- detector and judge mapping ------------------------------------------------


def test_detector_appears_mapping():
    assert MH.detector_appears("True", "True") == "yes"
    assert MH.detector_appears("True", "False") == "no"
    assert MH.detector_appears("False", "False") == "no_checklist"
    assert MH.detector_appears(True, False) == "no"


def test_detector_verdict_and_judge_mapping():
    assert MH.detector_verdict("flagged") == "flagged_noncompliant"
    assert MH.detector_verdict("unflagged") == "compliant"
    assert MH.detector_verdict("") == "absent"
    assert MH.detector_verdict(float("nan")) == "absent"
    assert MH.judge_departs("Stretched") == "yes"
    assert MH.judge_departs("violated") == "yes"
    assert MH.judge_departs("satisfied") == "no"
    with pytest.raises(ValueError):
        MH.detector_verdict("weird")
    with pytest.raises(ValueError):
        MH.judge_departs("")


def test_constraint_sha_ignores_whitespace():
    assert MH.constraint_sha("budget  under\nGBP 25k") == MH.constraint_sha(" budget under GBP 25k ")
    assert len(MH.constraint_sha("x")) == 12
    assert MH.row_key("r1", "a") != MH.row_key("r1", "b")


# --- agreement math -------------------------------------------------------------


def test_cohen_kappa_textbook_example():
    # 2x2 table: yes/yes 20, yes/no 5, no/yes 10, no/no 15 (n = 50).
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    # p_o = 0.7; p_e = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.4.
    assert MH.cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edge_cases():
    assert MH.cohen_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert math.isnan(MH.cohen_kappa([], []))
    assert math.isnan(MH.cohen_kappa(["yes"] * 3, ["yes"] * 3))
    # One constant rater: kappa is 0 by construction.
    assert MH.cohen_kappa(["yes"] * 4, ["yes", "no", "yes", "no"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        MH.cohen_kappa(["a"], [])


def test_agreement_row_drops_missing_and_counts():
    row = MH.agreement_row("appears", "detector", "llm", ["yes", "no", "yes", "no_checklist"],
                           ["yes", "", "no", "no_checklist"], 4)
    assert (row["n_rows"], row["n"], row["n_missing"], row["n_agree"]) == (4, 3, 1, 2)
    assert row["frac_agree"] == pytest.approx(2 / 3)
    assert row["a_n_yes"] == 2 and row["b_n_no"] == 1 and row["b_n_unclear"] == 0
    assert "yes->no: 1" in row["confusion"]


def test_agreement_row_notes_constant_rater():
    row = MH.agreement_row("departs", "judge", "llm", ["yes"] * 3, ["yes", "no", "unclear"], 3)
    assert row["n_agree"] == 1 and row["kappa"] == pytest.approx(0.0)
    assert "judge gives one label" in row["kappa_note"]


# --- run() from synthetic files, no network -----------------------------------


def _sample() -> pd.DataFrame:
    rows = [
        # detector: yes / compliant; judge stretched
        dict(run_id="r1", brief_id="b1", model_id="m", condition="multi_turn_pressure", constraint="budget under 10k",
             judge_status="stretched", checklist_found="True", appears_in_final_checklist="True", verdict="compliant"),
        # detector: no checklist / absent; judge violated
        dict(run_id="r2", brief_id="b1", model_id="m", condition="multi_turn_pressure", constraint="n at least 100",
             judge_status="violated", checklist_found="False", appears_in_final_checklist="False", verdict=""),
        # detector: yes / unflagged -> compliant
        dict(run_id="r2", brief_id="b1", model_id="m", condition="multi_turn_pressure", constraint="budget under 10k",
             judge_status="stretched", checklist_found="False", appears_in_final_checklist="True", verdict="unflagged"),
        # no LLM record for this row
        dict(run_id="r3", brief_id="b1", model_id="m", condition="checkpointed_pressure", constraint="no imaging",
             judge_status="stretched", checklist_found="True", appears_in_final_checklist="False", verdict=""),
    ]
    df = pd.DataFrame(rows)
    df["human_appears"] = ""
    df["human_verdict"] = ""
    df["human_note"] = ""
    return df


def _rec(run_id, constraint, appears, verdict, departs, sha, **extra):
    content = json.dumps({"appears": appears, "verdict": verdict, "departs": departs, "rationale": f"{run_id} r"})
    return {"run_id": run_id, "constraint_sha": MH.constraint_sha(constraint), "codebook_sha256": sha,
            "content": content, "cost_usd": 0.01, **extra}


@pytest.fixture
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def test_run_builds_tables_from_raw_responses_without_network(tmp_path, no_network):
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()
    sample = _sample()
    sample.to_csv(out_dir / MH.SAMPLE_NAME, index=False)
    raw = tmp_path / MH.RAW_SUBDIR
    raw.mkdir(parents=True)
    sha = codebook_sha256(MH.CODEBOOK_PATH.read_text())
    recs = [
        _rec("r1", "budget under 10k", "yes", "compliant", "yes", "0" * 64),  # stale codebook: ignored
        _rec("r1", "budget under 10k", "yes", "hedged", "unclear", sha),
        _rec("r2", "n at least 100", "no_checklist", "absent", "yes", sha),
        _rec("r2", "budget under 10k", "no", "absent", "no", sha),
        _rec("r3", "no imaging", "yes", "compliant", "yes", sha, error="RateLimitError: x"),  # error: ignored
    ]
    recs.append({**_rec("r3", "no imaging", "yes", "compliant", "yes", sha), "content": "not json"})  # unparseable
    (raw / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    out = MH.run(tmp_path, out_dir)
    table = out["mechanism_handcheck_llm"]
    assert len(table) == 4
    assert list(table.columns[: len(sample.columns)]) == list(sample.columns)
    assert table.loc[0, ["llm_appears", "llm_verdict", "llm_departs"]].tolist() == ["yes", "hedged", "unclear"]
    assert table.loc[3, "llm_appears"] == ""  # error and unparseable records give no label

    agree = out["mechanism_handcheck_agreement"].set_index("comparison")
    # appears: detector yes/no_checklist/yes/no vs llm yes/no_checklist/no/-- -> 2 of 3 agree.
    assert agree.loc["appears", "n"] == 3 and agree.loc["appears", "n_agree"] == 2
    assert agree.loc["appears", "n_missing"] == 1
    # verdict: detector compliant/absent/compliant vs llm hedged/absent/absent -> 1 of 3.
    assert agree.loc["verdict", "n_agree"] == 1
    # both-appear subset: only r1.
    assert agree.loc["verdict_both_appear", "n"] == 1
    # departs: judge yes on all; llm unclear/yes/no -> 1 of 3.
    assert agree.loc["departs", "n_agree"] == 1
    assert agree.loc["departs", "b_n_unclear"] == 1 and agree.loc["departs", "b_n_no"] == 1
    assert agree.loc["departs", "pct_agree"] == pytest.approx(100 / 3)
    # No human rows while the human columns are empty.
    assert not any(c.startswith(("appears_human", "verdict_human")) for c in agree.index)
    for name in (MH.LLM_CSV, MH.AGREEMENT_CSV):
        assert (out_dir / name).exists()
        prov = json.loads((out_dir / f"{name}.provenance.json").read_text())
        assert prov["rows_out"] in (4, len(agree))
    # The sample file itself is untouched.
    pd.testing.assert_frame_equal(pd.read_csv(out_dir / MH.SAMPLE_NAME, dtype=str, keep_default_na=False),
                                  sample.astype(str))


def test_run_adds_human_rows_when_filled(tmp_path, no_network):
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()
    sample = _sample()
    sample.loc[0, "human_appears"] = "yes"
    sample.loc[1, "human_appears"] = "No checklist"
    sample.to_csv(out_dir / MH.SAMPLE_NAME, index=False)
    raw = tmp_path / MH.RAW_SUBDIR
    raw.mkdir(parents=True)
    sha = codebook_sha256(MH.CODEBOOK_PATH.read_text())
    recs = [_rec("r1", "budget under 10k", "yes", "compliant", "yes", sha),
            _rec("r2", "n at least 100", "no", "absent", "yes", sha)]
    (raw / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    agree = MH.run(tmp_path, out_dir)["mechanism_handcheck_agreement"].set_index("comparison")
    assert agree.loc["appears_human_vs_llm", "n"] == 2
    assert agree.loc["appears_human_vs_llm", "n_agree"] == 1
    assert agree.loc["appears_human_vs_detector", "n_agree"] == 2
    assert "verdict_human_vs_llm" not in agree.index


def test_labels_from_records_last_valid_wins():
    sha = "s" * 64
    recs = [_rec("r1", "c", "yes", "compliant", "yes", sha), _rec("r1", "c", "no", "absent", "no", sha)]
    labels = MH.labels_from_records(recs, sha)
    assert labels[MH.row_key("r1", "c")]["appears"] == "no"


def test_build_prompt_carries_inputs_but_no_detector_output():
    system, user = MH.build_prompt("CODEBOOK", "obj", "budget under 10k", ["n >= 100"], "FINAL TEXT")
    assert system.endswith("CODEBOOK")
    assert "budget under 10k" in user and "- n >= 100" in user and "FINAL TEXT" in user
    for leaked in ("stretched", "violated", "detector", "judge"):
        assert leaked not in user.lower()


# --- X9 fix: judge arms, model overlap, README notes ---------------------------

SONNET = "anthropic/claude-sonnet-4-6"
GPT = "openai/gpt-5.4"
OPUS = "anthropic/claude-opus-4-6"
FLASH = "gemini/gemini-3.1-flash-lite-preview"


def _arm_table() -> pd.DataFrame:
    """Five departures: three Sonnet rows (GPT-5.4 arm), two Flash-Lite rows (Opus arm)."""
    rows = [
        dict(run_id="s1", model_id=SONNET, constraint="c1", llm_departs="no"),
        dict(run_id="s1", model_id=SONNET, constraint="c2", llm_departs="no"),
        dict(run_id="s2", model_id=SONNET, constraint="c1", llm_departs="yes"),
        dict(run_id="f1", model_id=FLASH, constraint="c1", llm_departs="unclear"),
        dict(run_id="f2", model_id=FLASH, constraint="c1", llm_departs="no"),
    ]
    df = pd.DataFrame(rows)
    df["brief_id"] = "b1"
    df["condition"] = "multi_turn_pressure"
    df["judge_status"] = "stretched"
    df["checklist_found"] = "True"
    df["appears_in_final_checklist"] = "True"
    df["verdict"] = "compliant"
    df["llm_appears"] = "yes"
    df["llm_verdict"] = ["compliant", "hedged", "compliant", "flagged_noncompliant", "compliant"]
    return df


def test_model_family_and_design_judge():
    assert MH.model_family(OPUS) == MH.model_family(SONNET) == "anthropic"
    assert MH.model_family("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo") == "together_ai"
    assert MH.design_judge(SONNET) == GPT
    assert MH.design_judge(FLASH) == OPUS and MH.design_judge("openai/gpt-5.4-mini") == OPUS


def test_judge_models_for_reads_departures_and_falls_back_to_design():
    t = _arm_table()
    design, info = MH.judge_models_for(t, None)
    assert design == [GPT, GPT, GPT, OPUS, OPUS]
    assert "design rule" in info["source"] and info["rows_differing_from_design_rule"] == 0
    dep = pd.DataFrame({"run_id": ["s1", "s1", "s2", "f1", "f2", "other"],
                        "judge_model": [GPT, GPT, GPT, OPUS, GPT, OPUS]})
    got, info = MH.judge_models_for(t, dep)
    assert got == [GPT, GPT, GPT, OPUS, GPT]  # the file wins over the design rule
    assert info["rows_differing_from_design_rule"] == 1 and MH.DEPARTURES_NAME in info["source"]
    with pytest.raises(ValueError, match="not in"):
        MH.judge_models_for(t, dep[dep["run_id"] != "f2"])
    with pytest.raises(ValueError, match="judge models"):
        MH.judge_models_for(t, pd.concat([dep, pd.DataFrame({"run_id": ["s2"], "judge_model": [OPUS]})]))


def test_model_overlap_counts_and_note():
    t = _arm_table()
    ov = MH.model_overlap(t, [GPT, GPT, GPT, OPUS, OPUS])
    assert ov["rows"] == 5 and ov["annotator"] == MH.ANNOTATOR_MODEL == OPUS
    assert ov["rows_annotator_is_judge"] == 2 and ov["subjects_annotator_is_judge"] == [FLASH]
    assert ov["rows_annotator_same_family_as_subject"] == 3 and ov["subjects_same_family"] == [SONNET]
    assert ov["judges_on_same_family_rows"] == [GPT]
    assert (ov["llm_departs_no"], ov["llm_departs_no_on_same_family_rows"]) == (3, 2)
    note = MH.overlap_note(ov)
    assert "on 2 of 5 rows" in note and "same model as the structured judge" in note
    assert "on 3 of 5 rows" in note and "2 of the 3 LLM 'no' labels" in note
    with pytest.raises(ValueError):
        MH.model_overlap(t, [GPT])
    none = MH.model_overlap(t.assign(model_id=FLASH), [GPT] * 5)
    assert MH.overlap_note(none) == "model overlap: none"


def test_agreement_table_splits_departs_by_judge_arm():
    t = _arm_table()
    agree = MH.agreement_table(t, [GPT, GPT, GPT, OPUS, OPUS]).set_index("comparison")
    assert list(agree.index[-3:]) == ["departs", "departs_gpt_arm", "departs_opus_arm"]
    g, o = agree.loc["departs_gpt_arm"], agree.loc["departs_opus_arm"]
    assert (g["n"], g["b_n_yes"], g["b_n_no"], g["b_n_unclear"]) == (3, 1, 2, 0)
    assert (o["n"], o["b_n_yes"], o["b_n_no"], o["b_n_unclear"]) == (2, 0, 1, 1)
    assert g["n_agree"] == 1 and o["n_agree"] == 0  # n_agree is the LLM yes count
    assert "own model family" in g["note"] and "same model as this judge" not in g["note"]
    assert "same model as this judge" in o["note"] and "own model family" not in o["note"]
    assert "not an agreement measure" in agree.loc["departs", "note"]
    assert "model overlap" in agree.loc["departs", "note"]
    # The verdict row counts the flagged-as-unmet departures (one here).
    assert agree.loc["verdict", "b_n_flagged_noncompliant"] == 1
    # Without judge models: no arm rows, no overlap clause.
    plain = MH.agreement_table(t).set_index("comparison")
    assert "departs_gpt_arm" not in plain.index and "model overlap" not in plain.loc["departs", "note"]
    with pytest.raises(ValueError, match="no judge arm"):
        MH.agreement_table(t, [GPT, GPT, GPT, OPUS, "x/judge"])


def test_agreement_table_skips_an_empty_arm():
    t = _arm_table().iloc[3:].reset_index(drop=True)
    agree = MH.agreement_table(t, [OPUS, OPUS]).set_index("comparison")
    assert "departs_opus_arm" in agree.index and "departs_gpt_arm" not in agree.index


def test_update_readme_notes_replaces_or_appends(tmp_path):
    readme = tmp_path / "README.md"
    MH.update_readme_notes(readme, [MH.README_BEGIN, "one", MH.README_END])
    assert readme.read_text() == f"{MH.README_BEGIN}\none\n{MH.README_END}\n"
    readme.write_text("# Title\n\nbody\n")
    MH.update_readme_notes(readme, [MH.README_BEGIN, "one", MH.README_END])
    MH.update_readme_notes(readme, [MH.README_BEGIN, "two", MH.README_END])
    text = readme.read_text()
    assert text.startswith("# Title\n\nbody\n\n") and text.count(MH.README_BEGIN) == 1
    assert "two" in text and "one" not in text


def test_review_notes_cite_the_gap_rows_with_current_labels():
    t = _arm_table()
    extra = t.iloc[:3].copy()
    extra["run_id"] = [MH.GAP_AT_CAP[0], MH.GAP_INCONSISTENT[0][0], MH.GAP_INCONSISTENT[1][0]]
    extra["constraint"] = [MH.GAP_AT_CAP[1], MH.GAP_INCONSISTENT[0][1], MH.GAP_INCONSISTENT[1][1]]
    extra["llm_departs"] = ["no", "unclear", "yes"]
    t = pd.concat([t, extra], ignore_index=True)
    judges = [MH.design_judge(m) for m in t["model_id"]]
    agree = MH.agreement_table(t, judges)
    lines = MH.review_notes(t, agree, MH.model_overlap(t, judges), 8, MH.REVIEWED_CODEBOOK_SHA, [])
    text = "\n".join(lines)
    assert lines[0] == MH.README_BEGIN and lines[-1] == MH.README_END
    assert f"row 5 (run `{MH.GAP_AT_CAP[0]}`" in text and "the label is `no`" in text and "P11" in text
    assert "Row 6 (run" in text and "row 7 (run" in text and "The labels are `unclear` and `yes`." in text
    assert "Rule 1 of (c)" in text and "not re-run" in text and "Archived passes made with another codebook: none" in text
    assert "openai/gpt-5.4: 2 / 3 / 1 of 6" in text  # the six Sonnet rows: yes / no / unclear
    assert "were written for codebook" not in text  # current codebook == reviewed codebook
    stale = "\n".join(MH.review_notes(t, agree, MH.model_overlap(t, judges), 8, "f" * 64, []))
    assert "were written for codebook sha256 `ce1d7855`" in stale
    # Without the gap rows the gap statements stay, the row citations go.
    base = "\n".join(MH.review_notes(t.iloc[:5], agree, MH.model_overlap(t.iloc[:5], judges[:5]), 5,
                                     MH.REVIEWED_CODEBOOK_SHA, []))
    assert "No rule covers a value exactly at a strict" in base and "run `f6e6ea06c07d`" not in base


def test_run_writes_judge_arms_overlap_and_readme(tmp_path, no_network):
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()
    sample = _arm_table().drop(columns=["llm_appears", "llm_verdict", "llm_departs"])
    for col in ("human_appears", "human_verdict", "human_note"):
        sample[col] = ""
    sample.to_csv(out_dir / MH.SAMPLE_NAME, index=False)
    pd.DataFrame({"run_id": ["s1", "s1", "s2", "f1", "f2"], "judge_model": [GPT, GPT, GPT, OPUS, OPUS],
                  "brief_id": "b1"}).to_csv(out_dir / MH.DEPARTURES_NAME, index=False)
    raw = tmp_path / MH.RAW_SUBDIR
    raw.mkdir(parents=True)
    (raw / "README.md").write_text("# E9 header\n\nkept text\n")
    sha = codebook_sha256(MH.CODEBOOK_PATH.read_text())
    labels = _arm_table()
    recs = [_rec(r.run_id, r.constraint, "yes", "compliant", r.llm_departs, sha) for r in labels.itertuples()]
    (raw / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    agree = MH.run(tmp_path, out_dir)["mechanism_handcheck_agreement"].set_index("comparison")
    assert (agree.loc["departs_gpt_arm", "n"], agree.loc["departs_gpt_arm", "b_n_no"]) == (3, 2)
    assert (agree.loc["departs_opus_arm", "n"], agree.loc["departs_opus_arm", "b_n_unclear"]) == (2, 1)
    prov = json.loads((out_dir / f"{MH.AGREEMENT_CSV}.provenance.json").read_text())
    assert prov["model_overlap"]["rows_annotator_is_judge"] == 2
    assert prov["model_overlap"]["llm_departs_no_on_same_family_rows"] == 2
    assert prov["counts"]["rows_per_judge"] == {OPUS: 2, GPT: 3}
    assert any(MH.DEPARTURES_NAME in i for i in prov["inputs"]) and MH.DEPARTURES_NAME in prov["filters"]["judge_model"]
    text = (raw / "README.md").read_text()
    assert text.startswith("# E9 header\n\nkept text\n\n") and "## Model overlap (disclosure)" in text
    assert "## Codebook gaps" in text and "2 of the 3 `departs = no` labels" in text
    # A second run leaves the README unchanged.
    MH.run(tmp_path, out_dir)
    assert (raw / "README.md").read_text() == text


def test_write_readme_keeps_the_review_notes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "responses.jsonl").write_text("")
    (raw / "README.md").write_text(f"old header\n\n{MH.README_BEGIN}\nnotes\n{MH.README_END}\n")
    MH.write_readme(raw, "s" * 64, OPUS, 5.0, 4.0, "", {})
    text = (raw / "README.md").read_text()
    assert text.startswith("# E9: LLM annotator") and "old header" not in text
    assert f"{MH.README_BEGIN}\nnotes\n{MH.README_END}" in text
    assert "do not measure agreement" in text
