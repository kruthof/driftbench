"""X4: finish_reason / response_model / temperature / max_tokens / litellm_version.

Old transcripts without the new keys must still load. New runs must
populate them. No network: litellm.acompletion and _call_llm are patched.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import jsonlines
import litellm
import pytest

from drift_bench.runners.base import LITELLM_VERSION, BaseRunner, Message, RunMetadata, Transcript
from drift_bench.runners.single_shot import SingleShotRunner
from drift_bench.runners.multi_turn import MultiTurnRunner


OLD_TRANSCRIPT = {
    "metadata": {
        "run_id": "old1", "brief_id": "b", "model_id": "openai/gpt-5.4",
        "condition": "single_shot", "repetition": 0, "start_time": 1.0, "end_time": 2.0,
        "total_input_tokens": 1, "total_output_tokens": 1, "total_cost_usd": 0.0, "error": None,
    },
    "messages": [
        {"role": "system", "content": "s", "timestamp": 1.0, "token_count": None,
         "latency_ms": None, "is_probe": False},
        {"role": "user", "content": "u", "timestamp": 1.0, "token_count": None,
         "latency_ms": None, "is_probe": False},
        {"role": "assistant", "content": "a", "timestamp": 1.0, "token_count": 5,
         "latency_ms": 10.0, "is_probe": False},
    ],
}


def test_old_transcript_dict_without_new_keys_loads() -> None:
    t = Transcript.from_dict(OLD_TRANSCRIPT)
    assert t.metadata.run_id == "old1"
    assert t.metadata.temperature is None
    assert t.metadata.max_tokens is None
    assert t.metadata.litellm_version is None
    assert len(t.messages) == 3
    assert t.messages[2].finish_reason is None
    assert t.messages[2].response_model is None
    assert t.messages[2].token_count == 5


def test_transcript_dict_with_unknown_keys_loads() -> None:
    d = {
        "metadata": {**OLD_TRANSCRIPT["metadata"], "future_meta": 1},
        "messages": [{**m, "future_msg": "x"} for m in OLD_TRANSCRIPT["messages"]],
    }
    t = Transcript.from_dict(d)
    assert t.metadata.run_id == "old1"
    assert len(t.messages) == 3


def test_direct_constructor_still_accepts_old_keys() -> None:
    """pipeline._find_existing_transcript used Message(**m) / RunMetadata(**meta)."""
    m = Message(**OLD_TRANSCRIPT["messages"][2])
    assert m.finish_reason is None
    meta = RunMetadata(**OLD_TRANSCRIPT["metadata"])
    assert meta.litellm_version is None


def test_new_fields_round_trip(tmp_path: Path) -> None:
    meta = RunMetadata(run_id="n1", brief_id="b", model_id="openai/gpt-5.4", condition="single_shot",
                       temperature=0.7, max_tokens=4096, litellm_version="1.2.3")
    msgs = [Message(role="assistant", content="a", finish_reason="length", response_model="gpt-5.4-2026")]
    path = Transcript(metadata=meta, messages=msgs).save(tmp_path)
    with jsonlines.open(path) as r:
        record = next(iter(r))
    assert record["metadata"]["temperature"] == 0.7
    assert record["metadata"]["max_tokens"] == 4096
    assert record["metadata"]["litellm_version"] == "1.2.3"
    assert record["messages"][0]["finish_reason"] == "length"
    assert record["messages"][0]["response_model"] == "gpt-5.4-2026"
    back = Transcript.from_dict(record)
    assert back.messages[0].finish_reason == "length"


def test_litellm_version_is_recorded() -> None:
    assert LITELLM_VERSION is None or isinstance(LITELLM_VERSION, str)
    assert LITELLM_VERSION, "litellm is installed, so the version must be readable"


def test_call_llm_returns_finish_reason_and_model(config_path: Path, monkeypatch) -> None:
    fake_response = SimpleNamespace(
        model="gpt-5.4-2026-01-01",
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="hello"))],
    )

    async def fake_acompletion(**kwargs):
        return fake_response

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    runner = BaseRunner(config_path)
    result = asyncio.run(runner._call_llm("openai/gpt-5.4", [{"role": "user", "content": "x"}]))
    assert result["content"] == "hello"
    assert result["finish_reason"] == "length"
    assert result["response_model"] == "gpt-5.4-2026-01-01"
    assert result["input_tokens"] == 7 and result["output_tokens"] == 3


def _fake_call_llm(self, model_id, messages, temperature=0.7, max_tokens=4096, response_format=None):
    async def _inner():
        return {
            "content": f"reply to {messages[-1]['content'][:10]}", "input_tokens": 2, "output_tokens": 3,
            "cost": 0.001, "latency_ms": 1.0, "finish_reason": "stop", "response_model": "resp-model-id",
        }
    return _inner()


def test_single_shot_runner_populates_new_fields(config_path: Path, sample_brief: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(SingleShotRunner, "_call_llm", _fake_call_llm)
    runner = SingleShotRunner(config_path)
    t = asyncio.run(runner.run(sample_brief, "openai/gpt-5.4", output_dir=tmp_path))
    assert t.metadata.temperature == runner._get_temperature("openai/gpt-5.4")
    assert t.metadata.max_tokens == runner._get_max_tokens("openai/gpt-5.4")
    assert t.metadata.litellm_version == LITELLM_VERSION
    assistant = [m for m in t.messages if m.role == "assistant"]
    assert assistant and assistant[0].finish_reason == "stop"
    assert assistant[0].response_model == "resp-model-id"
    # The saved file carries the keys and reloads.
    saved = next(tmp_path.glob("*.jsonl"))
    with jsonlines.open(saved) as r:
        record = next(iter(r))
    assert record["metadata"]["litellm_version"] == LITELLM_VERSION
    assert Transcript.from_dict(record).messages[-1].finish_reason == "stop"


def test_multi_turn_runner_populates_probe_fields(config_path: Path, sample_brief: dict, monkeypatch) -> None:
    monkeypatch.setattr(MultiTurnRunner, "_call_llm", _fake_call_llm)
    runner = MultiTurnRunner(config_path)
    t = asyncio.run(runner.run(sample_brief, "openai/gpt-5.4", condition="multi_turn_neutral"))
    probes = [m for m in t.messages if m.is_probe and m.role == "assistant"]
    mains = [m for m in t.messages if not m.is_probe and m.role == "assistant"]
    assert len(mains) == 6 and len(probes) == 5
    assert all(m.finish_reason == "stop" and m.response_model == "resp-model-id" for m in mains + probes)
    assert t.metadata.max_tokens == 4096


def test_cost_fallback_from_config_pricing(config_path: Path, monkeypatch) -> None:
    """When litellm reports no cost, the config `pricing` map estimates it for the cap."""
    fake_response = SimpleNamespace(
        model="gpt-5.4-mini-2026",
        usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="x"))],
    )

    async def fake_acompletion(**kwargs):
        return fake_response

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    runner = BaseRunner(config_path)
    pricing = runner.config["pricing"]["openai/gpt-5.4-mini"]
    expected = 1_000_000 * pricing["input_cost_per_token"] + 1_000_000 * pricing["output_cost_per_token"]

    result = asyncio.run(runner._call_llm("openai/gpt-5.4-mini", [{"role": "user", "content": "x"}]))
    assert result["cost"] == pytest.approx(expected)
    assert result["cost_estimated"] is True
    # A model absent from the pricing map stays at zero and is not marked estimated.
    result2 = asyncio.run(runner._call_llm("together_ai/unknown/model", [{"role": "user", "content": "x"}]))
    assert result2["cost"] == 0.0 and result2["cost_estimated"] is False
    # litellm's own cost wins when present.
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.5)
    result3 = asyncio.run(runner._call_llm("openai/gpt-5.4-mini", [{"role": "user", "content": "x"}]))
    assert result3["cost"] == 0.5 and result3["cost_estimated"] is False
