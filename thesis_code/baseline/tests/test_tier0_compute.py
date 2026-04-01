from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from baseline.tiers.tier0_compute import ComputeEngine


LOCAL_MODEL_PATH = Path(
    "/Users/ashripal/mem_optim/thesis_code/models/Qwen2.5-0.5B-Instruct"
)


@dataclass
class DummyConfig:
    model_id: str = str(LOCAL_MODEL_PATH)
    max_input_tokens: int = 128
    max_new_tokens: int = 16
    device: str = "cpu"
    dtype: str = "fp32"
    cpu_fallback_on_long: bool = False
    use_fast_tokenizer: bool = True
    attn_implementation: str | None = None
    trust_remote_code: bool = False


def _require_local_model() -> None:
    if not LOCAL_MODEL_PATH.exists():
        pytest.skip(f"Local model path does not exist: {LOCAL_MODEL_PATH}")


def test_compute_engine_loads_local_model_cpu():
    _require_local_model()

    cfg = DummyConfig(device="cpu", dtype="fp32")
    engine = ComputeEngine(cfg)

    assert engine is not None
    assert str(engine.model_id) == str(LOCAL_MODEL_PATH)
    assert engine.active_device == "cpu"
    assert engine.tokenizer is not None
    assert engine.model is not None


def test_generate_returns_expected_fields():
    _require_local_model()

    cfg = DummyConfig(
        device="cpu",
        dtype="fp32",
        max_input_tokens=128,
        max_new_tokens=16,
    )
    engine = ComputeEngine(cfg)

    out = engine.generate(
        prompt="What is the capital of France?",
        max_input_tokens=cfg.max_input_tokens,
        max_new_tokens=cfg.max_new_tokens,
    )

    assert isinstance(out, dict)
    assert out.get("ok") is True

    # Required keys based on current baseline/tier0 behavior
    required_keys = [
        "ok",
        "device",
        "dtype",
        "gen_time_s",
        "output_text",
    ]
    for key in required_keys:
        assert key in out, f"Missing key: {key}. Full output: {out}"

    assert isinstance(out["output_text"], str)
    assert isinstance(out["gen_time_s"], (int, float))
    assert out["gen_time_s"] >= 0
    assert out["device"] == "cpu"
    assert out["dtype"] in {"fp32", "float32", "none"} or isinstance(out["dtype"], str)

    # Optional keys: only validate type if present
    if "input_tokens" in out:
        assert isinstance(out["input_tokens"], int)
        assert out["input_tokens"] > 0

    if "output_tokens" in out:
        assert isinstance(out["output_tokens"], int)
        assert out["output_tokens"] >= 0

    if "truncated" in out:
        assert isinstance(out["truncated"], bool)

    if "generation_backend" in out:
        assert isinstance(out["generation_backend"], str)


def test_generate_long_prompt_runs_successfully():
    _require_local_model()

    cfg = DummyConfig(
        device="cpu",
        dtype="fp32",
        max_input_tokens=32,
        max_new_tokens=8,
    )
    engine = ComputeEngine(cfg)

    long_prompt = " ".join(["token"] * 2000)

    out = engine.generate(
        prompt=long_prompt,
        max_input_tokens=cfg.max_input_tokens,
        max_new_tokens=cfg.max_new_tokens,
    )

    assert out.get("ok") is True
    assert "output_text" in out
    assert isinstance(out["output_text"], str)

    # If truncation metadata exists, validate it.
    if "input_tokens" in out:
        assert out["input_tokens"] <= cfg.max_input_tokens
    if "truncated" in out:
        assert out["truncated"] is True


def test_generate_uses_local_model_path_not_hub_id():
    _require_local_model()

    cfg = DummyConfig()
    engine = ComputeEngine(cfg)

    assert Path(engine.model_id).resolve() == LOCAL_MODEL_PATH.resolve()