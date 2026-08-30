#!/usr/bin/env python3
"""Lightweight compatibility checks for the Qwen3.8-27B FastFlowLM port.

The official Qwen3.8-27B config advertises the Transformers architecture
Qwen3_5ForConditionalGeneration, so FastFlowLM reuses its Qwen3.5 VLM runtime.
Keep this test dependency-free so it can run in CI without installing
Transformers.
"""

import json
import pathlib
import urllib.request

MODEL_CONFIG = "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json"
ROOT = pathlib.Path(__file__).resolve().parents[1]
ALL_MODELS = ROOT / "src/include/AutoModel/all_models.hpp"


def load_config():
    with urllib.request.urlopen(MODEL_CONFIG, timeout=30) as response:
        return json.load(response)


def test_qwen38_architecture():
    config = load_config()
    assert "Qwen3_5ForConditionalGeneration" in config["architectures"]
    assert config["model_type"] == "qwen3_5"

    text = config["text_config"]
    assert text["hidden_size"] == 5120
    assert text["num_hidden_layers"] == 64
    assert text["intermediate_size"] == 17408
    assert text["head_dim"] == 256
    assert text["num_attention_heads"] == 24
    assert text["num_key_value_heads"] == 4
    assert text["full_attention_interval"] == 4
    assert text["linear_num_key_heads"] == 16
    assert text["linear_num_value_heads"] == 48
    assert text["linear_key_head_dim"] == 128
    assert text["linear_value_head_dim"] == 128
    assert text["vocab_size"] == 248320

    layer_types = text["layer_types"]
    assert len(layer_types) == 64
    assert all(layer_types[i : i + 4] == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ] for i in range(0, 64, 4))

    vision = config["vision_config"]
    assert vision["hidden_size"] == 1152
    assert vision["depth"] == 27
    assert vision["patch_size"] == 16
    assert vision["spatial_merge_size"] == 2
    assert vision["temporal_patch_size"] == 2
    assert vision["out_hidden_size"] == 5120


def test_fastflowlm_routes_qwen38_to_qwen35_runtime():
    source = ALL_MODELS.read_text(encoding="utf-8")
    assert '{"qwen3.8", SupportedModelFamily::qwen3_8}' in source
    assert "case SupportedModelFamily::qwen3_8:" in source
    assert "std::make_unique<Qwen3_5VL>(npu_device_inst)" in source


if __name__ == "__main__":
    test_qwen38_architecture()
    test_fastflowlm_routes_qwen38_to_qwen35_runtime()
    print("Qwen3.8-27B compatibility checks passed")
