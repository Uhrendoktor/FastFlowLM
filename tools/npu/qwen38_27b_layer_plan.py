#!/usr/bin/env python3
"""Derive the Qwen3.8-27B hybrid layer plan from its actual geometry.

The plan is intentionally separate from the historical dense Qwen3 full-layer
runner. It is the contract the eventual AIE generator must consume.
"""
from __future__ import annotations

import json
from pathlib import Path

HIDDEN = 5120
FFN = 17408
VOCAB = 248320
LAYERS = 64
FULL_INTERVAL = 4
FULL_Q = 24
FULL_KV = 4
FULL_HD = 256
LINEAR_QK = 16
LINEAR_V = 48
LINEAR_HD = 128
CONV_K = 4


def layer_kind(i: int) -> str:
    return "full_attention" if (i + 1) % FULL_INTERVAL == 0 else "linear_attention"


def make_plan() -> dict:
    layers = []
    for i in range(LAYERS):
        kind = layer_kind(i)
        if kind == "full_attention":
            q_width = FULL_Q * FULL_HD
            kv_width = FULL_KV * FULL_HD
            layers.append({
                "index": i,
                "kind": kind,
                "q_heads": FULL_Q,
                "kv_heads": FULL_KV,
                "head_dim": FULL_HD,
                "q_width": q_width,
                "q_gate_width": q_width * 2,
                "k_width": kv_width,
                "v_width": kv_width,
                "attn_output_width": HIDDEN,
                "mlp_gate_up_width": FFN * 2,
                "mlp_down_in": FFN,
                "mlp_down_out": HIDDEN,
                "rotary_dim": 64,
            })
        else:
            qk_width = LINEAR_QK * LINEAR_HD
            value_width = LINEAR_V * LINEAR_HD
            qkv_width = qk_width + qk_width + value_width
            layers.append({
                "index": i,
                "kind": kind,
                "q_heads": LINEAR_QK,
                "k_heads": LINEAR_QK,
                "v_heads": LINEAR_V,
                "head_dim": LINEAR_HD,
                "qk_width": qk_width,
                "value_width": value_width,
                "qkv_width": qkv_width,
                "z_width": value_width,
                "qkvz_width": qkv_width + value_width,
                "conv_channels": qkv_width,
                "conv_kernel": CONV_K,
                "a_projection_width": LINEAR_V,
                "b_projection_width": LINEAR_V,
                "attn_output_width": HIDDEN,
                "mlp_gate_up_width": FFN * 2,
                "mlp_down_in": FFN,
                "mlp_down_out": HIDDEN,
            })
    return {
        "model": "qwen3.8:27b",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS,
        "full_attention_interval": FULL_INTERVAL,
        "layer_types": [x["kind"] for x in layers],
        "layers": layers,
    }


def validate(plan: dict) -> None:
    assert plan["hidden_size"] == HIDDEN
    assert plan["intermediate_size"] == FFN
    assert plan["vocab_size"] == VOCAB
    assert len(plan["layers"]) == LAYERS
    full = [x for x in plan["layers"] if x["kind"] == "full_attention"]
    linear = [x for x in plan["layers"] if x["kind"] == "linear_attention"]
    assert len(full) == 16 and len(linear) == 48
    assert [x["index"] for x in full] == list(range(3, 64, 4))
    for x in full:
        assert x["q_width"] == 6144
        assert x["q_gate_width"] == 12288
        assert x["k_width"] == 1024 and x["v_width"] == 1024
        assert x["mlp_gate_up_width"] == 34816
    for x in linear:
        assert x["qk_width"] == 2048
        assert x["value_width"] == 6144
        assert x["qkv_width"] == 10240
        assert x["z_width"] == 6144
        assert x["qkvz_width"] == 16384
        assert x["conv_channels"] == 10240
        assert x["conv_kernel"] == 4
        assert x["a_projection_width"] == 48
        assert x["b_projection_width"] == 48
        assert x["mlp_gate_up_width"] == 34816


def main() -> int:
    plan = make_plan()
    validate(plan)
    out = Path(__file__).with_name("qwen38_27b_layer_plan.json")
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Qwen3.8:27b hybrid plan PASS: {out}")
    print("full layers:", [x["index"] for x in plan["layers"] if x["kind"] == "full_attention"])
    print("linear layers:", [x["index"] for x in plan["layers"] if x["kind"] == "linear_attention"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
