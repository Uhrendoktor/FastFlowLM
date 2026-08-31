#!/usr/bin/env python3
"""Authoritative geometry gate for Qwen3.8:27b.

This file intentionally contains no 8B/9B substitutions.  It is used by CI
and generator adapters before any artifact is accepted as a 27B artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GEOMETRY = {
    "model": "qwen3.8:27b",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_hidden_layers": 64,
    "full_attention": {
        "q_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
    },
    "linear_attention": {
        "q_heads": 16,
        "v_heads": 48,
        "head_dim": 128,
    },
}


def validate(obj: dict) -> None:
    if obj != GEOMETRY:
        raise SystemExit(
            "Qwen3.8:27b geometry mismatch; refusing to build or accept "
            "an artifact with substituted dimensions.\n"
            f"expected={json.dumps(GEOMETRY, sort_keys=True)}\n"
            f"actual={json.dumps(obj, sort_keys=True)}"
        )


def main() -> int:
    if len(sys.argv) == 1:
        print(json.dumps(GEOMETRY, indent=2, sort_keys=True))
        return 0
    path = Path(sys.argv[1])
    validate(json.loads(path.read_text()))
    print(f"Qwen3.8:27b geometry PASS: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
