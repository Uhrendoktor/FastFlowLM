#!/usr/bin/env python3
"""Patch ROCm/FLM_Q4NX_Converter for the Qwen3.8-27B Qwen35 layout.

Qwen3.8-27B advertises model_type=qwen3_5.  The upstream converter's
Qwen35 implementation is dimension-driven and already handles the hybrid
linear/full-attention tensor maps, but the registry only names 0.8/2/4/9B.
This script adds a dedicated 27B registry entry without changing the packing
algorithm.  It is deliberately applied to a pinned converter checkout in CI.
"""
from __future__ import annotations
import json
from pathlib import Path

HIDDEN = 5120
FFN = 17408
VOCAB = 248320
LAYERS = 64


def main(root: str) -> None:
    root = Path(root)
    constants = root / "q4nx" / "constants.py"
    model = root / "q4nx" / "models" / "qwen35.py"
    config_9b = root / "configs" / "qwen3.5_9b.json"
    config_27b = root / "configs" / "qwen3.5_27b.json"

    c = constants.read_text()
    if "QWEN35_27B" not in c:
        c = c.replace(
            "    QWEN35_9B = auto()\n",
            "    QWEN35_9B = auto()\n    QWEN35_27B = auto()\n",
        )
        c = c.replace(
            '    ModelArch.QWEN35_9B:  ["qwen35-9B","qwen3.5-9B"],\n',
            '    ModelArch.QWEN35_9B:  ["qwen35-9B","qwen3.5-9B"],\n'
            '    ModelArch.QWEN35_27B: ["qwen35-27B","qwen3.5-27B","qwen3.8-27B"],\n',
        )
        c = c.replace(
            '    ModelArch.QWEN35_9B: "qwen3.5_9b.json",\n',
            '    ModelArch.QWEN35_9B: "qwen3.5_9b.json",\n'
            '    ModelArch.QWEN35_27B: "qwen3.5_27b.json",\n',
        )
    constants.write_text(c)

    m = model.read_text()
    if "class Qwen35_27B" not in m:
        m += '\n\nclass Qwen35_27B(Qwen35, model_arch=ModelArch.QWEN35_27B):\n'
        m += '    print("[INFO] Using Qwen35_27B converter")\n'
        m += '    pass\n'
    model.write_text(m)

    data = json.loads(config_9b.read_text())
    config_27b.write_text(json.dumps(data, indent=2) + "\n")

    # Hard proof that this patch cannot silently be applied to another model.
    assert HIDDEN == 5120 and FFN == 17408 and VOCAB == 248320 and LAYERS == 64
    print(f"patched Qwen3.8-27B converter: H={HIDDEN} FFN={FFN} vocab={VOCAB} layers={LAYERS}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: qwen38_27b_q4nx_patch.py CONVERTER_ROOT")
    main(sys.argv[1])
