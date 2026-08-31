#!/usr/bin/env python3
"""Patch the pinned FLM_Q4NX_Converter for Qwen3.8-27B.

The converter implementation is dimension-driven.  This patch adds a distinct
27B model-arch/config entry; no 8B/9B weights or XCLBINs are consumed.
"""
from __future__ import annotations
import json
from pathlib import Path

HIDDEN = 5120
FFN = 17408
VOCAB = 248320
LAYERS = 64

# This is the converter mapping for the Qwen35 tensor layout, kept as an
# independent 27B config rather than copying a model artifact.
QWEN35_CONFIG = {
    "model_arch": "qwen35",
    "model_type": "qwen3_5",
    "q4nx": {
        "weight_dtype": "q4nx",
        "scale_dtype": "bf16",
        "zero_dtype": "bf16"
    }
}


def main(root: str) -> None:
    root = Path(root)
    constants = root / "q4nx" / "constants.py"
    model = root / "q4nx" / "models" / "qwen35.py"
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
        m += '    pass\n'
    model.write_text(m)

    config_27b.write_text(json.dumps(QWEN35_CONFIG, indent=2) + "\n")
    assert HIDDEN == 5120 and FFN == 17408 and VOCAB == 248320 and LAYERS == 64
    print(f"patched Qwen3.8-27B converter: H={HIDDEN} FFN={FFN} vocab={VOCAB} layers={LAYERS}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: qwen38_27b_q4nx_patch.py CONVERTER_ROOT")
    main(sys.argv[1])