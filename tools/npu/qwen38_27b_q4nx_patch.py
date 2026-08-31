#!/usr/bin/env python3
"""Patch the pinned FLM_Q4NX_Converter for Qwen3.8-27B.

The Qwen35 converter mapping is architecture/layout metadata only; the actual
weights and model dimensions come from the supplied Qwen3.8-27B GGUF. No 8B/9B
weights or XCLBINs are used.
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
    reference = root / "configs" / "qwen3.5_9b.json"
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
        m += '\n\nclass Qwen35_27B(Qwen35, model_arch=ModelArch.QWEN35_27B):\n    pass\n'
    model.write_text(m)

    # This source file is only the historical Qwen35 tensor-name map.  It is
    # not a model config and contains no hidden/intermediate/vocab dimensions.
    data = json.loads(reference.read_text())
    forbidden = {"hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers"}
    assert not (forbidden & set(json.dumps(data).split()))
    config_27b.write_text(json.dumps(data, indent=2) + "\n")

    # Hard proof that this patch cannot silently be applied to another model.
    assert HIDDEN == 5120 and FFN == 17408 and VOCAB == 248320 and LAYERS == 64
    print(f"patched Qwen3.8-27B converter registry: H={HIDDEN} FFN={FFN} vocab={VOCAB} layers={LAYERS}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: qwen38_27b_q4nx_patch.py CONVERTER_ROOT")
    main(sys.argv[1])