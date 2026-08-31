#!/usr/bin/env python3
"""Patch the pinned FLM_Q4NX_Converter for Qwen3.8-27B.

The Qwen35 converter mapping is architecture/layout metadata only; the actual
weights and model dimensions come from the supplied Qwen3.8-27B GGUF. No 8B/9B
weights or XCLBINs are used.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

HIDDEN = 5120
FFN = 17408
VOCAB = 248320
LAYERS = 64


def main(root: str) -> None:
    root = Path(root)
    constants = root / "q4nx" / "constants.py"
    model = root / "q4nx" / "models" / "qwen35.py"
    models_init = root / "q4nx" / "models" / "__init__.py"
    reference = root / "configs" / "qwen3.5_9b.json"
    config_27b = root / "configs" / "qwen3.5_27b.json"

    c = constants.read_text()
    if "QWEN35_27B" not in c:
        c, n = re.subn(r"(    QWEN35_9B\s*=\s*auto\(\)\n)", r"\1    QWEN35_27B = auto()\n", c, count=1)
        assert n == 1, "could not add QWEN35_27B enum"
        c, n = re.subn(r'(    ModelArch\.QWEN35_9B:\s*\["qwen35-9B","qwen3\.5-9B"\],\s*\n)', r'\1    ModelArch.QWEN35_27B: ["qwen35-27B","qwen3.5-27B","qwen3.8-27B"],\n', c, count=1)
        assert n == 1, "could not add QWEN35_27B name mapping"
        c, n = re.subn(r'(    ModelArch\.QWEN35_9B:\s*"qwen3\.5_9b\.json",\s*\n)', r'\1    ModelArch.QWEN35_27B: "qwen3.5_27b.json",\n', c, count=1)
        assert n == 1, "could not add QWEN35_27B config mapping"
    constants.write_text(c)

    m = model.read_text()
    # The pinned Qwen35 converter assumes every ssm_alpha/beta tensor unpacks
    # to (d, m, q). Qwen3.8-27B GGUF carries these as scalar F32 tensors, so
    # preserve them directly before that legacy reorder path is reached.
    guard = '''                if len(unpacked) == 1 and ("ssm_alpha.weight" in gguf_tensor.name or "ssm_beta.weight" in gguf_tensor.name):
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = unpacked[0].contiguous()
                    continue
\n'''
    needle = "                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])\n\n"
    if guard not in m:
        assert needle in m, "converter unpack site not found"
        m = m.replace(needle, needle + guard, 1)
    if "len(unpacked) == 1 and (\"ssm_alpha.weight\"" not in m:
        raise AssertionError("Qwen3.8 F32 ssm_alpha/beta guard was not installed")
    if "class Qwen35_27B" not in m:
        m += '\n\nclass Qwen35_27B(Qwen35, model_arch=ModelArch.QWEN35_27B):\n    print("[INFO] Using Qwen35_27B converter")\n    pass\n'
    model.write_text(m)

    mi = models_init.read_text()
    if "Qwen35_27B" not in mi:
        mi, n = re.subn(
            r"from \.qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B",
            "from .qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B, Qwen35_27B",
            mi,
            count=1,
        )
        assert n == 1, "could not export Qwen35_27B from models package"
        mi, n = re.subn(r"('Qwen35_9B',\s*'Qwen35_08B')", r"\1, 'Qwen35_27B'", mi, count=1)
        assert n == 1, "could not add Qwen35_27B to models __all__"
        models_init.write_text(mi)

    data = json.loads(reference.read_text())
    forbidden = {"hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers"}
    assert not (forbidden & set(data))
    config_27b.write_text(json.dumps(data, indent=2) + "\n")

    assert HIDDEN == 5120 and FFN == 17408 and VOCAB == 248320 and LAYERS == 64
    print(f"patched Qwen3.8-27B converter registry: H={HIDDEN} FFN={FFN} vocab={VOCAB} layers={LAYERS}; F32 ssm alpha/beta guard installed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: qwen38_27b_q4nx_patch.py CONVERTER_ROOT")
    main(sys.argv[1])
