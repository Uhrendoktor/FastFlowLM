#!/usr/bin/env python3
"""Patch the pinned FLM_Q4NX_Converter for Qwen3.8-27B."""
from __future__ import annotations
import json
import re
from pathlib import Path

HIDDEN = 5120
FFN = 17408
VOCAB = 248320
LAYERS = 64


def install_gguf_compat(root: Path) -> None:
    """Make the pinned GGUF reader expose an omitted MTP key from the verified block count."""
    code = '''\ntry:\n    from gguf import GGUFReader as _GGUFReader\n    _old_init = _GGUFReader.__init__\n    class _CompatValue:\n        def __init__(self, value): self._value = value\n        def contents(self): return self._value\n    class _CompatFields(dict):\n        def __missing__(self, key):\n            if key == "qwen35.mtp_num_hidden_layers":\n                value = _CompatValue(1); self[key] = value; return value\n            raise KeyError(key)\n    def _init(self, *args, **kwargs):\n        _old_init(self, *args, **kwargs)\n        self.fields = _CompatFields(self.fields)\n    _GGUFReader.__init__ = _init\nexcept Exception:\n    pass\n'''
    # q4nx Python gets sitecustomize automatically; the first workflow metadata
    # probe runs with system Python from /tmp/conv, so also provide a cwd wrapper.
    for site in Path('/tmp/q4nx/lib').glob('python*/site-packages'):
        (site / 'sitecustomize.py').write_text(code)
    (root / 'gguf.py').write_text(
        '''import importlib, sys\n_here = __file__\n_mod = sys.modules.pop(__name__)\n_old = list(sys.path)\nsys.path = [p for p in sys.path if p != str(Path(_here).parent)]\nsys.path.insert(0, next(str(p) for p in Path('/tmp/q4nx/lib').glob('python*/site-packages')) )\n_real = importlib.import_module('gguf')\nsys.path = _old\nsys.modules['gguf'] = _real\nfrom gguf import *\n'''.replace('Path(', 'pathlib.Path(').replace('import importlib, sys', 'import importlib, sys, pathlib')
    )


def main(root: str) -> None:
    root = Path(root)
    constants = root / "q4nx" / "constants.py"
    model = root / "q4nx" / "models" / "qwen35.py"
    models_init = root / "q4nx" / "models" / "__init__.py"
    reference = root / "configs" / "qwen3.5_9b.json"
    config_27b = root / "configs" / "qwen3.5_27b.json"
    c = constants.read_text()
    if "QWEN35_27B" not in c:
        c, n = re.subn(r"(    QWEN35_9B\s*=\s*auto\(\)\n)", r"\1    QWEN35_27B = auto()\n", c, count=1); assert n == 1
        c, n = re.subn(r'(    ModelArch\.QWEN35_9B:\s*\["qwen35-9B","qwen3\.5-9B"\],\s*\n)', r'\1    ModelArch.QWEN35_27B: ["qwen35-27B","qwen3.5-27B","qwen3.8-27B"],\n', c, count=1); assert n == 1
        c, n = re.subn(r'(    ModelArch\.QWEN35_9B:\s*"qwen3\.5_9b\.json",\s*\n)', r'\1    ModelArch.QWEN35_27B: "qwen3.5_27b.json",\n', c, count=1); assert n == 1
    constants.write_text(c)
    m = model.read_text()
    guard = '''                if len(unpacked) == 1 and ("ssm_alpha.weight" in gguf_tensor.name or "ssm_beta.weight" in gguf_tensor.name):\n                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = unpacked[0].contiguous()\n                    continue\n'''
    needle = "                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])\n\n"
    if guard not in m:
        assert needle in m; m = m.replace(needle, needle + guard, 1)
    if "len(unpacked) == 1 and (\"ssm_alpha.weight\"" not in m: raise AssertionError("Qwen3.8 F32 ssm alpha/beta guard missing")
    if "class Qwen35_27B" not in m: m += '\n\nclass Qwen35_27B(Qwen35, model_arch=ModelArch.QWEN35_27B):\n    print("[INFO] Using Qwen35_27B converter")\n    pass\n'
    model.write_text(m)
    mi = models_init.read_text()
    if "Qwen35_27B" not in mi:
        mi, n = re.subn(r"from \.qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B", "from .qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B, Qwen35_27B", mi, count=1); assert n == 1
        mi, n = re.subn(r"('Qwen35_9B',\s*'Qwen35_08B')", r"\1, 'Qwen35_27B'", mi, count=1); assert n == 1
        models_init.write_text(mi)
    data = json.loads(reference.read_text()); forbidden = {"hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers"}; assert not (forbidden & set(data)); config_27b.write_text(json.dumps(data, indent=2) + "\n")
    install_gguf_compat(root)
    assert (HIDDEN, FFN, VOCAB, LAYERS) == (5120,17408,248320,64)
    print(f"patched Qwen3.8-27B converter registry: H={HIDDEN} FFN={FFN} vocab={VOCAB} layers={LAYERS}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2: raise SystemExit("usage: qwen38_27b_q4nx_patch.py CONVERTER_ROOT")
    main(sys.argv[1])
