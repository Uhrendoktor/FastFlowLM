#!/usr/bin/env python3
from __future__ import annotations
import json
import re
from pathlib import Path

HIDDEN=5120; FFN=17408; VOCAB=248320; LAYERS=64


def install_gguf_compat(root: Path) -> None:
    """Patch the copied llama.cpp reader; this affects only the missing optional MTP key."""
    shim = '''\n        if 'qwen35.mtp_num_hidden_layers' not in self.fields:\n            class _CompatMTPField:\n                def contents(self): return 1\n            self.fields['qwen35.mtp_num_hidden_layers'] = _CompatMTPField()\n'''
    patched=False
    for reader in Path('/tmp/q4nx/lib').glob('python*/site-packages/gguf/gguf_reader.py'):
        s=reader.read_text()
        if "qwen35.mtp_num_hidden_layers' not in self.fields" not in s:
            marker='        offs = self._build_fields(offs, kv_count)\n'
            if marker not in s: raise SystemExit('GGUFReader field-build anchor not found')
            s=s.replace(marker, marker+shim, 1); reader.write_text(s)
        patched=True
    if not patched: raise SystemExit('copied gguf_reader.py not found')
    # Keep a wrapper for the system-python metadata probe; it loads the same patched package.
    (root/'gguf.py').write_text('''import importlib, pathlib, sys\n_here=pathlib.Path(__file__).parent\nsys.path=[p for p in sys.path if p != str(_here)]\nsys.path.insert(0,next(str(p) for p in pathlib.Path("/tmp/q4nx/lib").glob("python*/site-packages")))\n_real=importlib.import_module("gguf")\nsys.modules["gguf"]=_real\nfrom gguf import *\n''')


def main(root: str) -> None:
    root=Path(root); constants=root/'q4nx/constants.py'; model=root/'q4nx/models/qwen35.py'; models_init=root/'q4nx/models/__init__.py'; reference=root/'configs/qwen3.5_9b.json'; config_27b=root/'configs/qwen3.5_27b.json'
    c=constants.read_text()
    if 'QWEN35_27B' not in c:
        c,n=re.subn(r'(    QWEN35_9B\s*=\s*auto\(\)\n)',r'\1    QWEN35_27B = auto()\n',c,count=1); assert n==1
        c,n=re.subn(r'(    ModelArch\.QWEN35_9B:\s*\["qwen35-9B","qwen3\.5-9B"\],\s*\n)',r'\1    ModelArch.QWEN35_27B: ["qwen35-27B","qwen3.5-27B","qwen3.8-27B"],\n',c,count=1); assert n==1
        c,n=re.subn(r'(    ModelArch\.QWEN35_9B:\s*"qwen3\.5_9b\.json",\s*\n)',r'\1    ModelArch.QWEN35_27B: "qwen3.5_27b.json",\n',c,count=1); assert n==1
    constants.write_text(c)
    m=model.read_text(); guard='''                if len(unpacked) == 1 and ("ssm_alpha.weight" in gguf_tensor.name or "ssm_beta.weight" in gguf_tensor.name):\n                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = unpacked[0].contiguous()\n                    continue\n'''; needle="                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])\n\n"
    if guard not in m: assert needle in m; m=m.replace(needle,needle+guard,1)
    model.write_text(m)
    mi=models_init.read_text()
    if 'Qwen35_27B' not in mi:
        mi,n=re.subn(r"from \.qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B", "from .qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B, Qwen35_27B", mi, count=1); assert n==1
        mi,n=re.subn(r"('Qwen35_9B',\s*'Qwen35_08B')",r"\1, 'Qwen35_27B'",mi,count=1); assert n==1
        models_init.write_text(mi)
    data=json.loads(reference.read_text()); assert not ({'hidden_size','intermediate_size','vocab_size','num_hidden_layers'} & set(data)); config_27b.write_text(json.dumps(data,indent=2)+'\n')
    install_gguf_compat(root); assert (HIDDEN,FFN,VOCAB,LAYERS)==(5120,17408,248320,64); print('patched Qwen3.8-27B converter registry and GGUFReader optional MTP compatibility')

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2: raise SystemExit('usage: qwen38_27b_q4nx_patch.py CONVERTER_ROOT')
    main(sys.argv[1])
