#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path

def replace_once(path:Path,old:str,new:str)->bool:
    text=path.read_text()
    if new in text:return False
    if old not in text:return False
    path.write_text(text.replace(old,new,1)); return True

def patch_runner(runner:Path)->bool:
    s=runner.read_text()
    if 'QWEN38_27B_BUILD_ONLY' in s:return False
    marker='''def build_only(\n    current_token: int | None = None,\n    model_path: Path | None = None,\n    layer: int = DEFAULT_LAYER,\n    download_model: bool = False,\n) -> bool:\n'''
    if marker not in s: raise SystemExit('build_only signature not found')
    inject='''    if os.environ.get("QWEN38_27B_BUILD_ONLY") == "1":\n        target_schedule = _target_schedule(current_token)\n        build_schedule = _build_schedule(target_schedule)\n        xclbin_path, insts_path = build_kernel(build_schedule, _capacity_build_name(build_schedule))\n        print(f"  PASS: built {xclbin_path}")\n        print(f"  PASS: built {insts_path}")\n        return True\n\n'''
    s=s.replace(marker,marker+inject,1); s=s.replace('from pathlib import Path\n','from pathlib import Path\nimport os\n',1); runner.write_text(s); return True

def main(root:Path)->int:
    root=root.resolve(); ex=root/'examples/qwen3-decode-layer'; gen=ex/'cases/full_layer_engine_generate.py'; runner=ex/'cases/qwen3_8b_decode_layer_runner.py'; changes=[]
    specs=[('return_window_dwords != 512','return_window_dwords != 768','return window 768 dwords'),('require_source_side_packet_replay(source, 8)','require_source_side_packet_replay(source, 4)','source replay locks 4'),('require_source_side_packet_replay(src, 8)','require_source_side_packet_replay(src, 4)','source replay locks 4'),('require_attention_block_shapes(hub, WEIGHT_DWORDS * 8)','require_attention_block_shapes(hub, ATTENTION_OUTPUT_DWORDS)','attention output packet width')]
    for old,new,msg in specs:
        if replace_once(gen,old,new): changes.append(msg)
        elif old not in gen.read_text() and new not in gen.read_text(): raise SystemExit(f'27B generator contract missing both forms: {old} / {new}')
    if patch_runner(runner): changes.append('graph-only runner mode')
    # npu_build.py is intentionally left on the pinned toolchain's native
    # Chess compiler. The toolchain's clang++ is not a generic LLVM AIE2P
    # compiler and must not be substituted here.
    for c in changes: print('PATCHED:',c)
    print('PATCHED: retained pinned Chess npu_build')
    return 0
if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); raise SystemExit(main(a.root))
