#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, shutil, subprocess
from pathlib import Path
K=5120; N=248320; M=128

def main(root:Path)->int:
    root=root.resolve(); ex=root/'examples/gemm_asymmetric_tile_buffering'; cfg=ex/'config1'
    subprocess.run(['make','clean'],cwd=cfg,check=False)
    env=os.environ.copy(); env.update({'QWEN38_LM_HEAD_K':str(K),'QWEN38_LM_HEAD_N':str(N),'QWEN38_LM_HEAD_M':str(M)})
    # The pinned torch2aie toolchain's gemm_asymmetric_tile_buffering flow is
    # Chess-based. Keep its native compiler/ABI and only override the Python
    # source name so the runtime-visible target remains n1_core.
    cmd=['make',f'M={M}',f'K={K}',f'N={N}','targetname=n1_core','aie_py_src=n1_core_bf16.py','use_chess=1']
    subprocess.run(cmd,cwd=cfg,check=True,env=env)
    xs=sorted(cfg.glob('build/*.xclbin'),key=lambda p:p.stat().st_mtime,reverse=True); ms=sorted(cfg.glob('build/*.mlir'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not xs: raise SystemExit('no lm_head xclbin')
    x=xs[0]; m=ms[0] if ms else None
    if m:
        t=m.read_text()
        if str(K) not in t or str(N) not in t: raise SystemExit('MLIR does not contain K=5120 and N=248320')
        if 'MLIR_AIE' not in t: raise SystemExit('MLIR missing MLIR_AIE kernel')
    out=Path(os.environ.get('QWEN38_LM_HEAD_OUT',root/'lm_head.xclbin')); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(x,out)
    if m: shutil.copy2(m,out.with_suffix('.mlir'))
    out.with_suffix('.manifest.json').write_text('{"M":128,"K":5120,"N":248320,"kernel":"MLIR_AIE","input_dtype":"bf16","weight_dtype":"bf16","output_dtype":"bf16","abi_source":"torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_bf16.py","targetname":"n1_core","compiler":"pinned-toolchain-chess"}\n')
    print(out); return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); a=ap.parse_args(); raise SystemExit(main(a.root))
