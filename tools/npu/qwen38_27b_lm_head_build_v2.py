#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, shutil, subprocess
from pathlib import Path
K=5120; N=248320; M=128

def peano(root:Path)->Path:
    for p in root.rglob('clang++'):
        if p.is_file() and p.parent.name=='bin' and os.access(p,os.X_OK):
            try: v=subprocess.check_output([str(p),'--version'],text=True,stderr=subprocess.STDOUT,timeout=10)
            except Exception: continue
            if any(x in v for x in ('Peano','AI Engine','AIE')): return p.parent.parent
    raise SystemExit('no Peano clang++ found in pinned toolchain')

def patch(p:Path, pd:Path)->None:
    s=p.read_text().replace('use_chess?=1','use_chess?=0')
    s=s.replace('ifneq (${use_chess}, 1)\n$(error gemm_asymmetric_tile_buffering in torch2aie is Chess-only; use use_chess=1)\nendif\n','')
    s=s.replace('KERNEL_CC=xchesscc_wrapper\nKERNEL_CFLAGS=aie2p -I ${AIETOOLS_DIR}/include -I ${MLIR_AIE_DIR}/include',f'KERNEL_CC={pd}/bin/clang++\nKERNEL_CFLAGS=-O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -I ${{AIETOOLS_DIR}}/include -I ${{MLIR_AIE_DIR}}/include')
    s=s.replace('aiecc_chess_flags=--unified','aiecc_chess_flags=--no-xchesscc --no-xbridge --peano '+str(pd))
    p.write_text(s)

def main(root:Path)->int:
    root=root.resolve(); ex=root/'examples/gemm_asymmetric_tile_buffering'; patch(ex/'makefile_common',peano(root)); cfg=ex/'config1'
    subprocess.run(['make','clean'],cwd=cfg,check=False)
    subprocess.run(['make',f'M={M}',f'K={K}',f'N={N}','targetname=n1_core_bf16'],cwd=cfg,check=True,env=os.environ|{'QWEN38_LM_HEAD_K':str(K),'QWEN38_LM_HEAD_N':str(N),'QWEN38_LM_HEAD_M':str(M)})
    xs=sorted(cfg.glob('build/*.xclbin'),key=lambda p:p.stat().st_mtime,reverse=True); ms=sorted(cfg.glob('build/*.mlir'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not xs: raise SystemExit('no lm_head xclbin')
    x=xs[0]; m=ms[0] if ms else None
    if m:
        t=m.read_text();
        if str(K) not in t or str(N) not in t: raise SystemExit('MLIR does not contain K=5120 and N=248320')
        if 'MLIR_AIE' not in t: raise SystemExit('MLIR missing MLIR_AIE kernel')
    out=Path(os.environ.get('QWEN38_LM_HEAD_OUT',root/'lm_head.xclbin')); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(x,out)
    if m: shutil.copy2(m,out.with_suffix('.mlir'))
    out.with_suffix('.manifest.json').write_text('{"M":128,"K":5120,"N":248320,"kernel":"MLIR_AIE","input_dtype":"bf16","weight_dtype":"bf16","output_dtype":"bf16","abi_source":"torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_bf16.py"}\n')
    print(out); return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); a=ap.parse_args(); raise SystemExit(main(a.root))
