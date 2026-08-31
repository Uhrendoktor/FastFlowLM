#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, shutil, subprocess
from pathlib import Path

K=5120; N=248320; M=128

def find_peano(root: Path) -> Path:
    for p in root.rglob('clang++'):
        if p.is_file() and p.parent.name=='bin' and os.access(p,os.X_OK):
            try:
                v=subprocess.check_output([str(p),'--version'],text=True,stderr=subprocess.STDOUT,timeout=10)
            except Exception: continue
            if 'AI Engine' in v or 'AIE' in v or 'Peano' in v:
                return p.parent.parent
    raise SystemExit('pinned toolchain does not contain an identifiable Peano clang++')

def patch_makefile_common(p: Path, peano: Path) -> None:
    s=p.read_text()
    s=s.replace('use_chess?=1','use_chess?=0')
    s=s.replace('ifneq (${use_chess}, 1)\n$(error gemm_asymmetric_tile_buffering in torch2aie is Chess-only; use use_chess=1)\nendif\n','')
    s=s.replace('KERNEL_CC=xchesscc_wrapper\nKERNEL_CFLAGS=aie2p -I ${AIETOOLS_DIR}/include -I ${MLIR_AIE_DIR}/include',f'KERNEL_CC={peano}/bin/clang++\nKERNEL_CFLAGS=-O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -I ${{AIETOOLS_DIR}}/include -I ${{MLIR_AIE_DIR}}/include')
    s=s.replace('aiecc_chess_flags=--unified','aiecc_chess_flags=--no-xchesscc --no-xbridge --peano '+str(peano))
    p.write_text(s)

def main(root: Path) -> int:
    root=root.resolve(); ex=root/'examples/gemm_asymmetric_tile_buffering'; common=ex/'makefile_common'; peano=find_peano(root); patch_makefile_common(common,peano)
    cfg=ex/'config1'; subprocess.run(['make','clean'],cwd=cfg,check=False)
    env=os.environ.copy(); env['QWEN38_LM_HEAD_K']=str(K); env['QWEN38_LM_HEAD_N']=str(N); env['QWEN38_LM_HEAD_M']=str(M)
    subprocess.run(['make',f'M={M}',f'K={K}',f'N={N}','targetname=n1_core_bf16'],cwd=cfg,env=env,check=True)
    xs=sorted(cfg.glob('build/*.xclbin'),key=lambda p:p.stat().st_mtime,reverse=True); mlirs=sorted(cfg.glob('build/*.mlir'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not xs: raise SystemExit('lm_head build produced no xclbin')
    x=xs[0]; mlir=mlirs[0] if mlirs else None
    # Generated MLIR is the authoritative tensor/ABI contract for the generic
    # MLIR_AIE GEMM kernel used by FastFlowLM's lm_head path.
    if mlir:
        t=mlir.read_text();
        for q in (f'M={M}', f'K={K}', f'N={N}'):
            if q not in t: raise SystemExit(f'lm_head MLIR missing {q}')
    out=Path(os.environ.get('QWEN38_LM_HEAD_OUT',root/'lm_head.xclbin')); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(x,out)
    manifest=out.with_suffix('.manifest.json'); manifest.write_text(f'{{"M":{M},"K":{K},"N":{N},"kernel":"MLIR_AIE","input_dtype":"bf16","weight_dtype":"bf16","output_dtype":"bf16","abi_source":"examples/gemm_asymmetric_tile_buffering/config1/n1_core_bf16.py"}}\n')
    if mlir: shutil.copy2(mlir,out.with_suffix('.mlir'))
    print(f'LM_HEAD_XCLBIN={out}')
    print(f'LM_HEAD_MANIFEST={manifest}')
    return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); a=ap.parse_args(); raise SystemExit(main(a.root))
