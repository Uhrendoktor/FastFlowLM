#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.metadata, os, shutil, subprocess, sys, urllib.request
from pathlib import Path
K=5120; N=248320; M=128
PEANO_WHEEL_URL="https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-21.0.0.2026061701%2B742b6c9b-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
PEANO_WHEEL_SHA256="01da9bf7fd6d6fc86a77f85c5841fc3d74508eae118f6c0e5007ae48bacc748a"

def ensure_peano(root:Path)->Path:
    configured=os.environ.get('PEANO_INSTALL_DIR')
    if configured and (Path(configured)/'bin'/'clang++').is_file(): return Path(configured)
    cache=Path.home()/'.cache'/'qwen38-peano'; cache.mkdir(parents=True,exist_ok=True)
    wheel=cache/Path(PEANO_WHEEL_URL.split('/')[-1]).name.replace('%2B','+')
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest()!=PEANO_WHEEL_SHA256:
        urllib.request.urlretrieve(PEANO_WHEEL_URL,wheel); got=hashlib.sha256(wheel.read_bytes()).hexdigest()
        if got!=PEANO_WHEEL_SHA256: raise SystemExit(f'Peano wheel SHA256 mismatch: {got}')
    uv=shutil.which('uv') or str(Path.home()/'.local/bin/uv')
    subprocess.run([uv,'pip','install','--python',sys.executable,'--no-deps',str(wheel)],check=True)
    dist=importlib.metadata.distribution('llvm-aie'); pd=Path(dist.locate_file('llvm-aie'))
    if not (pd/'bin'/'clang++').is_file():
        candidates=list(Path(dist.locate_file('')).rglob('clang++')); pd=candidates[0].parent.parent if candidates else pd
    if not (pd/'bin'/'clang++').is_file(): raise SystemExit(f'Peano clang++ missing after install: {pd}')
    os.environ['PEANO_INSTALL_DIR']=str(pd); return pd

def patch(p:Path,pd:Path)->None:
    s=p.read_text().replace('use_chess?=1','use_chess?=0')
    s=s.replace('ifneq (${use_chess}, 1)\n$(error gemm_asymmetric_tile_buffering in torch2aie is Chess-only; use use_chess=1)\nendif\n','')
    s=s.replace('KERNEL_CC=xchesscc_wrapper\nKERNEL_CFLAGS=aie2p -I ${AIETOOLS_DIR}/include -I ${MLIR_AIE_DIR}/include',f'KERNEL_CC={pd}/bin/clang++\nKERNEL_CFLAGS=-O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG -Wno-parentheses -Wno-attributes -Wno-macro-redefined -I ${{AIETOOLS_DIR}}/include -I ${{MLIR_AIE_DIR}}/include')
    s=s.replace('aiecc_chess_flags=--unified',f'aiecc_chess_flags=--no-xchesscc --no-xbridge --peano {pd}')
    p.write_text(s)

def main(root:Path)->int:
    root=root.resolve(); pd=ensure_peano(root); ex=root/'examples/gemm_asymmetric_tile_buffering'; patch(ex/'makefile_common',pd); cfg=ex/'config1'
    subprocess.run(['make','clean'],cwd=cfg,check=False)
    env=os.environ.copy(); env.update({'QWEN38_LM_HEAD_K':str(K),'QWEN38_LM_HEAD_N':str(N),'QWEN38_LM_HEAD_M':str(M)})
    subprocess.run(['make',f'M={M}',f'K={K}',f'N={N}','targetname=n1_core_bf16'],cwd=cfg,check=True,env=env)
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
