#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, shutil, subprocess
from pathlib import Path
K=5120; N=248320; M=128

def configure_peano_makefile(cfg:Path,peano:Path)->None:
    p=cfg.parent/'makefile_common'; s=p.read_text(); s=s.replace('use_chess?=1','use_chess?=0'); s=s.replace('ifneq (${use_chess}, 1)\n$(error gemm_asymmetric_tile_buffering in torch2aie is Chess-only; use use_chess=1)\nendif\n','')
    start='KERNEL_CC=xchesscc_wrapper\nKERNEL_CFLAGS=aie2p -I ${AIETOOLS_DIR}/include -I ${MLIR_AIE_DIR}/include'; repl=f'KERNEL_CC={peano}/bin/clang++\nKERNEL_CFLAGS=-triple aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -Wno-parentheses -Wno-attributes -Wno-macro-redefined -I ${{AIETOOLS_DIR}}/include -I ${{MLIR_AIE_DIR}}/include'
    if start not in s: raise SystemExit('gemm makefile kernel compiler anchor missing')
    s=s.replace(start,repl,1).replace('aiecc_chess_flags=--unified',f'aiecc_chess_flags=--no-xchesscc --no-xbridge --peano {peano}',1); p.write_text(s)
    m=cfg/'Makefile'; m.write_text(m.read_text().replace('kernelsrc := mm_bfp_mixed.cc','kernelsrc := mm_bf16.cc'))

def main(root:Path)->int:
    root=root.resolve(); cfg=root/'examples/gemm_asymmetric_tile_buffering/config1'; peano_env=os.environ.get('PEANO_INSTALL_DIR');
    if not peano_env: raise SystemExit('PEANO_INSTALL_DIR must point at pinned toolchain Peano')
    peano=Path(peano_env); assert (peano/'bin/clang++').is_file(); configure_peano_makefile(cfg,peano); subprocess.run(['make','clean'],cwd=cfg,check=False)
    env=os.environ.copy(); env.update({'QWEN38_LM_HEAD_K':str(K),'QWEN38_LM_HEAD_N':str(N),'QWEN38_LM_HEAD_M':str(M)}); subprocess.run(['make',f'M={M}',f'K={K}',f'N={N}','targetname=n1_core','aie_py_src=n1_core_bf16.py','use_chess=0'],cwd=cfg,check=True,env=env)
    xs=sorted(cfg.glob('build/*.xclbin'),key=lambda p:p.stat().st_mtime,reverse=True); ms=sorted(cfg.glob('build/*.mlir'),key=lambda p:p.stat().st_mtime,reverse=True); 
    if not xs: raise SystemExit('no lm_head xclbin')
    x=xs[0]; m=ms[0] if ms else None
    if m:
        t=m.read_text(); assert str(K) in t and str(N) in t and 'MLIR_AIE' in t
    out=Path(os.environ.get('QWEN38_LM_HEAD_OUT',root/'lm_head.xclbin')); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(x,out)
    if m: shutil.copy2(m,out.with_suffix('.mlir'))
    out.with_suffix('.manifest.json').write_text('{"M":128,"K":5120,"N":248320,"kernel":"MLIR_AIE","input_dtype":"bf16","weight_dtype":"bf16","output_dtype":"bf16","abi_source":"torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_bf16.py + mm_bf16.cc","targetname":"n1_core","compiler":"pinned-toolchain-peano"}\n'); print(out); return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); a=ap.parse_args(); raise SystemExit(main(a.root))
