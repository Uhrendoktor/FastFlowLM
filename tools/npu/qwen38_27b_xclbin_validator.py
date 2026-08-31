#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path

EXPECTED = {
    'hidden_size': 5120,
    'intermediate_size': 17408,
    'vocab_size': 248320,
    'num_hidden_layers': 64,
    'num_attention_heads': 24,
    'num_key_value_heads': 4,
    'head_dim': 256,
}


def info(xclbin: Path, xclbinutil: str) -> str:
    p=subprocess.run([xclbinutil,'--info','--input',str(xclbin)],text=True,capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f'xclbinutil --info failed for {xclbin}:\n{p.stdout}\n{p.stderr}')
    return p.stdout


def validate_common(text: str, label: str) -> None:
    low=text.lower()
    for needle in ('kernel','uuid','memory','instance'):
        if needle not in low: raise SystemExit(f'{label}: xclbin info missing {needle}')
    if 'mlir_aie' not in low and 'mlir-aie' not in low:
        raise SystemExit(f'{label}: runtime MLIR_AIE kernel metadata not present')


def validate_mlir(path: Path, label: str) -> None:
    s=path.read_text()
    for needle in ('aie.device','aie.core','aie.lock','aie.dma_bd','aie.use_lock'):
        if needle not in s: raise SystemExit(f'{label}: MLIR missing {needle}')
    if s.count('aie.lock') < 1: raise SystemExit(f'{label}: no locks declared')
    if s.count('aie.dma_bd') < 1: raise SystemExit(f'{label}: no DMA BDs declared')
    if 'link_with' in s and '0.6B' in s: raise SystemExit(f'{label}: stale 0.6B link metadata')


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--xclbin',type=Path,required=True); ap.add_argument('--xclbinutil',required=True); ap.add_argument('--label',required=True); ap.add_argument('--mlir',type=Path); ap.add_argument('--manifest',type=Path); ap.add_argument('--reference-info',type=Path)
    a=ap.parse_args()
    if not a.xclbin.is_file() or a.xclbin.stat().st_size==0: raise SystemExit(f'{a.label}: missing/empty xclbin')
    t=info(a.xclbin,a.xclbinutil); validate_common(t,a.label)
    if a.mlir: validate_mlir(a.mlir,a.label)
    if a.manifest:
        m=json.loads(a.manifest.read_text())
        for k,v in EXPECTED.items():
            if m.get(k)!=v: raise SystemExit(f'{a.label}: {k}={m.get(k)!r}, expected {v!r}')
    if a.reference_info:
        r=a.reference_info.read_text()
        # Runtime ABI reference: preserve kernel symbol/interface family.  Do not
        # accept a binary that merely has the requested dimensions.
        rk=re.findall(r'(?im)^\s*(?:Kernel|Name)\s*[:=]\s*([^\s]+)',r)
        nk=re.findall(r'(?im)^\s*(?:Kernel|Name)\s*[:=]\s*([^\s]+)',t)
        if rk and nk and not any(x in nk for x in rk):
            raise SystemExit(f'{a.label}: generated kernel symbols do not match reference ABI: {nk} vs {rk}')
    print(json.dumps({'label':a.label,'bytes':a.xclbin.stat().st_size,'sha256':hashlib.sha256(a.xclbin.read_bytes()).hexdigest(),'xclbin_info_valid':True},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
