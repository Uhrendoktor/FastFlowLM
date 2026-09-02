#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, tempfile
from pathlib import Path

EXPECTED={'hidden_size':5120,'intermediate_size':17408,'vocab_size':248320,'num_hidden_layers':64,'num_attention_heads':24,'num_key_value_heads':4,'head_dim':256}
FORBIDDEN=('qwen3-0.6b','qwen3-8b','qwen3-9b','qwen3-30b')
REQUIRED_SECTIONS=('IP_LAYOUT','MEM_TOPOLOGY','CONNECTIVITY','EMBEDDED_METADATA','AIE_METADATA')

def run_dump(xclbin:Path,xclbinutil:str,section:str,fmt:str,tmp:Path)->bytes:
    out=tmp/f'{section.lower()}.{fmt.lower()}'
    p=subprocess.run([xclbinutil,'--dump-section',f'{section}:{fmt}:{out}','--input',str(xclbin)],text=True,capture_output=True)
    if p.returncode!=0 or not out.is_file() or out.stat().st_size==0:
        raise SystemExit(f'{xclbin.name}: cannot dump {section}:{fmt}:\n{p.stdout}\n{p.stderr}')
    return out.read_bytes()

def info(xclbin:Path,xclbinutil:str)->str:
    p=subprocess.run([xclbinutil,'--info','--input',str(xclbin)],text=True,capture_output=True)
    if p.returncode!=0: raise SystemExit(f'xclbinutil --info failed for {xclbin}:\n{p.stdout}\n{p.stderr}')
    return p.stdout

def validate_mlir(path:Path,label:str)->None:
    s=path.read_text().lower()
    for needle in ('aie.device','aie.core','aie.lock','aie.dma_bd','aie.use_lock'):
        if needle not in s: raise SystemExit(f'{label}: MLIR missing {needle}')
    if s.count('aie.lock')<1 or s.count('aie.dma_bd')<1: raise SystemExit(f'{label}: missing lock/BD declarations')
    if any(x in s for x in FORBIDDEN): raise SystemExit(f'{label}: forbidden smaller-model identity in MLIR')

def validate_manifest(path:Path,label:str)->dict:
    m=json.loads(path.read_text())
    for k,v in EXPECTED.items():
        if k in m and m[k]!=v: raise SystemExit(f'{label}: {k}={m[k]!r}, expected {v!r}')
    return m

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--xclbin',type=Path,required=True); ap.add_argument('--xclbinutil',required=True); ap.add_argument('--label',required=True); ap.add_argument('--mlir',type=Path); ap.add_argument('--manifest',type=Path); ap.add_argument('--expected-k',type=int); ap.add_argument('--expected-n',type=int)
    a=ap.parse_args()
    if not a.xclbin.is_file() or a.xclbin.stat().st_size==0: raise SystemExit(f'{a.label}: missing/empty xclbin')
    text=info(a.xclbin,a.xclbinutil); low=text.lower()
    if 'mlir_aie' not in low and 'mlir-aie' not in low: raise SystemExit(f'{a.label}: MLIR_AIE kernel metadata not present')
    if any(x in low for x in FORBIDDEN): raise SystemExit(f'{a.label}: forbidden smaller-model identity in xclbin info')
    with tempfile.TemporaryDirectory(prefix='qwen38-xclbin-') as td:
        tmp=Path(td)
        for sec in REQUIRED_SECTIONS:
            data=run_dump(a.xclbin,a.xclbinutil,sec,'JSON' if sec!='EMBEDDED_METADATA' and sec!='AIE_METADATA' else 'RAW',tmp)
            blob=data.decode('utf-8','ignore').lower()
            if any(x in blob for x in FORBIDDEN): raise SystemExit(f'{a.label}: forbidden smaller-model identity in {sec}')
            if sec=='IP_LAYOUT':
                if 'mlir_aie' not in blob and 'mlir-aie' not in blob: raise SystemExit(f'{a.label}: IP_LAYOUT lacks MLIR_AIE')
            if sec=='MEM_TOPOLOGY' and 'm_mem_data' not in blob: raise SystemExit(f'{a.label}: malformed MEM_TOPOLOGY')
            if sec=='CONNECTIVITY' and 'm_connection' not in blob: raise SystemExit(f'{a.label}: malformed CONNECTIVITY')
            if sec=='AIE_METADATA':
                for marker in ('tile','buffer','lock','bd'):
                    if marker not in blob: raise SystemExit(f'{a.label}: AIE_METADATA lacks {marker} topology marker')
    if a.mlir: validate_mlir(a.mlir,a.label)
    manifest=None
    if a.manifest: manifest=validate_manifest(a.manifest,a.label)
    if a.expected_k is not None:
        k=manifest.get('K',manifest.get('k')) if manifest else None
        if k!=a.expected_k: raise SystemExit(f'{a.label}: K={k!r}, expected {a.expected_k}')
    if a.expected_n is not None:
        n=manifest.get('N',manifest.get('n')) if manifest else None
        if n!=a.expected_n: raise SystemExit(f'{a.label}: N={n!r}, expected {a.expected_n}')
    digest=hashlib.sha256(a.xclbin.read_bytes()).hexdigest()
    print(json.dumps({'label':a.label,'bytes':a.xclbin.stat().st_size,'sha256':digest,'xclbin_info_valid':True,'required_sections_validated':list(REQUIRED_SECTIONS),'runtime_kernel':'MLIR_AIE','hardware_executed':False},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
