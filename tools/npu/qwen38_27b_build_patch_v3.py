#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.metadata, os, shutil, subprocess, sys, urllib.request
from pathlib import Path
PEANO_WHEEL_URL="https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-21.0.0.2026061701%2B742b6c9b-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
PEANO_WHEEL_SHA256="01da9bf7fd6d6fc86a77f85c5841fc3d74508eae118f6c0e5007ae48bacc748a"

def replace_once(path:Path,old:str,new:str)->bool:
    text=path.read_text()
    if new in text:return False
    if old not in text: raise SystemExit(f'anchor missing in {path}: {old[:120]!r}')
    path.write_text(text.replace(old,new,1)); return True

def ensure_peano()->Path:
    configured=os.environ.get('PEANO_INSTALL_DIR')
    if configured and (Path(configured)/'bin'/'clang++').is_file(): return Path(configured)
    cache=Path.home()/'.cache'/'qwen38-peano'; cache.mkdir(parents=True,exist_ok=True)
    wheel=cache/Path(PEANO_WHEEL_URL.split('/')[-1]).name.replace('%2B','+')
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest()!=PEANO_WHEEL_SHA256:
        urllib.request.urlretrieve(PEANO_WHEEL_URL,wheel); got=hashlib.sha256(wheel.read_bytes()).hexdigest()
        if got!=PEANO_WHEEL_SHA256: raise SystemExit(f'Peano wheel SHA256 mismatch: {got} != {PEANO_WHEEL_SHA256}')
    uv=shutil.which('uv') or str(Path.home()/'.local/bin/uv')
    subprocess.run([uv,'pip','install','--python',sys.executable,'--no-deps',str(wheel)],check=True)
    dist=importlib.metadata.distribution('llvm-aie'); base=Path(dist.locate_file(''))
    candidates=list(base.rglob('clang++')); pd=candidates[0].parent.parent if candidates else Path(dist.locate_file('llvm-aie'))
    if not (pd/'bin'/'clang++').is_file(): raise SystemExit(f'installed llvm-aie has no Peano clang++ under {pd}')
    os.environ['PEANO_INSTALL_DIR']=str(pd); return pd

def patch_runner(runner:Path)->bool:
    s=runner.read_text()
    if 'QWEN38_27B_BUILD_ONLY' in s:return False
    marker='''def build_only(\n    current_token: int | None = None,\n    model_path: Path | None = None,\n    layer: int = DEFAULT_LAYER,\n    download_model: bool = False,\n) -> bool:\n'''
    if marker not in s: raise SystemExit('build_only signature not found')
    inject='''    if os.environ.get("QWEN38_27B_BUILD_ONLY") == "1":\n        target_schedule = _target_schedule(current_token)\n        build_schedule = _build_schedule(target_schedule)\n        xclbin_path, insts_path = build_kernel(build_schedule, _capacity_build_name(build_schedule))\n        print(f"  PASS: built {xclbin_path}")\n        print(f"  PASS: built {insts_path}")\n        return True\n\n'''
    s=s.replace(marker,marker+inject,1); s=s.replace('from pathlib import Path\n','from pathlib import Path\nimport os\n',1); runner.write_text(s); return True

def patch_npu_build(nb:Path)->None:
    s=nb.read_text(); start=s.find('def _compile_aie_object('); end=s.find('\ndef _linked_role_objects',start)
    if start<0 or end<0: raise SystemExit('cannot locate _compile_aie_object')
    helper='''def _compile_aie_object(source_names: tuple[str, ...], object_name: str) -> None:\n    if not source_names: raise ValueError(f"missing source for role object: {object_name}")\n    src=EXPERIMENT_DIR/source_names[0]; obj=_role_object_path(object_name); obj.parent.mkdir(parents=True,exist_ok=True)\n    if os.environ.get("QWEN38_USE_PEANO") == "1":\n        peano=Path(os.environ["PEANO_INSTALL_DIR"]); compiler=peano/"bin"/"clang++"\n        cmd=[str(compiler),"-O2","-std=c++20","--target=aie2p-none-unknown-elf","-Wno-parentheses","-Wno-attributes","-Wno-macro-redefined","-DNDEBUG",f"-I{EXPERIMENT_DIR}",f"-I{AIETOOLS_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels/aie2p'}","-c",str(src),"-o",str(obj)]\n        print(f"  Compiling {object_name} with Peano from {src.name}...")\n    else:\n        compiler=TOOLCHAIN_DIR/"bin"/"xchesscc_wrapper"; cmd=[str(compiler),"aie2p",f"-I{EXPERIMENT_DIR}",f"-I{AIETOOLS_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels/aie2p'}","-c",str(src),"-o",str(obj)]\n        print(f"  Compiling {object_name} with Chess from {src.name}...")\n    run_command(cmd)\n'''
    s=s[:start]+helper+s[end:]
    anchor='''    cmd = [\n        str(aiecc),\n        "-v",\n        f"-j{AIECC_JOBS}",\n        f"--aietools={AIETOOLS_DIR}",\n        "--no-compile-host",\n'''
    if anchor not in s: raise SystemExit('aiecc command anchor not found')
    s=s.replace(anchor,anchor+'''        *(["--no-xchesscc","--no-xbridge","--peano",os.environ["PEANO_INSTALL_DIR"]] if os.environ.get("QWEN38_USE_PEANO") == "1" else []),\n''',1); nb.write_text(s)

def main(root:Path)->int:
    root=root.resolve(); pd=ensure_peano(); os.environ['PEANO_INSTALL_DIR']=str(pd); ex=root/'examples/qwen3-decode-layer'; gen=ex/'cases/full_layer_engine_generate.py'; runner=ex/'cases/qwen3_8b_decode_layer_runner.py'; nb=ex/'npu_build.py'; changes=[]
    for old,new,msg in [('return_window_dwords != 512','return_window_dwords != 768','return window 768 dwords'),('require_source_side_packet_replay(source, 8)','require_source_side_packet_replay(source, 4)','source replay locks 4'),('require_source_side_packet_replay(src, 8)','require_source_side_packet_replay(src, 4)','source replay locks 4'),('require_attention_block_shapes(hub, WEIGHT_DWORDS * 8)','require_attention_block_shapes(hub, ATTENTION_OUTPUT_DWORDS)','attention output packet width')]:
        if replace_once(gen,old,new): changes.append(msg)
    if patch_runner(runner): changes.append('graph-only runner mode')
    patch_npu_build(nb); changes.append('Peano-capable AIE object/MLIR compilation')
    for c in changes: print('PATCHED:',c)
    return 0
if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); raise SystemExit(main(a.root))
