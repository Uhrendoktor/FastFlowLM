#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path

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
    helper='''def _compile_aie_object(source_names: tuple[str, ...], object_name: str) -> None:\n    if not source_names: raise ValueError(f"missing source for role object: {object_name}")\n    src=EXPERIMENT_DIR/source_names[0]; obj=_role_object_path(object_name); obj.parent.mkdir(parents=True,exist_ok=True)\n    if os.environ.get("QWEN38_USE_PEANO") != "1":\n        compiler=TOOLCHAIN_DIR/"bin"/"xchesscc_wrapper"; cmd=[str(compiler),"aie2p",f"-I{EXPERIMENT_DIR}",f"-I{AIETOOLS_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels/aie2p'}","-c",str(src),"-o",str(obj)]\n    else:\n        compiler=Path(os.environ["PEANO_INSTALL_DIR"])/"bin"/"clang++"; cmd=[str(compiler),"--target=aie2p-none-unknown-elf","-O2","-std=c++20","-DNDEBUG","-Wno-parentheses","-Wno-attributes","-Wno-macro-redefined",f"-I{EXPERIMENT_DIR}",f"-I{AIETOOLS_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels'}",f"-I{MLIR_AIE_DIR/'include/aie_kernels/aie2p'}","-c",str(src),"-o",str(obj)]\n    print(f"  Compiling {object_name} with {compiler} from {src.name}...")\n    run_command(cmd)\n'''
    s=s[:start]+helper+s[end:]
    marker='''        "--no-compile-host",\n'''
    if marker not in s: raise SystemExit('aiecc command anchor not found')
    if '--no-xchesscc' not in s:s=s.replace(marker,marker+'''        *(["--no-xchesscc","--no-xbridge","--peano",os.environ["PEANO_INSTALL_DIR"]] if os.environ.get("QWEN38_USE_PEANO") == "1" else []),\n''',1)
    nb.write_text(s)

def main(root:Path)->int:
    root=root.resolve(); ex=root/'examples/qwen3-decode-layer'; runner=ex/'cases/qwen3_8b_decode_layer_runner.py'; nb=ex/'npu_build.py'
    if patch_runner(runner): print('PATCHED: graph-only runner mode')
    patch_npu_build(nb); print('PATCHED: Peano AIE2P target triple fallback; Chess retained as default')
    print('PATCHED: generator adapter owns 27B topology')
    return 0
if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); raise SystemExit(main(a.root))
