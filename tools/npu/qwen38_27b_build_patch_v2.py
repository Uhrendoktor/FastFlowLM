#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path

def once(p: Path, old: str, new: str) -> bool:
    s=p.read_text()
    if new in s: return False
    if old not in s: raise SystemExit(f'anchor missing in {p}: {old!r}')
    p.write_text(s.replace(old,new,1)); return True

def main(root: Path) -> None:
    root=root.resolve(); ex=root/'examples/qwen3-decode-layer'
    gen=ex/'cases/full_layer_engine_generate.py'
    runner=ex/'cases/qwen3_8b_decode_layer_runner.py'
    nb=ex/'npu_build.py'
    changes=[]
    if once(gen,'return_window_dwords != 512','return_window_dwords != 768'): changes.append('return-window=768 dwords')
    if once(gen,'require_source_side_packet_replay(source, 8)','require_source_side_packet_replay(source, 4)'): changes.append('source replay locks=4')
    if once(gen,'require_source_side_packet_replay(src, 8)','require_source_side_packet_replay(src, 4)'): changes.append('source replay locks=4')
    if once(gen,'require_attention_block_shapes(hub, WEIGHT_DWORDS * 8)','require_attention_block_shapes(hub, ATTENTION_OUTPUT_DWORDS)'): changes.append('attention output=ATTENTION_OUTPUT_DWORDS')
    if once(nb,'compiler = TOOLCHAIN_DIR / "bin" / "xchesscc_wrapper"','compiler = Path(os.environ.get("PEANO_INSTALL_DIR", "")) / "bin" / "clang++" if os.environ.get("QWEN38_USE_PEANO") == "1" else TOOLCHAIN_DIR / "bin" / "xchesscc_wrapper"'): changes.append('role objects use Peano when requested')
    # Insert Peano flags into the role-object command only for the Peano path.
    anchor='    cmd = [\n        str(compiler),\n        "aie2p",'
    repl='    if os.environ.get("QWEN38_USE_PEANO") == "1":\n        cmd = [str(compiler), "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-DNDEBUG"]\n    else:\n        cmd = [str(compiler), "aie2p",'
    # Rebuild the closing portion of the command only once.
    s=nb.read_text()
    if 'QWEN38_USE_PEANO' not in s:
        if anchor not in s: raise SystemExit('npu_build compile command anchor missing')
        s=s.replace(anchor,repl,1)
        s=s.replace('        f"-I{MLIR_AIE_DIR / \'include/aie_kernels/aie2p\'}",\n        "-c",', '        f"-I{MLIR_AIE_DIR / \'include/aie_kernels/aie2p\'}",\n        "-c",',1)
        # close the conditional command before the print; the common include/source/output args are
        # valid for both compilers, so keep them after the compiler prefix.
        s=s.replace('        str(obj),\n    ]\n    print(f"  Compiling {object_name}', '        str(obj),\n    ]\n    if os.environ.get("QWEN38_USE_PEANO") == "1":\n        cmd += ["-I" + str(EXPERIMENT_DIR), "-I" + str(AIETOOLS_DIR / "include"), "-I" + str(MLIR_AIE_DIR / "include"), "-I" + str(MLIR_AIE_DIR / "include/aie_kernels"), "-I" + str(MLIR_AIE_DIR / "include/aie_kernels/aie2p"), "-c", str(src), "-o", str(obj)]\n    print(f"  Compiling {object_name}',1)
        nb.write_text(s); changes.append('Peano role-object command')
    # The previous transformation is intentionally conservative; if the source still contains the
    # original Chess command, replace the entire helper with a clean implementation.
    s=nb.read_text()
    start=s.find('def _compile_aie_object('); end=s.find('\ndef _linked_role_objects',start)
    if start<0 or end<0: raise SystemExit('cannot locate _compile_aie_object')
    helper='''def _compile_aie_object(source_names: tuple[str, ...], object_name: str) -> None:\n    if not source_names:\n        raise ValueError(f"missing source for role object: {object_name}")\n    src = EXPERIMENT_DIR / source_names[0]\n    obj = _role_object_path(object_name)\n    obj.parent.mkdir(parents=True, exist_ok=True)\n    if os.environ.get("QWEN38_USE_PEANO") == "1":\n        peano = Path(os.environ["PEANO_INSTALL_DIR"])\n        compiler = peano / "bin" / "clang++"\n        cmd = [str(compiler), "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf",\n               "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-DNDEBUG",\n               f"-I{EXPERIMENT_DIR}", f"-I{AIETOOLS_DIR / 'include'}",\n               f"-I{MLIR_AIE_DIR / 'include'}", f"-I{MLIR_AIE_DIR / 'include/aie_kernels'}",\n               f"-I{MLIR_AIE_DIR / 'include/aie_kernels/aie2p'}", "-c", str(src), "-o", str(obj)]\n        print(f"  Compiling {object_name} with Peano from {src.name}...")\n    else:\n        compiler = TOOLCHAIN_DIR / "bin" / "xchesscc_wrapper"\n        cmd = [str(compiler), "aie2p", f"-I{EXPERIMENT_DIR}", f"-I{AIETOOLS_DIR / 'include'}",\n               f"-I{MLIR_AIE_DIR / 'include'}", f"-I{MLIR_AIE_DIR / 'include/aie_kernels'}",\n               f"-I{MLIR_AIE_DIR / 'include/aie_kernels/aie2p'}", "-c", str(src), "-o", str(obj)]\n        print(f"  Compiling {object_name} with Chess from {src.name}...")\n    run_command(cmd)\n'''
    nb.write_text(s[:start]+helper+s[end:])
    changes.append('normalized _compile_aie_object')
    # Use the documented open-source aiecc path when Peano is selected.
    nb=ex/'npu_build.py'; s=nb.read_text()
    if '"--no-xchesscc"' not in s:
        old='''        "--no-compile-host",\n        "--alloc-scheme=basic-sequential",'''
        new='''        "--no-compile-host",\n        "--no-xchesscc",\n        "--no-xbridge",\n        "--peano", os.environ.get("PEANO_INSTALL_DIR", ""),\n        "--alloc-scheme=basic-sequential",'''
        if old not in s: raise SystemExit('aiecc command anchor missing')
        s=s.replace(old,new,1); nb.write_text(s); changes.append('aiecc uses Peano/no-xchesscc')
    # Build-only must not load an unrelated 0.6B/8B model. It still generates and validates the exact
    # graph; model/Q4NX validation is performed in the separate artifact job.
    if 'QWEN38_27B_BUILD_ONLY' not in runner.read_text():
        s=runner.read_text(); marker='''def build_only(\n    current_token: int | None = None,\n    model_path: Path | None = None,\n    layer: int = DEFAULT_LAYER,\n    download_model: bool = False,\n) -> bool:\n'''
        if marker not in s: raise SystemExit('build_only signature anchor missing')
        inject='''    if os.environ.get("QWEN38_27B_BUILD_ONLY") == "1":\n        target_schedule = _target_schedule(current_token)\n        build_schedule = _build_schedule(target_schedule)\n        xclbin_path, insts_path = build_kernel(build_schedule, _capacity_build_name(build_schedule))\n        print(f"  PASS: built {xclbin_path}")\n        print(f"  PASS: built {insts_path}")\n        return True\n\n'''
        s=s.replace(marker,marker+inject,1)
        s=s.replace('from pathlib import Path\n','from pathlib import Path\nimport os\n',1)
        runner.write_text(s); changes.append('graph-only build bypasses legacy model loader')
    for c in changes: print('PATCHED:',c)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); main(a.root)
''