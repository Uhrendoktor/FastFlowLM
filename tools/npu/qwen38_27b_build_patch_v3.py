#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise SystemExit(f'anchor missing in {path}: {old[:120]!r}')
    path.write_text(text.replace(old, new, 1))
    return True


def patch_runner(runner: Path) -> bool:
    s = runner.read_text()
    if 'QWEN38_27B_BUILD_ONLY' in s:
        return False
    marker = '''def build_only(\n    current_token: int | None = None,\n    model_path: Path | None = None,\n    layer: int = DEFAULT_LAYER,\n    download_model: bool = False,\n) -> bool:\n'''
    if marker not in s:
        raise SystemExit('build_only signature not found')
    inject = '''    if os.environ.get("QWEN38_27B_BUILD_ONLY") == "1":\n        target_schedule = _target_schedule(current_token)\n        build_schedule = _build_schedule(target_schedule)\n        xclbin_path, insts_path = build_kernel(build_schedule, _capacity_build_name(build_schedule))\n        print(f"  PASS: built {xclbin_path}")\n        print(f"  PASS: built {insts_path}")\n        return True\n\n'''
    s = s.replace(marker, marker + inject, 1)
    s = s.replace('from pathlib import Path\n', 'from pathlib import Path\nimport os\n', 1)
    runner.write_text(s)
    return True


def patch_npu_build(nb: Path) -> bool:
    s = nb.read_text()
    start = s.find('def _compile_aie_object(')
    end = s.find('\ndef _linked_role_objects', start)
    if start < 0 or end < 0:
        raise SystemExit('cannot locate _compile_aie_object')
    helper = '''def _compile_aie_object(source_names: tuple[str, ...], object_name: str) -> None:\n    if not source_names:\n        raise ValueError(f"missing source for role object: {object_name}")\n    src = EXPERIMENT_DIR / source_names[0]\n    obj = _role_object_path(object_name)\n    obj.parent.mkdir(parents=True, exist_ok=True)\n    if os.environ.get("QWEN38_USE_PEANO") == "1":\n        peano = Path(os.environ["PEANO_INSTALL_DIR"])\n        compiler = peano / "bin" / "clang++"\n        cmd = [str(compiler), "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf",\n               "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-DNDEBUG",\n               f"-I{EXPERIMENT_DIR}", f"-I{AIETOOLS_DIR / 'include'}",\n               f"-I{MLIR_AIE_DIR / 'include'}", f"-I{MLIR_AIE_DIR / 'include/aie_kernels'}",\n               f"-I{MLIR_AIE_DIR / 'include/aie_kernels/aie2p'}", "-c", str(src), "-o", str(obj)]\n        print(f"  Compiling {object_name} with Peano from {src.name}...")\n    else:\n        compiler = TOOLCHAIN_DIR / "bin" / "xchesscc_wrapper"\n        cmd = [str(compiler), "aie2p", f"-I{EXPERIMENT_DIR}", f"-I{AIETOOLS_DIR / 'include'}",\n               f"-I{MLIR_AIE_DIR / 'include'}", f"-I{MLIR_AIE_DIR / 'include/aie_kernels'}",\n               f"-I{MLIR_AIE_DIR / 'include/aie_kernels/aie2p'}", "-c", str(src), "-o", str(obj)]\n        print(f"  Compiling {object_name} with Chess from {src.name}...")\n    run_command(cmd)\n'''
    s = s[:start] + helper + s[end:]
    if 'QWEN38_USE_PEANO' not in s:
        raise SystemExit('Peano helper insertion failed')
    anchor = '''    cmd = [\n        str(aiecc),\n        "-v",\n        f"-j{AIECC_JOBS}",\n        f"--aietools={AIETOOLS_DIR}",\n        "--no-compile-host",\n'''
    if anchor not in s:
        raise SystemExit('aiecc command anchor not found')
    inject = anchor + '''        *( ["--no-xchesscc", "--no-xbridge", "--peano", os.environ["PEANO_INSTALL_DIR"]] if os.environ.get("QWEN38_USE_PEANO") == "1" else [] ),\n'''
    s = s.replace(anchor, inject, 1)
    s = s.replace('Compiling MLIR-AIE with Chess-enabled aiecc...', 'Compiling MLIR-AIE with Peano/chess-selected aiecc...')
    nb.write_text(s)
    return True


def main(root: Path) -> int:
    root = root.resolve()
    ex = root / 'examples/qwen3-decode-layer'
    gen = ex / 'cases/full_layer_engine_generate.py'
    runner = ex / 'cases/qwen3_8b_decode_layer_runner.py'
    nb = ex / 'npu_build.py'
    changes=[]
    for old,new,msg in [
        ('return_window_dwords != 512','return_window_dwords != 768','return window 768 dwords'),
        ('require_source_side_packet_replay(source, 8)','require_source_side_packet_replay(source, 4)','source replay locks 4'),
        ('require_source_side_packet_replay(src, 8)','require_source_side_packet_replay(src, 4)','source replay locks 4'),
        ('require_attention_block_shapes(hub, WEIGHT_DWORDS * 8)','require_attention_block_shapes(hub, ATTENTION_OUTPUT_DWORDS)','attention output packet width'),
    ]:
        if replace_once(gen,old,new): changes.append(msg)
    if patch_runner(runner): changes.append('graph-only runner mode')
    patch_npu_build(nb); changes.append('Peano-capable AIE object/MLIR compilation')
    for c in changes: print('PATCHED:',c)
    return 0

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); raise SystemExit(main(a.root))
