#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, shutil, subprocess
from pathlib import Path

K = 5120
N = 248320
M = 128
AIE_COLS = 4


def configure_peano_makefile(cfg: Path, peano: Path) -> None:
    p = cfg.parent / 'makefile_common'
    s = p.read_text()
    s = s.replace('use_chess?=1', 'use_chess?=0')
    s = re.sub(r'-triple\s+aie2p-none-unknown-elf', '--target=aie2p-none-unknown-elf', s)
    # Keep the pinned compiler selected even if the historical makefile uses :=/+= syntax.
    s = re.sub(r'^KERNEL_CC\s*[:?+]?=\s*.*$', f'KERNEL_CC={peano}/bin/clang++', s, flags=re.MULTILINE)
    s = re.sub(r'^KERNEL_CFLAGS\s*[:?+]?=\s*.*$',
               'KERNEL_CFLAGS=--target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG '
               '-Wno-parentheses -Wno-attributes -Wno-macro-redefined '
               '-I ${AIETOOLS_DIR}/include -I ${MLIR_AIE_DIR}/include',
               s, flags=re.MULTILINE)
    if '--no-xchesscc' not in s:
        s = s.replace('aiecc_chess_flags=--unified',
                      f'aiecc_chess_flags=--no-xchesscc --no-xbridge --peano {peano}', 1)
    p.write_text(s)
    m = cfg / 'Makefile'
    m.write_text(m.read_text().replace('kernelsrc := mm_bfp_mixed.cc', 'kernelsrc := mm_bf16.cc'))


def configure_lm_head_design(cfg: Path) -> None:
    """Adapt the pinned 8-column design to the exact 27B vocab geometry.

    N/128 = 1940 output tiles. 1940 is not divisible by 8, but is exactly
    divisible by 4, giving 485 complete groups per AIE column. The B tensor
    tiling must change with the column count; otherwise the last vocabulary
    tiles are omitted or the taps alias the weight matrix.
    """
    src = cfg / 'n1_core_bf16.py'
    text = src.read_text()
    if 'n_aie_cols = 8' not in text:
        raise SystemExit('unexpected n1_core_bf16.py: 8-column anchor missing')
    text = text.replace('n_aie_cols = 8', f'n_aie_cols = {AIE_COLS}', 1)
    old = 'B_taps = TensorTiler2D.group_tiler((1, N * K // 8), (1, n * K // 8), (1, 1))'
    new = 'B_taps = TensorTiler2D.group_tiler((1, N * K // n_aie_cols), (1, n * K // n_aie_cols), (1, 1))'
    if old not in text:
        raise SystemExit('unexpected n1_core_bf16.py: 8-column B tap anchor missing')
    text = text.replace(old, new, 1)
    src.write_text(text)


def main(root: Path) -> int:
    root = root.resolve()
    cfg = root / 'examples/gemm_asymmetric_tile_buffering/config1'
    peano_env = os.environ.get('PEANO_INSTALL_DIR')
    if not peano_env:
        raise SystemExit('PEANO_INSTALL_DIR must point at pinned toolchain Peano')
    if N % 128:
        raise SystemExit(f'vocab N={N} must be divisible by 128')
    if (N // 128) % AIE_COLS:
        raise SystemExit(f'vocab tile count {N // 128} must be divisible by AIE_COLS={AIE_COLS}')
    peano = Path(peano_env)
    if not (peano / 'bin/clang++').is_file():
        raise SystemExit(f'pinned Peano compiler missing: {peano / "bin/clang++"}')

    configure_peano_makefile(cfg, peano)
    configure_lm_head_design(cfg)
    subprocess.run(['make', 'clean'], cwd=cfg, check=False)

    env = os.environ.copy()
    env.update({
        'QWEN38_LM_HEAD_K': str(K),
        'QWEN38_LM_HEAD_N': str(N),
        'QWEN38_LM_HEAD_M': str(M),
        'QWEN38_LM_HEAD_AIE_COLS': str(AIE_COLS),
    })
    subprocess.run(
        ['make', f'M={M}', f'K={K}', f'N={N}',
         'targetname=n1_core', 'aie_py_src=n1_core_bf16.py', 'use_chess=0'],
        cwd=cfg,
        check=True,
        env=env,
    )

    xs = sorted(cfg.glob('build/*.xclbin'), key=lambda p: p.stat().st_mtime, reverse=True)
    ms = sorted(cfg.glob('build/*.mlir'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xs:
        raise SystemExit('no lm_head xclbin')
    x = xs[0]
    m = ms[0] if ms else None
    if m:
        t = m.read_text()
        for needle in (str(K), str(N), str(M), 'MLIR_AIE'):
            if needle not in t:
                raise SystemExit(f'generated MLIR missing required marker {needle!r}')

    out = Path(os.environ.get('QWEN38_LM_HEAD_OUT', root / 'lm_head.xclbin'))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(x, out)
    if m:
        shutil.copy2(m, out.with_suffix('.mlir'))

    import json
    manifest = {
        'M': M, 'K': K, 'N': N,
        'output_tiles': N // 128,
        'aie_cols': AIE_COLS,
        'groups_per_column': (N // 128) // AIE_COLS,
        'kernel': 'MLIR_AIE',
        'input_dtype': 'bf16', 'weight_dtype': 'bf16', 'output_dtype': 'bf16',
        'abi_source': 'torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_bf16.py + mm_bf16.cc',
        'targetname': 'n1_core',
        'compiler': 'pinned-toolchain-peano',
        'geometry_proof': '248320/128=1940 output tiles; 1940/4=485 groups per AIE column',
    }
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out)
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    a = ap.parse_args()
    raise SystemExit(main(a.root))
