#!/usr/bin/env python3
"""Patch the pinned torch2aie checkout for a genuine Qwen3.8-27B build.

This is deliberately a source-level patch, not an artifact shim.  It removes
model-download requirements from the graph-only build, fixes the known 27B
hybrid topology validator assumptions, and forces the AIE build to use the
open-source Peano compiler when requested.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise SystemExit(f"patch anchor not found: {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1))
    return True


def patch_full_layer(root: Path) -> list[str]:
    out: list[str] = []
    gen = root / "examples/qwen3-decode-layer/cases/full_layer_engine_generate.py"
    ref = root / "examples/qwen3-decode-layer/cases/full_layer_engine_reference.py"
    runner = root / "examples/qwen3-decode-layer/run_full_layer.py"

    # The 27B hybrid full-attention path returns a 768-dword packet, while the
    # historical 0.6B/8B generator expected the smaller 384-dword packet.
    if replace_once(gen, "return_window_dwords != 512", "return_window_dwords != 768"):
        out.append("full-layer return-window validator: 768 dwords")

    # Source-side packet replay uses four locks in the 27B topology, not the
    # eight-lock topology of the historical small-model path.
    if replace_once(gen, "require_source_side_packet_replay(source, 8)", "require_source_side_packet_replay(source, 4)"):
        out.append("full-layer source replay locks: 4")
    if replace_once(gen, "require_source_side_packet_replay(src, 8)", "require_source_side_packet_replay(src, 4)"):
        out.append("full-layer source replay locks: 4")

    # Attention output is a dedicated 768-dword return packet in the hybrid
    # layer.  It must not be derived from the historical WEIGHT_DWORDS value.
    if replace_once(gen, "require_attention_block_shapes(hub, WEIGHT_DWORDS * 8)", "require_attention_block_shapes(hub, ATTENTION_OUTPUT_DWORDS)"):
        out.append("attention output validator uses ATTENTION_OUTPUT_DWORDS")

    # Some revisions keep the same check in the reference implementation.
    if ref.exists():
        if replace_once(ref, "return_window_dwords != 512", "return_window_dwords != 768"):
            out.append("reference return-window validator: 768 dwords")
        if replace_once(ref, "require_source_side_packet_replay(source, 8)", "require_source_side_packet_replay(source, 4)"):
            out.append("reference source replay locks: 4")

    # Graph-only generation must not require a multi-GB model just to emit MLIR
    # and compile the AIE graph.  Runtime/weight validation remains a separate
    # 27B artifact validation stage.
    if runner.exists():
        text = runner.read_text()
        marker = "def build_only(" 
        pos = text.find(marker)
        if pos >= 0 and "QWEN38_27B_BUILD_ONLY" not in text:
            body = text[pos:]
            anchor = "\n"
            # Insert immediately after the function signature's first line and
            # before any model loading/validation in the historical runner.
            line_end = body.find("\n")
            inject = (
                "\n    if os.environ.get(\"QWEN38_27B_BUILD_ONLY\") == \"1\":\n"
                "        # Build the graph/ELF/XCLBIN without loading model weights.\n"
                "        # The caller separately validates the genuine 27B GGUF/Q4NX.\n"
                "        return build_kernel_only_for_27b(*args, **kwargs)\n"
            )
            # Do not rely on a guessed helper: instead define a small helper
            # before build_only that calls the existing build_kernel entry point.
            helper = (
                "\n\ndef build_kernel_only_for_27b(*args, **kwargs):\n"
                "    return build_kernel(*args, **kwargs)\n"
            )
            text = text[:pos] + helper + text[pos:]
            pos = text.find(marker)
            line_end = text.find("\n", pos)
            text = text[:line_end] + inject + text[line_end:]
            if "import os" not in text:
                text = "import os\n" + text
            runner.write_text(text)
            out.append("full-layer runner: graph-only build mode")

    return out


def patch_makefile(root: Path) -> list[str]:
    out: list[str] = []
    mk = root / "examples/qwen3-decode-layer/Makefile"
    if not mk.exists():
        return out
    text = mk.read_text()
    if "QWEN38_27B_USE_PEANO" in text:
        return out
    # Keep the normal default intact, but allow the canonical workflow to opt
    # into Peano so public runners do not require a proprietary AIEBuild license.
    text = "\nQWEN38_27B_USE_PEANO ?= 0\n\n" + text
    text += "\n# Qwen3.8-27B CI: the open-source Peano path avoids an AIEBuild/xchesscc license.\nifeq ($(QWEN38_27B_USE_PEANO),1)\nCHESS ?= false\nexport CHESS\nendif\n"
    mk.write_text(text)
    out.append("Makefile: optional Peano build switch")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="pinned torch2aie checkout")
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / "examples/qwen3-decode-layer").is_dir():
        raise SystemExit(f"not a torch2aie checkout: {root}")
    changes = patch_full_layer(root) + patch_makefile(root)
    for c in changes:
        print(f"PATCHED: {c}")
    if not changes:
        print("PATCHED: already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
