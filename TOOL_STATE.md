# Qwen3.8-27B NPU bring-up state

- Branch: `feat/qwen3.8-27b`
- Current head: `0e518c952a96b4e856a75885b561e3df22d34f29`
- Target is genuine Qwen3.8-27B only: H=5120, FFN=17408, vocab=248320, 64 layers, 24 attention heads, 4 KV heads, head_dim=256.
- Hybrid topology is exactly `[linear_attention, linear_attention, linear_attention, full_attention] * 16`.
- Historical 8B/9B XCLBINs are reference-only and are never accepted as 27B substitutes.
- Pinned torch2aie: `1e17e9fe0a0bc428b8a371dc8e4adbe69ee31fcc`.
- Pinned toolchain: `toolchain-2026-06-02`, archive SHA256 `70222aa334057fde23be96b8763fb63b53ff7615ec004ac16ecba3abb3d5aefb`.
- Pinned Q4NX converter: `d1d5232d0b82871bf5265e990fa2d26cdad22327`.
- Intended source GGUF: `bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_1.gguf`.
- Pending blob from earlier work: `dfc8ab4514859ab24c6979a27b1ff72afae4039f`.

## Historical generator findings

The previous runs exposed stale small-model assumptions in the historical full-layer generator. The 27B adaptation requires the genuine GQA geometry (24 Q heads, 4 KV heads, head_dim 256), four 6Q/1KV windows, Q width 6144, O width 5120, up/gate 17408, down 5120, 384-dword weight chunks, a 768-dword attention return packet, and the 27B lock/BD topology. The adapter already corrected the attention-output validator and related stale lock/topology assumptions.

## Previous execution findings

- Canonical run `33414548588` geometry passed.
- Q4NX job was cancelled while conversion was still running.
- `lm_head` failed before generation because `xchesscc_wrapper` requires an unavailable AIEBuild license on the hosted Ubuntu runner.
- `layer` reached the same AIEBuild/xchesscc license failure after the historical structural validation stage.
- The exact failure was: `AIEBuild license not found` from `xchesscc`.
- The pinned toolchain contains the Chess path but the hosted runner does not provide a proprietary AIEBuild license. A Peano path was therefore added to the canonical build flow; it is used only if the pinned archive actually contains an identifiable Peano clang++.

## Changes made after the failed run

- Added `tools/npu/qwen38_27b_build_patch_v3.py` to patch the pinned torch2aie layer runner/generator and select Peano for AIE object/MLIR compilation.
- Added `tools/npu/qwen38_27b_lm_head_build_v2.py` to generate K=5120, N=248320, M=128 lm_head using the pinned MLIR-AIE GEMM ABI and the Peano compiler path instead of Chess.
- Added `tools/npu/qwen38_27b_xclbin_validator.py` for non-placeholder XCLBIN parsing plus MLIR lock/BD/runtime-kernel checks and exact 27B manifest checks.
- The canonical workflow was rewritten to split geometry, Q4NX, layer, lm_head, and final packaging into bounded jobs with retained artifacts.
- The lm_head job compares the generated kernel metadata against the existing Qwen3.5-9B `lm_head.xclbin` as a runtime-ABI reference only; the 9B binary is never used as a 27B artifact.
- The package job is wired to execute `flm pull qwen3.8:27b` against the generated local package.

## Artifact gate

No artifact is marked PASS yet. At the time of this update there is **no verified 27B `layer.xclbin`, no verified 27B `lm_head.xclbin`, and no verified 27B `model.q4nx` hash** from the new flow. No hardware execution has been performed.

Do not record PASS until the underlying artifact, ABI, dimensions, BD/lock-memory topology, Q4NX metadata/hash, package installation, and `flm pull qwen3.8:27b` validations have actually executed. Final hashes and the successful Actions run number must be appended here after execution.

## Actions execution limitation

The GitHub connector available in this session can update repository contents and inspect/rerun existing Actions jobs, but it does not expose the `workflow_dispatch` write endpoint. Repository-content commits made through the connector did not create a new Actions run in the available run/status APIs. The canonical workflow therefore has been prepared and committed, but the new workflow has not yet executed in this session; this is intentionally recorded as a limitation rather than a false PASS.
