# Qwen3.8-27B NPU bring-up state

- Branch: `feat/qwen3.8-27b`
- Current head: `6cb76c9a71a8959e31ecf4a9f56b7414dc541812`
- Target is genuine Qwen3.8-27B only: H=5120, FFN=17408, vocab=248320, 64 layers, 24 attention heads, 4 KV heads, head_dim=256.
- Hybrid topology is exactly `[linear_attention, linear_attention, linear_attention, full_attention] * 16`.
- Historical 8B/9B XCLBINs are reference-only and are never accepted as 27B substitutes.
- Pinned torch2aie: `1e17e9fe0a0bc428b8a371dc8e4adbe69ee31fcc`.
- Pinned toolchain: `toolchain-2026-06-02`, archive SHA256 `70222aa334057fde23be96b8763fb63b53ff7615ec004ac16ecba3abb3d5aefb`.
- Pinned Q4NX converter: `d1d5232d0b82871bf5265e990fa2d26cdad22327`.
- Intended source GGUF: `bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_1.gguf`.
- Pending blob from earlier work: `dfc8ab4514859ab24c6979a27b1ff72afae4039f`.

## Historical generator findings

The historical full-layer generator contains stale small-model assumptions. The 27B adaptation uses genuine GQA geometry (24 Q heads, 4 KV heads, head_dim 256), Q width 6144, O width 5120, up/gate 17408, down 5120, 384-dword weight chunks, a 768-dword attention return packet, and the 27B lock/BD topology. The adapter also corrects stale attention-output and source replay lock assumptions.

## Runtime ABI findings

FastFlowLM's NPU application manager creates an `MLIR_AIE` kernel interface and invokes the runtime kernel with the three scalar control arguments followed by BO arguments. The lm-head generator is therefore required to produce the MLIR-AIE GEMM ABI rather than an arbitrary dimension-matched GEMM. The existing Qwen3.5-9B lm-head XCLBIN is used only as an ABI-family reference; it is never copied, renamed, or packaged as 27B.

## Previous execution findings

- Canonical run `33414548588`: geometry passed; Q4NX was cancelled during conversion; layer and lm_head failed on the proprietary `xchesscc`/AIEBuild license path.
- Canonical run `33625669477` (run 81, commit `a7ed8a5f30b059fb7fc40ddd950e84a78a345bc4`) failed immediately in geometry because the layer plan did not expose `num_attention_heads`, `num_key_value_heads`, and `head_dim` at the top level.
- Fixed that geometry contract in `21c8df31e50be364f877551d3ece23c8bef67899`.
- Canonical run `33625763903` then passed geometry but failed the three build jobs. lm_head failed because the venv created by pinned torch2aie has no `pip`; layer had the same hosted-toolchain/Peano bootstrap constraint, and Q4NX failed during its conversion/validation job. The lm-head log specifically showed `/home/runner/torch2aie/.venv/bin/python3: No module named pip`.
- The build scripts were changed to install a pinned open-source `llvm-aie`/Peano wheel through `uv`, not through the unavailable venv `pip`.
- Supplemental Peano wheel pinned for hosted execution: `llvm_aie-21.0.0.2026061701+742b6c9b-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, SHA256 `01da9bf7fd6d6fc86a77f85c5841fc3d74508eae118f6c0e5007ae48bacc748a`. This does not replace the pinned torch2aie/toolchain archive; it supplies the open-source AIE core compiler absent from the hosted runner.

## Current execution

- Execution commit: `6cb76c9a71a8959e31ecf4a9f56b7414dc541812`.
- Canonical pull-request execution run: `33625944114`.
- At the last inspection, geometry had completed successfully and the q4nx, layer, and lm_head jobs were executing. No final artifact is marked PASS yet.
- An execution-only PR `#3` / branch `ci/qwen3.8-27b-canonical` was created solely to obtain a `pull_request`-triggered canonical run because content-API commits do not themselves trigger a new push workflow. The implementation remains on `feat/qwen3.8-27b`.

## Artifact gate

No artifact is marked PASS yet. There is currently no verified final 27B `layer.xclbin`, no verified final 27B `lm_head.xclbin`, and no verified final 27B `model.q4nx` hash recorded here. No hardware execution has been performed.

Do not record PASS until the underlying artifact, XCLBIN sections, runtime ABI, dimensions, BD/lock-memory topology, Q4NX metadata/hash, package installation, and `flm pull qwen3.8:27b` validations have actually executed. Final hashes, sizes, successful Actions run number/status, package result, and hardware status must be appended after execution.
