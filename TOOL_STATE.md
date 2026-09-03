# Qwen3.8-27B NPU bring-up state

- Branch: `feat/qwen3.8-27b`
- Current head: `c20cc09d5215072160671e58d9c7d54f749f4ae2`
- Target is genuine Qwen3.8-27B only: H=5120, FFN=17408, vocab=248320, 64 layers, 24 attention heads, 4 KV heads, head_dim=256.
- Hybrid topology is exactly `[linear_attention, linear_attention, linear_attention, full_attention] * 16`.
- Historical 8B/9B XCLBINs are reference-only and are never accepted as 27B substitutes.
- Pinned torch2aie: `1e17e9fe0a0bc428b8a371dc8e4adbe69ee31fcc`.
- Pinned toolchain: `toolchain-2026-06-02`, archive SHA256 `70222aa334057fde23be96b8763fb63b53ff7615ec004ac16ecba3abb3d5aefb`.
- Pinned Q4NX converter: `d1d5232d0b82871bf5265e990fa2d26cdad22327`.
- Intended source GGUF: `bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_1.gguf`.
- Pending blob from earlier work: `dfc8ab4514859ab24c6979a27b1ff72afae4039f`.

## Historical generator findings

The historical `examples/qwen3-decode-layer` generator is a Qwen3 full-attention design. Its reference implementation does not contain a Qwen3.8 hybrid GDN/linear-attention layer implementation. The 27B adapter can correct GQA geometry (24 Q heads, 4 KV heads, head_dim 256), Q width 6144, O width 5120, up/gate 17408, down 5120, attention packet sizing, and the 4-window replay lock/BD assumptions, but this does NOT by itself implement the required three-GDN/one-full-attention repeating topology.

The required hybrid GDN decoding implementation is therefore not recovered from the pinned torch2aie tree. A generated full-attention XCLBIN must not be called a genuine Qwen3.8-27B layer artifact.

## Runtime ABI findings

FastFlowLM's NPU application-manager path uses an MLIR-AIE kernel interface and BO-backed runtime arguments. The existing Qwen3.5-9B lm-head binary is ABI-family reference only; it is never copied, renamed, or packaged as 27B.

The pinned asymmetric GEMM example has an 8-column layout. For `N=248320`, 128-wide output tiles give `248320/128=1940` tiles, which is not divisible by 8 but is exactly divisible by 4, giving 485 groups per AIE column. The lm-head work therefore changed the generated design to 4 columns and changed the B tensor tap partition together with it.

## Q4NX findings

The canonical Q4NX job successfully downloaded the exact filename `Qwen3.8-27B-Q4_1.gguf` from `bartowski/Qwen3.8-27B-GGUF` in run `33701251533`. The subsequent GGUF metadata probe failed because the GGUF omits the optional key `qwen35.mtp_num_hidden_layers`; this was not treated as proof of an incorrect model. A compatibility patch was added to the pinned converter environment, but the run using it was not yet completed to a verified Q4NX artifact/hash.

No Q4NX SHA256, byte size, tensor manifest, or final `model.q4nx` is recorded because those validations did not complete. Do not mark Q4NX PASS.

## Build/toolchain findings

The pinned hosted toolchain contains the raw AIE2P LLVM clang driver at `.../target_aie2p/.../llvm/bin/clang++`, but that driver rejects the `aie2p-none-unknown-elf` target as an unknown target and ignores `-arch aie2p`. The historical `xchesscc_wrapper` path requires an unavailable Xilinx license on the hosted runner. Attempts to bypass that with the raw clang driver therefore fail before producing AIE kernel objects. No genuine XCLBIN was produced.

## Previous execution findings

- Canonical run `33414548588`: geometry passed; Q4NX was cancelled during conversion; layer and lm_head failed on the proprietary `xchesscc`/AIEBuild license path.
- Canonical run `33625669477` (run 81, commit `a7ed8a5f30b059fb7fc40ddd950e84a78a345bc4`) failed immediately in geometry because the layer plan did not expose `num_attention_heads`, `num_key_value_heads`, and `head_dim` at the top level.
- Fixed that geometry contract in `21c8df31e50be364f877551d3ece23c8bef67899`.
- Canonical run `33625763903` then passed geometry but failed the three build jobs. lm_head failed because the venv created by pinned torch2aie has no `pip`; layer had the same hosted-toolchain/Peano bootstrap constraint, and Q4NX failed during its conversion/validation job. The lm-head log specifically showed `/home/runner/torch2aie/.venv/bin/python3: No module named pip`.
- Canonical run `33701251533` (run 116, head `6093d5fa283012a8b91f8077cefbdefa5a6a8f14`) failed all three artifact jobs: Q4NX at missing optional MTP metadata, lm_head at raw clang target selection, and layer at raw clang compilation of AIE intrinsics (`bfloat16`, `accfloat`, Chess pragmas) because the raw driver was not an AIE target driver.
- Canonical run `33701446270` (run 119, head `c20cc09d5215072160671e58d9c7d54f749f4ae2`) was started with the latest fixes. It reached the three expensive artifact jobs; their final results were failure, so package was skipped. No artifacts were uploaded.

## Current code changes

- Added/updated 27B geometry and Q4NX registry validation.
- Added Q4NX converter compatibility handling for GGUFs that encode the extra MTP block in `qwen35.block_count` without an explicit `qwen35.mtp_num_hidden_layers` key. This does not replace architecture verification.
- Adapted the pinned lm-head example to 4 AIE columns for the exact 248320 vocabulary geometry and kept K=5120/M=128.
- Added Peano/AIE compiler fallback attempts and explicit 27B attention replay lock/packet checks. These remain unproven because the hosted raw LLVM driver is not an AIE target compiler.

## Artifact gate

No artifact is marked PASS.

- `layer.xclbin`: NOT GENERATED / NOT VALIDATED.
- `lm_head.xclbin`: NOT GENERATED / NOT VALIDATED.
- `model.q4nx`: NOT GENERATED / NOT VALIDATED.
- Source GGUF SHA256: NOT RECORDED; the exact 27B filename was downloaded, but the metadata probe failed before the hash/manifest stage completed.
- `flm pull qwen3.8:27b`: NOT EXECUTED successfully against a verified package.
- Runtime smoke test: NOT EXECUTED.
- Hardware execution: NOT PERFORMED.
- BD/lock-memory/XCLBIN ABI validation: NOT PASSED.
- Canonical Actions run `33701446270` / run 119: FAILED.

## Critical remaining blockers

1. A real AIE2P kernel compilation path is required. The pinned hosted raw clang is not sufficient, while the Chess/AIEBuild path requires a license unavailable on the hosted runner.
2. A genuine Qwen3.8-27B hybrid decoder layer must be implemented or recovered; adapting the historical Qwen3 full-attention generator is insufficient because the required topology is GDN/GDN/GDN/full repeated 16 times.
3. The lm-head must be proven against the actual FastFlowLM runtime ABI and weight representation, not merely dimension-matched.
4. The Q4NX job must complete the real conversion and record source/Q4NX hashes and byte sizes.

Do not record PASS or SUCCESS until the underlying binary, metadata, ABI, BD/lock topology, Q4NX, package, `flm pull`, and hardware/offline validation claims actually execute. If physical XDNA2 hardware is unavailable, explicitly distinguish generated, structurally validated, runtime-loadable, and hardware-executed states.
