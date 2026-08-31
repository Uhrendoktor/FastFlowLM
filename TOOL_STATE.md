# Qwen3.8-27B NPU bring-up state

- Branch: `feat/qwen3.8-27b`
- Target: Qwen3.8-27B only — H=5120, FFN=17408, vocab=248320, 64 layers, 16 full-attention layers at indices 3,7,...,63; remaining layers are hybrid linear/GDN.
- Historical 8B/9B XCLBINs are reference-only and are never used as 27B substitutes.
- Pinned torch2aie: `1e17e9fe0a0bc428b8a371dc8e4adbe69ee31fcc`.
- Pinned toolchain release: `toolchain-2026-06-02`, archive SHA256 `70222aa334057fde23be96b8763fb63b53ff7615ec004ac16ecba3abb3d5aefb`.
- Pinned Q4NX converter: `d1d5232d0b82871bf5265e990fa2d26cdad22327`.
- Genuine source: `bartowski/Qwen3.8-27B-GGUF`, Q4_1 file; source repository verified externally and the canonical workflow requests that exact 27B artifact.

## Latest blocker and adaptation

Run `33398992567` reached historical generator structural validation and failed with:
- attention helper counts 5 vs 4 (definition token is included in textual count; expected count must remain 5 for four runtime windows);
- carrier 60 vs stale 80 expectation;
- return window 384 vs stale 512 expectation;
- `npu.writebd` IDs 16–19 outside the 0..15 limit;
- hub BD contract mismatch.

The adapter was then updated to:
- use 24 Q heads / 4 KV heads, head_dim 256, GQA ratio 6;
- use four Q/KV windows (6 Q heads + 1 KV head per window);
- derive attention window dwords from the 27B packet geometry;
- keep the historical function-occurrence validator at 5;
- constrain weight-span/writebd IDs to 0..15 by using 384-chunk spans and eight span BDs;
- align the attention-dataflow hub BD tuples with the four-window compact fabric;
- update 27B projection/weight chunk counts and bases.

The canonical workflow was also corrected to invoke the pinned MLIR-AIE `aiecc --aie-generate-xclbin --aie-generate-npu-insts` flow instead of a nonexistent `npu_build` module.

## Not yet success

No genuine 27B `layer.xclbin` or `lm_head.xclbin` has been generated/validated yet. Do not claim success until both exist and metadata/ABI/BD-lock-memory topology checks pass, and Q4NX weights + manifest are validated.
