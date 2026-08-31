# Qwen3.8-27B NPU bring-up state

- Branch: `feat/qwen3.8-27b`
- Target: Qwen3.8-27B only — H=5120, FFN=17408, vocab=248320, 64 layers, 16 full-attention layers at indices 3,7,...,63; remaining layers are hybrid linear/GDN.
- Historical 8B/9B XCLBINs are reference-only and are never used as 27B substitutes.
- Pinned torch2aie: `1e17e9fe0a0bc428b8a371dc8e4adbe69ee31fcc`.
- Pinned toolchain release: `toolchain-2026-06-02`, archive SHA256 `70222aa334057fde23be96b8763fb63b53ff7615ec004ac16ecba3abb3d5aefb`.
- Pinned Q4NX converter: `d1d5232d0b82871bf5265e990fa2d26cdad22327`.
- Genuine source: `bartowski/Qwen3.8-27B-GGUF`, Q4_1 file.

## Historical-generator adaptation

Run `33398992567` exposed stale 0.6B assumptions: helper count 5 vs 4 (definition token counted), carrier 60 vs stale 80 dwords, return 384 vs stale 512, writebd IDs 16–19 outside 0..15, and hub-BD contract/bank mismatches.

The adapter now uses the genuine 27B full-attention geometry: Q=24, KV=4, head_dim=256, GQA=6, four 6Q/1KV windows; Q width=6144; O width=5120; up/gate=17408; down=5120. Weight spans use 384 chunks so generated NPU writebd IDs remain within 0..15. Hub Q/return BDs retain the historical bank-safe mapping, while the 27B O path gets ten non-overlapping high-bank BDs.

## Validation progress

- Run `33400923176`: structural validation still failed only because the historical validator passed `WEIGHT_DWORDS * 8` (384) as attention output width; this was corrected to the genuine 27B `ATTENTION_OUTPUT_DWORDS` (768).
- The following run `33400923176` successor passed the historical full-layer structural validation and advanced beyond the generator stage. No XCLBIN success has been claimed yet.
- The Q4NX workflow was fixed to use the pinned converter's `create_converter(...).convert(...)` API instead of the obsolete CLI invocation. It now validates the official Qwen3.8-27B `config.json` dimensions and emits a manifest containing the source GGUF SHA256 and 27B geometry.
- Canonical workflow run currently active: `33401225836`, still in pinned toolchain bootstrap.

## Artifact gate

Do not claim success until genuine 27B `/tmp/layer.xclbin` and `/tmp/lm_head.xclbin` are generated, `xclbinutil --info` ABI metadata is checked, tensor dimensions are 5120/17408/248320 as applicable, BD-lock-memory topology is validated, the genuine 27B Q4NX `model.q4nx` and manifest are validated, and `flm pull qwen3.8:27b` resolves to the resulting artifact package. 
