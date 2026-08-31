# Qwen3.8-27B XDNA2 build state

Target geometry: hidden 5120, FFN 17408, vocab 248320, 64 layers; full attention 24Q/4KV; linear/GDN 16Q/48V.

The canonical Qwen kernel workflow is `.github/workflows/qwen38-kernel-build.yml`.

Historical recovery targets:
- 1bit-MONSTER commit `5525db348bcfdbf455d6b6c30019d4c736103e69` (Q4NX / FastFlowLM extraction)
- `e0d479e72261e13c3744dab81b8a5cd6f87de740` (FLM-parity INT8 generator)
- `21e18923e7e5bfeaeda877a83dbf45baeafc733d` (fixture-scale fused full-layer generation)
- `715f8ec4a3c5fe11e4fef7eb212229d72c8d6a10` (open full-layer generator limitations)
- `962a6a15c6656c38fdc7dc98f916b5afe5a915ba` (toolchain/NPU triage)

Do not substitute Qwen3.5-9B XCLBINs for 27B. Completion requires genuine 27B `layer.xclbin` and `lm_head.xclbin`, ABI validation, Q4NX/model manifest wiring, `flm pull qwen3.8:27b`, green CI, and hardware execution validation.
