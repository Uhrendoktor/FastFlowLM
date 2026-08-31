#!/usr/bin/env python3
"""Adapt the recovered torch2aie full-layer generator to Qwen3.8-27B.

The historical contract is retained verbatim except for dimension-dependent
constants. Keeping its helper functions and validators is important: the
full-layer generator relies on the same Q4NX projection/BD/lock ABI that FLM
uses in production.
"""
from pathlib import Path
import sys

H=5120
IM=17408
HD=256
NQ=24
NKV=4


def patch(root: Path) -> None:
    p=root/'examples/qwen3-decode-layer/contract.py'
    text=p.read_text()
    replacements={
      'HIDDEN_DIM = 1024':'HIDDEN_DIM = 5120',
      'INTERMEDIATE_DIM = 3072':'INTERMEDIATE_DIM = 17408',
      'HEAD_DIM = 128':'HEAD_DIM = 256',
      'NUM_Q_HEADS = 16':'NUM_Q_HEADS = 24',
      'NUM_KV_HEADS = 8':'NUM_KV_HEADS = 4',
      'C6R2_INPUT_DWORDS = 512':'C6R2_INPUT_DWORDS = HIDDEN_DIM // 2',
      '# === Qwen3-0.6B specific aux sizes ===':'# === Qwen3.8-27B specific aux sizes ===',
      '# = 1024 + (3 * 128) / 2 = 1216':'# = 5120 + (3 * 256) / 2 = 5504',
      '# = 1024 + (3 * 128) / 2 = 1216 i32':'# = 5120 + (3 * 256) / 2 = 5504 i32',
    }
    for a,b in replacements.items():
        if a not in text and a.startswith(('HIDDEN','INTERMEDIATE','HEAD_DIM','NUM_','C6R2')):
            raise RuntimeError(f'missing expected historical contract token: {a}')
        text=text.replace(a,b)

    # The historical validator contains literal 0.6B expectations. Replace it
    # with a dimension-derived validator so the structural generator remains
    # authoritative instead of bypassing its checks.
    marker='def validate_contract() -> list[str]:'
    if marker not in text:
        raise RuntimeError('historical validate_contract helper missing')
    text=text[:text.index(marker)] + '''def validate_contract() -> list[str]:
    errors: list[str] = []
    expected_blocks = tuple(x // OUTPUT_BLOCK_ROWS for x in PHASE_OUTPUT_DIMS)
    expected_chunks = tuple(x // K_CHUNK for x in PHASE_INPUT_DIMS)
    if PHASE_BLOCKS != expected_blocks:
        errors.append(f"phase blocks mismatch: {PHASE_BLOCKS} != {expected_blocks}")
    if PHASE_CHUNKS != expected_chunks:
        errors.append(f"phase chunks mismatch: {PHASE_CHUNKS} != {expected_chunks}")
    if (HIDDEN_DIM, INTERMEDIATE_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS) != (5120,17408,256,24,4):
        errors.append("not the Qwen3.8-27B contract")
    if GQA_RATIO != 6:
        errors.append(f"GQA ratio mismatch: {GQA_RATIO}")
    if C6R2_INPUT_DWORDS != HIDDEN_DIM // 2:
        errors.append(f"C6R2 input mismatch: {C6R2_INPUT_DWORDS}")
    if HIDDEN_DWORDS != HIDDEN_DIM // 2 or OUTPUT_DWORDS != HIDDEN_DIM // 2:
        errors.append(f"host dword mismatch: {HIDDEN_DWORDS}/{OUTPUT_DWORDS}")
    return errors
'''
    p.write_text(text)

    h=root/'examples/qwen3-decode-layer/qwen3_constants.h'
    text=h.read_text()
    repl={
      'kQBodyRecords = 4':'kQBodyRecords = 12',
      'kOBodyRecords = 2':'kOBodyRecords = 10',
      'kUpGateReplays = 12':'kUpGateReplays = 68',
      'kDownBodyRecords = 2':'kDownBodyRecords = 10',
      'kQChunksPerRecord = 4':'kQChunksPerRecord = 20',
      'kKvChunksPerRecord = 4':'kKvChunksPerRecord = 20',
      'kOChunksPerRecord = 8':'kOChunksPerRecord = 24',
      'kUpGateChunksPerReplay = 4':'kUpGateChunksPerReplay = 20',
      'kDownChunksPerRecord = 12':'kDownChunksPerRecord = 68',
      'kKWeightChunkBase = 16':'kKWeightChunkBase = 240',
      'kVWeightChunkBase = 24':'kVWeightChunkBase = 280',
      'kFullLayerOWeightChunkBase = 32':'kFullLayerOWeightChunkBase = 320',
      'kFullLayerUpGateWeightChunkBase = 48':'kFullLayerUpGateWeightChunkBase = 560',
      'kFullLayerDownWeightChunkBase = 96':'kFullLayerDownWeightChunkBase = 1920',
    }
    for a,b in repl.items():
        text=text.replace(a,b)
    text=text.replace('Qwen3-0.6B specific','Qwen3.8-27B specific')
    h.write_text(text)

    print('patched historical full-layer generator contract for Qwen3.8-27B')

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: qwen38_27b_torch2aie_adapter.py TORCH2AIE_ROOT')
    patch(Path(sys.argv[1]))
