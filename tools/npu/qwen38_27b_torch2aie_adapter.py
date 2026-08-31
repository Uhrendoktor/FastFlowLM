#!/usr/bin/env python3
"""Adapt the recovered torch2aie Qwen3 full-layer generator to Qwen3.8-27B."""
from pathlib import Path
import sys


def replace(path: Path, replacements: dict[str,str]) -> None:
    text=path.read_text()
    for old,new in replacements.items():
        if old not in text:
            raise RuntimeError(f'{path}: missing historical token: {old}')
        text=text.replace(old,new)
    path.write_text(text)


def patch(root: Path) -> None:
    ex=root/'examples/qwen3-decode-layer'
    p=ex/'contract.py'
    replace(p, {
      'HIDDEN_DIM = 1024':'HIDDEN_DIM = 5120',
      'INTERMEDIATE_DIM = 3072':'INTERMEDIATE_DIM = 17408',
      'HEAD_DIM = 128':'HEAD_DIM = 256',
      'NUM_Q_HEADS = 16':'NUM_Q_HEADS = 24',
      'NUM_KV_HEADS = 8':'NUM_KV_HEADS = 4',
      'C6R2_INPUT_DWORDS = 512':'C6R2_INPUT_DWORDS = HIDDEN_DIM // 2',
      '# === Qwen3-0.6B specific aux sizes ===':'# === Qwen3.8-27B specific aux sizes ===',
    })
    text=p.read_text()
    marker='def validate_contract() -> list[str]:'
    if marker not in text: raise RuntimeError('historical validate_contract helper missing')
    text=text[:text.index(marker)] + '''def validate_contract() -> list[str]:
    errors=[]
    expected_blocks=tuple(x//OUTPUT_BLOCK_ROWS for x in PHASE_OUTPUT_DIMS)
    expected_chunks=tuple(x//K_CHUNK for x in PHASE_INPUT_DIMS)
    if PHASE_BLOCKS!=expected_blocks: errors.append(f"phase blocks mismatch: {PHASE_BLOCKS} != {expected_blocks}")
    if PHASE_CHUNKS!=expected_chunks: errors.append(f"phase chunks mismatch: {PHASE_CHUNKS} != {expected_chunks}")
    if (HIDDEN_DIM,INTERMEDIATE_DIM,HEAD_DIM,NUM_Q_HEADS,NUM_KV_HEADS)!=(5120,17408,256,24,4): errors.append('not the Qwen3.8-27B contract')
    if GQA_RATIO!=6: errors.append(f'GQA ratio mismatch: {GQA_RATIO}')
    if C6R2_INPUT_DWORDS!=HIDDEN_DIM//2: errors.append(f'C6R2 input mismatch: {C6R2_INPUT_DWORDS}')
    return errors
'''
    p.write_text(text)

    replace(ex/'qkv_compact_reference.py', {
      'WINDOW_DWORDS = 512':'WINDOW_DWORDS = ATTENTION_PACKET_DWORDS // 4',
    })
    replace(ex/'cases/attention_block_reference.py', {
      'HEADS_PER_WINDOW = 8':'HEADS_PER_WINDOW = 6',
      'KV_HEADS_PER_WINDOW = 2':'KV_HEADS_PER_WINDOW = 1',
      'HEAD_DIM = 128':'HEAD_DIM = 256',
    })
    replace(ex/'cases/decode_cache_reference.py', {'KV_HEADS = 8':'KV_HEADS = 4'})
    replace(ex/'attention_dataflow.py', {
      'next_start = f"^q{window + 1}_start" if window + 1 < 4 else "^return0_start"':'next_start = f"^q{window + 1}_start" if window + 1 < len(HUB_Q_OUT_BDS) else "^return0_start"',
      'next_start = f"^return{window + 1}_start" if window + 1 < 4 else "^packet_out_start"':'next_start = f"^return{window + 1}_start" if window + 1 < len(HUB_RETURN_IN_BDS) else "^packet_out_start"',
    })
    c=ex/'compact_dataflow.py'
    t=c.read_text()
    for a,b in {
      'HUB_Q_OUT_CHANNELS = (1, 2, 3, 4, 1, 2, 3, 4)':'HUB_Q_OUT_CHANNELS = (1, 2, 3, 4)',
      'HUB_Q_OUT_BDS = (25, 2, 26, 3, 37, 5, 36, 8)':'HUB_Q_OUT_BDS = (25, 2, 26, 3)',
      'HUB_RETURN_IN_CHANNELS = (2, 3, 4, 5, 2, 3, 4, 5)':'HUB_RETURN_IN_CHANNELS = (2, 3, 4, 5)',
      'HUB_RETURN_IN_BDS = (4, 28, 6, 30, 9, 38, 11, 39)':'HUB_RETURN_IN_BDS = (4, 28, 6, 30)',
      'HUB_DOWN_OUT_BDS = (27, 29, 31, 32, 33, 42, 43, 44)':'HUB_DOWN_OUT_BDS = (27, 29, 31, 32, 33, 42, 43, 44, 45, 46)',
      'HUB_WINDOWS = 8':'HUB_WINDOWS = 4',
    }.items(): t=t.replace(a,b)
    c.write_text(t)

    g=ex/'cases/full_layer_engine_generate.py'
    t=g.read_text()
    t=t.replace('"aie.use_lock(%hub_return_full, AcquireGreaterEqual, 8)"','"aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)"')
    t=t.replace('"aie.use_lock(%hub_return_empty, Release, 8)"','"aie.use_lock(%hub_return_empty, Release, 4)"')
    for name in ('make_carrier_masked','init_accum','accum_block','finish_accum'):
        t=t.replace(f'"qwen3_attention_bf16_{name}", mlir.count("qwen3_attention_bf16_{name}"), 5', f'"qwen3_attention_bf16_{name}", mlir.count("qwen3_attention_bf16_{name}"), 4')
    g.write_text(t)

    h=ex/'qwen3_constants.h'
    t=h.read_text()
    for a,b in {
      'kQBodyRecords = 4':'kQBodyRecords = 12','kOBodyRecords = 2':'kOBodyRecords = 10','kUpGateReplays = 12':'kUpGateReplays = 68','kDownBodyRecords = 2':'kDownBodyRecords = 10','kQChunksPerRecord = 4':'kQChunksPerRecord = 20','kKvChunksPerRecord = 4':'kKvChunksPerRecord = 20','kOChunksPerRecord = 8':'kOChunksPerRecord = 24','kUpGateChunksPerReplay = 4':'kUpGateChunksPerReplay = 20','kDownChunksPerRecord = 12':'kDownChunksPerRecord = 68','kKWeightChunkBase = 16':'kKWeightChunkBase = 240','kVWeightChunkBase = 24':'kVWeightChunkBase = 280','kFullLayerOWeightChunkBase = 32':'kFullLayerOWeightChunkBase = 320','kFullLayerUpGateWeightChunkBase = 48':'kFullLayerUpGateWeightChunkBase = 560','kFullLayerDownWeightChunkBase = 96':'kFullLayerDownWeightChunkBase = 1920'}.items(): t=t.replace(a,b)
    h.write_text(t)
    print('patched historical full-layer generator + attention/cache fabric for Qwen3.8-27B')

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: qwen38_27b_torch2aie_adapter.py TORCH2AIE_ROOT')
    patch(Path(sys.argv[1]))
