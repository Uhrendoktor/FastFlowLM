#!/usr/bin/env python3
"""Adapt the recovered torch2aie full-layer generator to Qwen3.8-27B.

The historical generator is retained; only the dimension contract and kernel
constant header are replaced. This is intentionally a disposable checkout
adapter used by the canonical CI workflow, so the upstream toolchain checkout
remains pinned and untouched in the repository.
"""
from pathlib import Path
import re
import sys

H=5120
IM=17408
HD=256
NQ=24
NKV=4

CONTRACT = f'''"""Qwen3.8-27B full-layer contract."""
from __future__ import annotations
from dataclasses import dataclass

MAIN_COLUMNS=(2,3,4,5)
MAIN_ROWS=(2,3,4,5)
EDGE_COLUMNS=(0,1,6,7)
ROWS_PER_COLUMN=4
M_PER_TILE=32
OUTPUT_BLOCK_ROWS=512
HIDDEN_DIM={H}
INTERMEDIATE_DIM={IM}
HEAD_DIM={HD}
NUM_Q_HEADS={NQ}
NUM_KV_HEADS={NKV}
GQA_RATIO=NUM_Q_HEADS//NUM_KV_HEADS
K_CHUNK=256
GROUP_SIZE=32
ACT_SLICE_BF16=256
RECORD_DWORDS=17
RECORD_PAYLOAD_DWORDS=RECORD_DWORDS-1
PHASE_NAMES=("Q","K","V","O","UP","GATE","DOWN")
PHASE_INPUT_DIMS=(HIDDEN_DIM,HIDDEN_DIM,HIDDEN_DIM,NUM_Q_HEADS*HEAD_DIM,HIDDEN_DIM,HIDDEN_DIM,INTERMEDIATE_DIM)
PHASE_OUTPUT_DIMS=(NUM_Q_HEADS*HEAD_DIM,NUM_KV_HEADS*HEAD_DIM,NUM_KV_HEADS*HEAD_DIM,HIDDEN_DIM,INTERMEDIATE_DIM,INTERMEDIATE_DIM,HIDDEN_DIM)
PHASE_BLOCKS=tuple(x//OUTPUT_BLOCK_ROWS for x in PHASE_OUTPUT_DIMS)
PHASE_CHUNKS=tuple(x//K_CHUNK for x in PHASE_INPUT_DIMS)
NUM_PHASES=len(PHASE_NAMES)
TOTAL_LOGICAL_BLOCKS=sum(PHASE_BLOCKS)
PATCHES_PER_COLUMN=2
ROWS_PER_PATCH=2
CHUNK_BF16=2560
PATCH_BF16_BY_PHASE=tuple(ROWS_PER_PATCH*PHASE_CHUNKS[p]*CHUNK_BF16 for p in range(NUM_PHASES))
PHASE_PATCH_COUNTS=tuple(PHASE_BLOCKS[p]*len(MAIN_COLUMNS)*PATCHES_PER_COLUMN for p in range(NUM_PHASES))
PHASE_WEIGHT_BF16=tuple(PHASE_PATCH_COUNTS[p]*PATCH_BF16_BY_PHASE[p] for p in range(NUM_PHASES))
TOTAL_PATCHES=sum(PHASE_PATCH_COUNTS)
TOTAL_WEIGHT_BF16=sum(PHASE_WEIGHT_BF16)
TOTAL_WEIGHT_I32=TOTAL_WEIGHT_BF16//2
O_CHUNKS=NUM_Q_HEADS*HEAD_DIM//ACT_SLICE_BF16
SWIGLU_SLICES=INTERMEDIATE_DIM//OUTPUT_BLOCK_ROWS
C1R2_PACKET_DWORDS=1+HIDDEN_DIM//2
C1R2_QKV_REPLAYS=sum(PHASE_BLOCKS[:3])
C1R2_UPGATE_REPLAYS=PHASE_BLOCKS[4]+PHASE_BLOCKS[5]
C1R2_FINAL_REPLAYS=1
C6R2_INPUT_DWORDS=HIDDEN_DIM//2
C6R2_HALF_DWORDS=C6R2_INPUT_DWORDS//2
COMPACT_PACKET_DWORDS=1+len(MAIN_COLUMNS)*ROWS_PER_COLUMN*RECORD_PAYLOAD_DWORDS
ATTENTION_PACKET_DWORDS=NUM_Q_HEADS*HEAD_DIM//2
DOWN_PACKET_DWORDS=INTERMEDIATE_DIM//2
SHAPE_WINDOW_DWORDS=(NUM_Q_HEADS//4)*HEAD_DIM//2
SHAPE_CARRIER_DWORDS=(0x100+0x40)//4
RMS_NORM_DWORDS=HIDDEN_DIM
HIDDEN_DWORDS=HIDDEN_DIM//2
OUTPUT_DWORDS=HIDDEN_DIM//2
QK_ROPE_BF16=3*HEAD_DIM
QK_ROPE_DWORDS=QK_ROPE_BF16//2
AUX_DWORDS=RMS_NORM_DWORDS+QK_ROPE_DWORDS

@dataclass(frozen=True)
class PhaseSpec:
    name:str; input_dim:int; output_dim:int; blocks:int; chunks:int; patches:int
PHASE_SPECS=tuple(PhaseSpec(PHASE_NAMES[i],PHASE_INPUT_DIMS[i],PHASE_OUTPUT_DIMS[i],PHASE_BLOCKS[i],PHASE_CHUNKS[i],PHASE_PATCH_COUNTS[i]) for i in range(NUM_PHASES))
'''


def patch(root: Path) -> None:
    p=root/'examples/qwen3-decode-layer/contract.py'
    p.write_text(CONTRACT)
    h=root/'examples/qwen3-decode-layer/qwen3_constants.h'
    text=h.read_text()
    # Replace only dimension-dependent constants while retaining the historical
    # lock/packet IDs and ABI definitions.
    repl={
      'kQBodyRecords = 4':'kQBodyRecords = 12',
      'kKvBodyRecords = 2':'kKvBodyRecords = 2',
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

    # Hard guard against accidental 8B/9B reuse.
    assert 'HIDDEN_DIM=5120' in CONTRACT
    assert 'INTERMEDIATE_DIM=17408' in CONTRACT
    assert 'HEAD_DIM=256' in CONTRACT
    assert 'NUM_Q_HEADS=24' in CONTRACT
    assert 'NUM_KV_HEADS=4' in CONTRACT
    print('patched historical full-layer generator contract for Qwen3.8-27B')

if __name__=='__main__':
    patch(Path(sys.argv[1]))
