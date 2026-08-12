#!/usr/bin/env python3
"""内存安全：GGUF 修复 v1/v2 后单模型贪心生成，验证是否连贯。"""
import os, sys, torch
torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import RWKV7
from rwkv_tokenizer import TRIE_TOKENIZER
from gguf_to_ov import build_state, repair_v12

GGUF="models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
PTH="models/rwkv7-g1i-1.5b.pth"
PROMPT="The Eiffel Tower is located in the city of"
N=16
tok=TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
ids=tok.encode(PROMPT)

import gguf as _g
reader=_g.GGUFReader(GGUF)
z,nh,hs=build_state(reader, torch.float16)
import gc; del reader; gc.collect()  # 释放 GGUFReader ~3GB 映射，避免与模型同驻超 8GB
print(f"[repair-gen] built z, repairing v1/v2 from pth ...", flush=True)
n=repair_v12(z, PTH)
print(f"[repair-gen] repaired {n} layers", flush=True)
m=RWKV7.from_state(z,nh,hs,dtype=torch.float16).eval()
del z; import gc; gc.collect()
print(f"[repair-gen] model built, generating ...", flush=True)

with torch.no_grad():
    m(*[torch.tensor([0],dtype=torch.int64),*m.zero_state()])
    sa,sk,sf=m.zero_state(); gen,last=[],None
    for t in list(ids)+[None]*N:
        idx=torch.tensor([t if t is not None else int(torch.argmax(last))],dtype=torch.int64)
        lg,sa,sk,sf=m(idx,sa,sk,sf); last=lg
        if t is None: gen.append(int(torch.argmax(lg)))
print(f"[repair-gen] GGUF-repaired gen: {tok.decode(gen)}", flush=True)
print("[repair-gen] DONE", flush=True)
