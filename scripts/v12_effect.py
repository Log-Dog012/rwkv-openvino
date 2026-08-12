#!/usr/bin/env python3
"""0.1b 受控实验：v1/v2 是否影响生成？
正常生成 -> 清零 att.v1/v2 缓冲 -> 再生成，比较。"""
import os, sys, torch
torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import RWKV7
from rwkv_tokenizer import TRIE_TOKENIZER

PTH="models/rwkv7-g1d-0.1b.pth"
PROMPT="The Eiffel Tower is located in the city of"
N=30
tok=TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
ids=tok.encode(PROMPT)

def greedy(model,n):
    with torch.no_grad():
        model(*[torch.tensor([0],dtype=torch.int64),*model.zero_state()])
        sa,sk,sf=model.zero_state(); gen,last=[],None
        for t in list(ids)+[None]*n:
            idx=torch.tensor([t if t is not None else int(torch.argmax(last))],dtype=torch.int64)
            lg,sa,sk,sf=model(idx,sa,sk,sf); last=lg
            if t is None: gen.append(int(torch.argmax(lg)))
    return gen

m=RWKV7(PTH,dtype=torch.float16).eval()
g0=greedy(m,N)
print(f"[v12] baseline   : {tok.decode(g0)}",flush=True)

# 清零 v1/v2 缓冲
L=m.n_layer
for i in range(L):
    for kk in ("v1","v2"):
        bn=m._map[f"blocks.{i}.att.{kk}"]
        old=getattr(m,bn)
        setattr(m,bn, torch.zeros_like(old))
print(f"[v12] zeroed v1/v2 across {L} layers",flush=True)
g1=greedy(m,N)
print(f"[v12] v1/v2=0    : {tok.decode(g1)}",flush=True)
eq=sum(1 for a,b in zip(g0,g1) if a==b)
print(f"[v12] token-match baseline vs v1/v2=0 : {eq}/{N}",flush=True)

# 再用随机噪声填充 v1/v2
torch.manual_seed(0)
for i in range(L):
    for kk in ("v1","v2"):
        bn=m._map[f"blocks.{i}.att.{kk}"]
        setattr(m,bn, torch.randn_like(getattr(m,bn)))
g2=greedy(m,N)
print(f"[v12] v1/v2=rand : {tok.decode(g2)}",flush=True)
eq2=sum(1 for a,b in zip(g0,g2) if a==b)
print(f"[v12] token-match baseline vs v1/v2=rand: {eq2}/{N}",flush=True)
print("[v12] DONE",flush=True)
