#!/usr/bin/env python3
"""控制实验：定位 GGUF 生成乱码是否由 v1/v2 编码错位（装了 a1/a2 的值）导致。

两套证据：
  (A) 数值比对：GGUF time_mix_v1/v2 与官方 pth 的 v1/v2、a1/a2 分别比形状+corr。
  (B) 生成对照：用官方 pth 的 v1/v2 覆盖 GGUF 构建的 z 后跑贪心生成，看是否恢复连贯。
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import RWKV7
from rwkv_tokenizer import TRIE_TOKENIZER

GGUF = "models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
PTH  = "models/rwkv7-g1i-1.5b.pth"

import gguf as _gguf_mod
reader = _gguf_mod.GGUFReader(GGUF)
tensors = {t.name: t for t in reader.tensors}

def deq(t):
    if t.tensor_type.name in ("F32", "F16"):
        return t.data.astype(np.float32)
    return _gguf_mod.dequantize(t.data, t.tensor_type).astype(np.float32)

# ---- (A) 数值比对（只看 blk.0，定性足够）----
raw = torch.load(PTH, map_location="cpu", mmap=True)
def flat(np_arr): return np_arr.reshape(-1)

for gg, pth_v, pth_a in [("time_mix_v1.weight", "v1", "a1"), ("time_mix_v2.weight", "v2", "a2")]:
    g = flat(deq(tensors["blk.0." + gg]))
    pv = flat(raw[f"blocks.0.att.{pth_v}"].to(torch.float32).numpy())
    pa = flat(raw[f"blocks.0.att.{pth_a}"].to(torch.float32).numpy())
    def corr(a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])
    gs = g.shape[0]; vs = pv.shape[0]; as_ = pa.shape[0]
    print(f"[probe] {gg}: GGUF_n={gs} | pth {pth_v}_n={vs} corr={corr(g, pv):.4f} | pth {pth_a}_n={as_} corr={corr(g, pa):.4f}", flush=True)

# ---- 用 gguf_to_ov.build_state 构建 z ----
from gguf_to_ov import build_state
z, n_head, head_size = build_state(reader, torch.float16)
C = z["emb.weight"].shape[1]
print(f"[probe] built z, C={C}, H={n_head}", flush=True)

tok = TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
prompt = "The Eiffel Tower is located in the city of"

def gen(model, n=24):
    with torch.no_grad():
        model(*[torch.tensor([0], dtype=torch.int64), *model.zero_state()])  # warmup
        sa, sk, sf = model.zero_state()
        ids = tok.encode(prompt)
        gen_ids, last = [], None
        for t in list(ids) + [None] * n:
            idx = torch.tensor([t if t is not None else int(torch.argmax(last))], dtype=torch.int64)
            lg, sa, sk, sf = model(idx, sa, sk, sf)
            last = lg
            if t is None:
                gen_ids.append(int(torch.argmax(lg)))
        return gen_ids

# (B1) 原始 GGUF z（v1/v2 按 GGUF 原值）
m_g = RWKV7.from_state(z, n_head, head_size, dtype=torch.float16).eval()
g_raw = gen(m_g)
print(f"[probe] GGUF-original gen: {tok.decode(g_raw)}", flush=True)
del m_g; import gc; gc.collect()

# (B2) 用官方 pth 的 v1/v2 覆盖
for i in range(24):
    v1 = raw[f"blocks.{i}.att.v1"].to(torch.float32).numpy()
    v2 = raw[f"blocks.{i}.att.v2"].to(torch.float32).numpy()
    z[f"blocks.{i}.att.v1"] = torch.from_numpy(np.ascontiguousarray(v1)).to(torch.float16)
    z[f"blocks.{i}.att.v2"] = torch.from_numpy(np.ascontiguousarray(v2)).to(torch.float16)
m_o = RWKV7.from_state(z, n_head, head_size, dtype=torch.float16).eval()
g_ov = gen(m_o)
print(f"[probe] pth-v1/v2-override gen: {tok.decode(g_ov)}", flush=True)
del m_o; gc.collect()
print("[probe] DONE", flush=True)
