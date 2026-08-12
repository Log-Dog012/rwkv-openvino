#!/usr/bin/env python3
"""全张量级对照：GGUF(build_state) vs 官方 pth(后处理) 的 z 字典，逐键比形状+corr。
内存安全：gz 用 float16（~3GB），pth 按 key 懒加载（mmap），不预建全量 dict，避免超 8GB。
目的：一次性暴露 GGUF 里所有坏张量（不止 v1/v2）。量化张量(Q4_K/Q6_K) corr≈0.99 属正常。"""
import os, sys
import numpy as np
import torch
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_to_ov import build_state
from rwkv7_torch import _T_KEYS

GGUF = "models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
PTH  = "models/rwkv7-g1i-1.5b.pth"

import gguf as _gg
reader = _gg.GGUFReader(GGUF)
gz, n_head, head_size = build_state(reader, torch.float16)
print(f"[verify] GGUF z: {len(gz)} tensors, C={gz['emb.weight'].shape[1]}, H={n_head}", flush=True)

raw = torch.load(PTH, map_location="cpu", mmap=True)

def ref_of(k):
    v = raw[k]
    t = v.to(torch.float32)
    if any(s in k for s in _T_KEYS):
        t = t.t()
    t = t.squeeze()
    if k.endswith("att.r_k"):
        t = t.flatten()
    return t.contiguous()

def corr(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return float('nan')
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

bad_shape, bad_corr = [], []
print(f"{'key':38s} {'GGUF':16s} {'ref':16s} {'corr':>8s}", flush=True)
for k in sorted(gz.keys()):
    ga = gz[k].numpy()
    try:
        ra = ref_of(k).numpy()
    except KeyError:
        print(f"{k:38s} {str(tuple(ga.shape)):16s} {'<absent>':16s}", flush=True)
        bad_shape.append(k); continue
    c = corr(ga, ra)
    gs, rs = str(tuple(ga.shape)), str(tuple(ra.shape))
    flag = ""
    if gs != rs:
        flag = " SHAPE"; bad_shape.append(k)
    elif c < 0.95:
        flag = " CORR"; bad_corr.append(k)
    print(f"{k:38s} {gs:16s} {rs:16s} {c:8.4f}{flag}", flush=True)

print(f"\n[verify] bad_shape={len(bad_shape)}: {bad_shape}", flush=True)
print(f"[verify] bad_corr(<0.95, 排除量化≈0.99): {len(bad_corr)}: {bad_corr}", flush=True)
print("[verify] DONE", flush=True)
