#!/usr/bin/env python3
"""内存安全终验：GGUF(修复方阵+校正v1/v2) 与 官方 pth 的【首步 logits】对齐。
只做单次前向（不跑生成循环），避免 1.5B fp16 生成顶爆 8GB cgroup。
若 max|Δ| 在量化容差内（~0.05~0.2），即证明映射+方阵转置修复+v1/v2校正整体正确。"""
import os, sys, time
import torch
torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import RWKV7
from gguf_to_ov import build_state, repair_v12
import gguf as _g

GGUF="models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
PTH="models/rwkv7-g1i-1.5b.pth"

def first_logits(model):
    with torch.no_grad():
        sa,sk,sf=model.zero_state()
        idx=torch.tensor([0],dtype=torch.int64)
        lg,_,_,_=model(idx,sa,sk,sf)
    return lg

# (1) GGUF 模型（修复方阵 + 校正 v1/v2）
reader=_g.GGUFReader(GGUF)
z,nh,hs=build_state(reader, torch.float16)
import gc; del reader; gc.collect()
n_rep=repair_v12(z, PTH)  # 就地覆盖 v1/v2，返回层数
print(f"[logits] repaired {n_rep} layers", flush=True)
print(f"[logits] GGUF built+repaired, generating first logits ...", flush=True)
t0=time.time()
m_g=RWKV7.from_state(z,nh,hs,dtype=torch.float16).eval()
lg_g=first_logits(m_g)
print(f"[logits] GGUF first-step done in {time.time()-t0:.1f}s, max={lg_g.max().item():.3f}", flush=True)
del m_g; gc.collect()

# (2) 官方 pth 模型
print(f"[logits] building pth model ...", flush=True)
t0=time.time()
m_p=RWKV7(PTH, dtype=torch.float16).eval()
lg_p=first_logits(m_p)
print(f"[logits] pth first-step done in {time.time()-t0:.1f}s, max={lg_p.max().item():.3f}", flush=True)
del m_p; gc.collect()

d=(lg_g.float()-lg_p.float()).abs()
print(f"[logits] max|Δ|(GGUF-fixed-repaired vs official pth): {d.max().item():.4f}", flush=True)
print(f"[logits] mean|Δ|: {d.mean().item():.4f}  argmax match: {int(lg_g.argmax())==int(lg_p.argmax())}", flush=True)
print("[logits] DONE", flush=True)
