#!/usr/bin/env python3
"""决定性权重方向测试：确认 OV 图(matmul GGUF原生 + transpose_b=?) 是否等于
torch 前向(matmul build_state 量)。

build_state 对 key/value/receptance/output 及 LoRA 对做了 orient="T"(.T)。
torch 前向 k = xk @ w(key)，w(key)=build_state量。
若 build_state量 == GGUF_raw.T：
    torch = xk @ GGUF_raw.T = MatMul(xk, GGUF_raw, transpose_b=True)
    => OV 须用 transpose_b=True 才匹配。
若 build_state量 == GGUF_raw：
    torch = xk @ GGUF_raw = MatMul(xk, GGUF_raw, transpose_b=False)
    => OV 用 transpose_b=False 匹配。

本脚本实测两种关系，并直接比较 x@raw 与 x@bs（以及 x@raw.T）。
"""
import sys, os
import numpy as np
import torch
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_to_ov import build_state

PATH = sys.argv[1] if len(sys.argv) > 1 else "../models/rwkv7-g1i-1.5b-Q4_K_M.gguf"

r = gguf.GGUFReader(PATH)
tensors = {t.name: t for t in r.tensors}

def graw(name):
    t = tensors[name]
    if t.tensor_type.name in ("F32", "F16"):
        return np.ascontiguousarray(t.data).astype(np.float32)
    return np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type)).astype(np.float32)

z, H, N = build_state(r, torch.float16)
C = int(z["emb.weight"].shape[1])

# 待查张量：(GGUF名, build_state键, 是否方阵)
checks = [
    ("blk.0.time_mix_key.weight",   "blocks.0.att.key.weight",            True),
    ("blk.0.time_mix_value.weight", "blocks.0.att.value.weight",          True),
    ("blk.0.time_mix_receptance.weight", "blocks.0.att.receptance.weight", True),
    ("blk.0.time_mix_output.weight","blocks.0.att.output.weight",         True),
    ("blk.0.time_mix_w1.weight",    "blocks.0.att.w1",                    False),
    ("blk.0.time_mix_w2.weight",    "blocks.0.att.w2",                    False),
    ("blk.0.channel_mix_key.weight","blocks.0.ffn.key.weight",            False),
    ("blk.0.channel_mix_value.weight","blocks.0.ffn.value.weight",        False),
]

rng = np.random.default_rng(0)
x = rng.standard_normal(C).astype(np.float32)

print(f"C={C} H={H} N={N}")
print(f"{'tensor':28s} {'raw.shape':14s} {'bs.shape':14s} {'bs==raw.T?':10s} {'bs==raw?':10s} {'x@raw==x@bs?':12s} {'x@raw==x@raw.T?':14s}")
for gname, bkey, square in checks:
    raw = graw(gname)
    bs = z[bkey].numpy().astype(np.float32)
    # 形状可能不同(raw与bs互为转置)，只比较能广播的
    try:
        eq_T = bool(np.allclose(bs, raw.T, atol=1e-3))
    except ValueError:
        eq_T = False
    try:
        eq_raw = bool(np.allclose(bs, raw, atol=1e-3))
    except ValueError:
        eq_raw = False
    # matmul: x @ raw (transpose_b=False) vs x @ bs (torch convention)
    try:
        mm_raw = x @ raw
        mm_bs = x @ bs
        mm_rawT = x @ raw.T
        eq_xraw_xbs = np.allclose(mm_raw, mm_bs, atol=1e-2)
        eq_xraw_xrawT = np.allclose(mm_raw, mm_rawT, atol=1e-2)
    except ValueError as e:
        mm_raw = mm_bs = mm_rawT = None
        eq_xraw_xbs = eq_xraw_xrawT = False
    print(f"{gname[-22:]:28s} {str(raw.shape):14s} {str(bs.shape):14s} {str(eq_T):10s} {str(eq_raw):10s} {str(eq_xraw_xbs):12s} {str(eq_xraw_xrawT):14s}")
    print(f"    -> 结论: torch用 x@bs; 若 bs==raw.T 则 torch=x@raw.T (OV须transpose_b=True); 若 bs==raw 则 torch=x@raw (OV须transpose_b=False)")

# 总结：判断 OV 应使用的 transpose_b
# 规则：对每一个 matmul，torch = x @ bs。OV 用 raw + transpose_b。
#   x@raw == x@bs  <=> OV transpose_b=False 正确
#   x@raw.T == x@bs <=> OV transpose_b=True 正确
print("\n=== 判定 OV transpose_b 取值 ===")
for gname, bkey, square in checks:
    raw = graw(gname); bs = z[bkey].numpy().astype(np.float32)
    mm_raw = x @ raw; mm_bs = x @ bs; mm_rawT = x @ raw.T
    if np.allclose(mm_raw, mm_bs, atol=1e-2):
        verdict = "transpose_b=False"
    elif np.allclose(mm_rawT, mm_bs, atol=1e-2):
        verdict = "transpose_b=True"
    else:
        verdict = "NEITHER (需检查)"
    print(f"  {gname[-22:]:24s} -> {verdict}")
