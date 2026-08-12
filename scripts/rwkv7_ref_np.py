#!/usr/bin/env python3
"""RWKV7 单层前向的 numpy 参考实现（用于 OV 图数值验证，省去 torch 3GB 常驻）。

权重方向严格对齐 rwkv7_torch.py + gguf_to_ov.build_state:
  * build_state 已对 matmul 权重做 orient="T"（转置）——即 W_torch = GGUF_raw.T。
    OV 图吃 GGUF 原生布局 + transpose_b=True == x@GGUF_raw.T == x@W_torch，与 torch 逐元素一致。
  * 注意：build_state 的 emb.weight 是 RAW（ln0 仅由 RWKV7._build_from_raw 在类内应用，
    build_state 不应用）。本模块显式把 ln0 应用到 emb（对齐 OV 的 emb_const 预归一），否则
    与 OV 比对会整体差一个 ln0 缩放（之前验证 FAIL 的根因之一）。
  * 仅抽取第 0 层权重（~50MB），避免反量化整模型（1.5B fp16 ≈ 3GB）。
"""
import numpy as np
import torch
from gguf_to_ov import build_state
import gguf


def layer0_weights(gguf_path):
    """返回第 0 层权重(dict, np.float32) + (C, H, N)。"""
    r = gguf.GGUFReader(gguf_path)
    z, n_head, head_size = build_state(r, torch.float16)
    del r
    C = int(z["emb.weight"].shape[1])
    p = "blocks.0."
    ga = lambda k: z[p + "att." + k].numpy().astype(np.float32)   # att 子块
    g = lambda k: z[p + k].numpy().astype(np.float32)             # ln1/ln2 等

    # build_state 的 emb 是 RAW，需显式加 ln0（对齐 OV emb_const 预归一 + torch RWKV7._build_from_raw）
    ln0_w = z["blocks.0.ln0.weight"].numpy().astype(np.float32)
    ln0_b = z["blocks.0.ln0.bias"].numpy().astype(np.float32)
    emb_raw = z["emb.weight"].numpy().astype(np.float32)
    mu = emb_raw.mean(-1, keepdims=True)
    va = emb_raw.var(-1, keepdims=True)
    emb_ln = (emb_raw - mu) / np.sqrt(va + 1e-5) * ln0_w + ln0_b   # [V,C]

    w = {
        "emb": emb_ln,   # 已含 ln0 预归一（对齐 OV）
        "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"),
        "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
        "x_r": ga("x_r"), "x_w": ga("x_w"), "x_k": ga("x_k"),
        "x_v": ga("x_v"), "x_a": ga("x_a"), "x_g": ga("x_g"),
        "r": ga("receptance.weight"), "w1": ga("w1"), "w2": ga("w2"),
        "key": ga("key.weight"), "val": ga("value.weight"),
        "a0": ga("a0"), "a1": ga("a1"), "a2": ga("a2"),
        "g1": ga("g1"), "g2": ga("g2"),
        "k_k": ga("k_k"), "k_a": ga("k_a"),
        "v0": ga("v0"), "v1": ga("v1"), "v2": ga("v2"),
        "w0": ga("w0"),
        "lnx_w": ga("ln_x.weight"), "lnx_b": ga("ln_x.bias"),
        "r_k": ga("r_k"),
        "out": ga("output.weight"),
        "ff_xk": g("ffn.x_k"), "ff_key": g("ffn.key.weight"), "ff_val": g("ffn.value.weight"),
    }
    del z
    return w, C, n_head, head_size


def _l2norm(x, axis):
    return x / np.sqrt(np.sum(x * x, axis=axis, keepdims=True))


def block_forward(x, s_att_x, s_kv, s_ffn, w, H, N):
    """单层前向。x:(C,) 已是 emb+ln0；返回 (xa_in:(C,), st:(H,N,N), xf_in:(C,))。"""
    C = x.shape[0]
    xa_in = (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) * w["ln1_w"] + w["ln1_b"]
    d = s_att_x - xa_in
    xr = xa_in + d * w["x_r"]
    xw = xa_in + d * w["x_w"]
    xk = xa_in + d * w["x_k"]
    xv = xa_in + d * w["x_v"]
    xa = xa_in + d * w["x_a"]
    xg = xa_in + d * w["x_g"]

    r = xr @ w["r"]
    wl = np.tanh(xw @ w["w1"]) @ w["w2"]
    k = xk @ w["key"]
    v = xv @ w["val"]
    a = 1.0 / (1.0 + np.exp(-(w["a0"] + (xa @ w["a1"]) @ w["a2"])))
    g = (1.0 / (1.0 + np.exp(-(xg @ w["g1"])))) @ w["g2"]

    kk = _l2norm((k * w["k_k"]).reshape(H, N), axis=1).reshape(H * N)
    k = k * (1.0 + (a - 1.0) * w["k_a"])
    # v_first / vfix 仅在 i>0 用；验证只跑 layer0 的 i=0，v_first=v，无 vfix
    decay = np.exp(-0.606531 * (1.0 / (1.0 + np.exp(-(w["w0"] + wl)))))

    st = s_kv.copy()
    vk = v.reshape(H, N, 1) @ k.reshape(H, 1, N)
    ab = (-kk).reshape(H, N, 1) @ (kk * a).reshape(H, 1, N)
    st = st * decay.reshape(H, 1, N) + st @ ab + vk

    _x = (st @ r.reshape(H, N, 1)).reshape(1, H, N).astype(np.float32)
    _mu = _x.mean(-1, keepdims=True)
    _var = ((_x - _mu) ** 2).mean(-1, keepdims=True)
    _y = (_x - _mu) / np.sqrt(_var + 64e-5)
    o = _y.reshape(1, H * N).astype(np.float32) * w["lnx_w"] + w["lnx_b"]
    gate = ((r * k * w["r_k"]).reshape(H, N).sum(-1, keepdims=True) * v.reshape(H, N)).reshape(H * N)
    o = o + gate

    x = x + (o * g) @ w["out"]
    xf_in = (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) * w["ln2_w"] + w["ln2_b"]
    return xa_in.astype(np.float32), st.astype(np.float32), xf_in.astype(np.float32)
