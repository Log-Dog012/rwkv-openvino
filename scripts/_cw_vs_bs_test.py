#!/usr/bin/env python3
"""决定性: 对每个 matmul 权值, 比较 OV cw 真实值 与 torch build_state(bs) 真实值,
并选 OV 的 matmul 取向(transpose_b) 使 x@operand_OV == x@bs (torch).

cw 真实值: 用 OV 编译 build_compressed_weight 子图取出(bit-exact 路径已验证)。
bs 真实值: gguf.dequantize(t) 得 graw, bs = graw.T (orient 测试已证 bs==graw.T)。
  仅取 layer-0 张量(每层 ~50MB), 不加载整模型, 内存可控。
"""
import sys, os
import numpy as np
import gguf
import openvino as ov
from openvino import opset13 as ops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_kquant_repack import repack_tensor
from gguf_to_ov_compressed import build_compressed_weight

PATH = sys.argv[1] if len(sys.argv) > 1 else "../models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
core = ov.Core()

r = gguf.GGUFReader(PATH)
T = {t.name: t for t in r.tensors}
C = int(T["token_embd.weight"].shape[1])

def get_cw(name):
    rep = repack_tensor(T[name])
    od, idm = int(rep["shape"][0]), int(rep["shape"][1])
    node = build_compressed_weight(rep, f"w_{name}")
    m = ov.Model([ops.result(node, "o")], [])
    comp = core.compile_model(m, "CPU")
    req = comp.create_infer_request()
    req.infer()
    v = np.array(req.get_output_tensor(0).data).astype(np.float32).reshape(od, idm)
    return v

def graw(name):
    t = T[name]
    if t.tensor_type.name in ("F16", "F32"):
        return np.ascontiguousarray(t.data).astype(np.float32)
    return np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type)).astype(np.float32)

# (GGUF名, torch build_state 键)
items = [
    ("blk.0.time_mix_receptance.weight", "blocks.0.att.receptance.weight"),
    ("blk.0.time_mix_key.weight",        "blocks.0.att.key.weight"),
    ("blk.0.time_mix_value.weight",      "blocks.0.att.value.weight"),
    ("blk.0.time_mix_output.weight",     "blocks.0.att.output.weight"),
    ("blk.0.time_mix_w1.weight",         "blocks.0.att.w1"),
    ("blk.0.time_mix_w2.weight",         "blocks.0.att.w2"),
    ("blk.0.channel_mix_key.weight",     "blocks.0.ffn.key.weight"),
    ("blk.0.channel_mix_value.weight",   "blocks.0.ffn.value.weight"),
    ("blk.0.time_mix_g1.weight",         "blocks.0.att.g1"),
    ("blk.0.time_mix_a1.weight",         "blocks.0.att.a1"),
]

rng = np.random.default_rng(1)
print(f"C={C}")
print(f"{'tensor':34s} {'cw.shape':12s} {'bs.shape':12s} {'cw==bs?':9s} {'cw==bs.T?':10s} {'x@cw==x@bs?':12s} {'x@cw.T==x@bs?':14s}  -> OV transpose_b")
for gname, bkey in items:
    cw = get_cw(gname)
    gr = graw(gname)
    bs = gr.T  # orient 测试: bs == graw.T
    # 选取与 bs 同 shape 的 x
    xlen = bs.shape[0]  # torch: x @ bs, x length = bs 行数
    x = rng.standard_normal(xlen).astype(np.float32)
    try:
        xcw = x @ cw
        xbs = x @ bs
        xcwT = x @ cw.T
        eq_cw = np.allclose(cw, bs, atol=5e-2)
        eq_cwT = np.allclose(cw.T, bs, atol=5e-2)
        e_xcw = np.allclose(xcw, xbs, atol=5e-2)
        e_xcwT = np.allclose(xcwT, xbs, atol=5e-2)
    except ValueError as e:
        xcw = xbs = xcwT = None
        eq_cw = eq_cwT = e_xcw = e_xcwT = False
    if e_xcw:
        verdict = "False"
    elif e_xcwT:
        verdict = "True"
    else:
        verdict = "NEITHER"
    print(f"{gname:34s} {str(cw.shape):12s} {str(bs.shape):12s} {str(eq_cw):9s} {str(eq_cwT):10s} {str(e_xcw):12s} {str(e_xcwT):14s}  -> {verdict}")
