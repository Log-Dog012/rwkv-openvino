#!/usr/bin/env python3
"""GPU int4 FC 累加精度探针：同一条 x @ W.T，三条路径对比 numpy f32 参考。

  A compressed_f16 : u4 -> f16 dequant -> MatMul(f16)   # 我们当前 IR 的形态
  B plain_f16      : f16 常量权重 -> MatMul(f16)        # 非 K-quant 对照
  C compressed_f32 : u4 -> f32 dequant -> MatMul(f32)  # 修复候选（权重仍 int4，仅计算抬 f32）

内存占用 ~百 MB。用法: python scripts/fc_precision_probe.py [--device GPU]
"""
import argparse
import numpy as np
import openvino as ov
from openvino import opset13 as ops

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="GPU")
ap.add_argument("--K", type=int, default=4096)
ap.add_argument("--N", type=int, default=4096)
args = ap.parse_args()

np.random.seed(0)
K, N = args.K, args.N
M = 1
W = (np.random.randn(N, K) * 0.02).astype(np.float32)
x = (np.random.randn(M, K) * 0.5).astype(np.float32)
ref_quant = x @ W.T

# 权重逐 32 元素对称量化成 u4（与 Q4_K 子块同构，无 zp 简化）
sc = np.abs(W).max(axis=1, keepdims=True) / 15 + 1e-12          # [N,1]
q = np.clip(np.round(W / (sc @ np.ones((1, K)))), -8, 7)        # [-8,7]
u4 = (q + 8).astype(np.uint8).reshape(-1)
sc32 = np.repeat(sc, K, axis=1).astype(np.float32)              # [N,K]
sc16 = np.repeat(sc, K // 32, axis=1).astype(np.float16).astype(np.float32)  # [N,K/32] 每 32 元素一组
W_f16approx = (q * np.repeat(sc16, 32, axis=1)).astype(np.float16).astype(np.float32)  # 各路径的共同"权重量化误差"下界
ref = x @ W_f16approx.T        # 量化后权重的 f32 精确参考（隔离累加误差）

x_node = ops.parameter([M, K], ov.Type.f32, "x")

def mm(inp, wt, in_f16, name):
    xi = ops.convert(x_node, ov.Type.f16, name=name+"/xc") if in_f16 else x_node
    return ops.matmul(xi, wt, False, True, name=name)

# A: compressed f16
cA = ops.constant(np.ascontiguousarray(u4), ov.Type.u4, "codesA")
sA = ops.constant(np.ascontiguousarray(sc16.astype(np.float16).reshape(N, K//32, 1)), ov.Type.f16, "sA")
wA = ops.multiply(ops.reshape(ops.convert(cA, ov.Type.f16, "convA"), [N, K//32, 32], True, "rA"),
                  sA, name="deqA")                # f16 [N,K/32,32]
vA = ops.reshape(wA, [N, K], True, "rA2")
yA = mm(x_node, vA, True, "mmA")
# B: plain f16 常量
wB = ops.constant(np.ascontiguousarray(W_f16approx.astype(np.float16)), ov.Type.f16, "wB")
yB = mm(x_node, wB, True, "mmB")
# C: compressed 但 dequant 到 f32（权重常数仍是 u4）
sC = ops.constant(np.ascontiguousarray(sc16.astype(np.float32).reshape(N, K//32, 1)), ov.Type.f32, "sC")
wC = ops.multiply(ops.reshape(ops.convert(cA, ov.Type.f32, "convC"), [N, K//32, 32], True, "rC"),
                  sC, name="deqC")                # f32 [N,K/32,32]
vC = ops.reshape(wC, [N, K], True, "rC2")
yC = ops.matmul(x_node, vC, False, True, "mmC")   # f32 matmul

model = ov.Model([ops.result(yA, "yA"), ops.result(yB, "yB"), ops.result(yC, "yC")], [x_node])
core = ov.Core()
core.set_property(args.device, {})
comp = core.compile_model(model, args.device)
ins = {i.any_name: i for i in comp.inputs}
req = comp.create_infer_request()
o = req.infer({list(ins.values())[0]: x})

print(f"device={args.device} K={K} N={N}")
import os
os.makedirs("out/probe", exist_ok=True)
np.savez(f"out/probe/fc_{args.device.lower()}.npz", **{str(i): np.asarray(v) for i, (k, v) in enumerate(o.items())})
print(f"原始权重量化误差 |ref_quant-ref| = {np.abs(ref_quant - ref).max():.6f}")
for name, out in o.items():
    y = np.asarray(out).reshape(-1).astype(np.float32)
    print(f"{name}: maxabs err vs f32 ref = {np.abs(y - ref.reshape(-1)).max():.6f}  "
          f"rel = {np.abs(y - ref.reshape(-1)).max()/np.abs(ref).max():.2e}")
