#!/usr/bin/env python3
"""K-quant 解量化保真探针：按 gguf_to_ov_compressed 的真实图结构，在 GPU vs CPU 上
逐元素对比 dequant 输出。定位 GPU 压缩权重融合 kernel 对哪种格式算错。

  Q4_K: u4(0..15) -> f16 -> (x - zp_eff) * scale   per32, zp_eff = min/scale 码空间零点
  Q6_K: i8(0..63) -> f16 -> x * scale              per16, 无 zp
  各有 f16 / f32 计算两个变体（权重常数始终 u4/i8，不变大）。
用法: python scripts/dequant_fidelity_probe.py [--device GPU]
"""
import argparse, os
import numpy as np
import openvino as ov
from openvino import opset13 as ops

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="GPU")
args = ap.parse_args()
np.random.seed(1)

N, K = 256, 4096
nb = K // 256  # 256 元素大块/行

# ---- Q4_K 数据（per32 非对称）----
c4 = np.random.randint(0, 16, N * K, dtype=np.uint8)
sc4 = (np.random.rand(N, nb, 8, 1).astype(np.float32) * 0.01 + 0.001)
zp4 = (np.random.rand(N, nb, 8, 1).astype(np.float32) * 8)
zp_eff = np.divide(zp4, sc4, out=np.zeros_like(sc4), where=sc4 != 0)  # 码空间零点
ref4 = (c4.astype(np.float32).reshape(N, nb, 8, 32) - zp4) * sc4      # gguf 语义: scale*code - min
ref4 = ref4.reshape(N, K)
sc4_flat = np.repeat(sc4, 32, axis=3).reshape(N, K)
zp_flat = np.repeat(zp_eff, 32, axis=3).reshape(N, K)

# ---- Q6_K 数据（per16 对称, 码 0..63）----
c6 = np.random.randint(0, 64, N * K).astype(np.int8)
sc6 = (np.random.rand(N, nb, 16, 1).astype(np.float32) * 0.01 + 0.001)
ref6 = (c6.astype(np.float32).reshape(N, nb, 16, 16) * sc6).reshape(N, K)
sc6_flat = np.repeat(sc6, 16, axis=3).reshape(N, K)

outs = {}
for base, ctype, per in (("q4k", ov.Type.u4, 32), ("q6k", ov.Type.i8, 16)):
    codes = c4 if base == "q4k" else c6
    for dtn in ("f16", "f32"):
        dt = ov.Type.f16 if dtn == "f16" else ov.Type.f32
        npdt = np.float16 if dtn == "f16" else np.float32
        tag = f"{base}_{dtn}"
        c = ops.constant(np.ascontiguousarray(codes), ctype, f"{tag}/codes")
        conv = ops.convert(c, dt, f"{tag}/conv")
        conv = ops.reshape(conv, [N, K // per, per], True, f"{tag}/cr")
        sc_flat = sc4_flat if base == "q4k" else sc6_flat
        sc = ops.constant(np.ascontiguousarray((sc4 if base == "q4k" else sc6).reshape(N, K // per, 1).astype(npdt)), dt, f"{tag}/sc")
        if base == "q4k":
            zp = ops.constant(np.ascontiguousarray(zp4.reshape(N, K // per, 1).astype(npdt)), dt, f"{tag}/zp")
            conv = ops.subtract(conv, zp, name=f"{tag}/sub")
        deq = ops.multiply(conv, sc, name=f"{tag}/deq")
        outs[tag] = ops.reshape(deq, [N, K], True, f"{tag}/out")

model = ov.Model([ops.result(v, k) for k, v in outs.items()], [], name="deq_probe")
core = ov.Core()
core.set_property(args.device, {})
comp = core.compile_model(model, args.device)
o = comp.create_infer_request().infer({})

print(f"device={args.device}")
os.makedirs("out/probe", exist_ok=True)
saved = {}
for name, out in zip(outs.keys(), o.values()):
    y = np.asarray(out).astype(np.float32).reshape(N, K)
    base = "q4k" if name.startswith("q4k") else "q6k"
    ref = ref4 if base == "q4k" else ref6
    saved[name] = y
    print(f"{name}: maxabs err vs numpy = {np.abs(y - ref).max():.6f}")
np.savez(f"out/probe/deq_{args.device.lower()}.npz", **saved)
