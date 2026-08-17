#!/usr/bin/env python3
"""速度对比：自定义 AVX2 kernel op vs OV 融合 int4 路径（同一 4096x4096 Q4_K matmul）。

两条路径处理同一个权重 blk.0.time_mix_key.weight：
  A. 自定义 op（rwkv_kquant.dll，AVX2 解量化+matmul，reference evaluate）
  B. 融合路径（build_compressed_weight 的 u4 Constant->Convert->Sub->Mul->MatMul，
     CPU 插件应融合成原生 int4 matmul）
各测多次取平均，输出对比。
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gguf_kquant_repack import repack_tensor
from gguf_to_ov_compressed import build_compressed_weight

import gguf
import openvino as ov
from openvino import opset13 as ops

GGUF = "C:/Users/Mcsof/models/gguf/rwkv7-g1i-13.3b-Q4_K_M.gguf"
DLL = "scripts/kquant_op/build/rwkv_kquant.dll"
XML = "scripts/kquant_op/test_kq.xml"

r = gguf.GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}
t = T["blk.0.time_mix_key.weight"]
rep = dict(repack_tensor(t))
rep["shape"] = list(rep["shape"])[::-1]
O, K = int(rep["shape"][0]), int(rep["shape"][1])
codes = rep["codes"]
scales = rep["scales"].astype(np.float32)
zp = rep["zp"].astype(np.float32)
N, G = codes.shape[0], scales.shape[0]

x = np.random.randn(1, K).astype(np.float16)
N_REP = 20

core = ov.Core()
core.add_extension(DLL)
core.set_property("CPU", {"SNIPPETS_MODE": "DISABLE"})

# ---------- A. 自定义 op ----------
model = core.read_model(XML)
comp_a = core.compile_model(model, "CPU")
req_a = comp_a.create_infer_request()
ins_a = {i.any_name: i for i in comp_a.inputs}
# warmup
req_a.infer({ins_a["x"]: x, ins_a["codes"]: codes, ins_a["scales"]: scales, ins_a["zp"]: zp})
t0 = time.time()
for _ in range(N_REP):
    req_a.infer({ins_a["x"]: x, ins_a["codes"]: codes, ins_a["scales"]: scales, ins_a["zp"]: zp})
dt_a = (time.time() - t0) / N_REP
print(f"[A] custom op   : {dt_a*1000:.2f} ms/matmul ({N_REP} reps)", flush=True)

# ---------- B. 融合路径 ----------
core2 = ov.Core()  # 不设 SNIPPETS_MODE，让融合路径走原生 int4
x_ph = ops.parameter(ov.Shape([1, K]), ov.Type.f16, name="x")
deq = build_compressed_weight(rep, "w_key")
mm = ops.matmul(x_ph, deq, transpose_a=False, transpose_b=True, name="mm")
res = ops.result(mm, name="y")
model_b = ov.Model([res], [x_ph])
comp_b = core2.compile_model(model_b, "CPU")
req_b = comp_b.create_infer_request()
ins_b = {i.any_name: i for i in comp_b.inputs}
req_b.infer({ins_b["x"]: x})
t0 = time.time()
for _ in range(N_REP):
    req_b.infer({ins_b["x"]: x})
dt_b = (time.time() - t0) / N_REP
print(f"[B] fused int4  : {dt_b*1000:.2f} ms/matmul ({N_REP} reps)", flush=True)

# ---------- 对比 ----------
print(f"\n=== custom/fused = {dt_a/dt_b:.2f}x ===", flush=True)
print(f"=== speedup vs fused = {dt_b/dt_a:.2f}x ===", flush=True)
