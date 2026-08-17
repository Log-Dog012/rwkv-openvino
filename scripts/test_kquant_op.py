#!/usr/bin/env python3
"""验证 rwkv_kquant.dll 自定义 op：加载、创建、推理正确性（对比 gguf ground truth）。

1. core.add_extension(dll) 加载
2. 从 blk.0.time_mix_key.weight repack 出 codes/scales/zp
3. 用自定义 op 建图（探 ops.extension 创建方式）
4. infer 对比 gguf.dequantize ground truth matmul
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gguf_kquant_repack import repack_tensor

import gguf
import openvino as ov
from openvino import opset13 as ops

GGUF = "C:/Users/Mcsof/models/gguf/rwkv7-g1i-13.3b-Q4_K_M.gguf"
DLL = "scripts/kquant_op/build/rwkv_kquant.dll"

core = ov.Core()
print("[t] add_extension ...", flush=True)
core.add_extension(DLL)
print("[t] extension loaded OK", flush=True)

# 取 blk.0.time_mix_key.weight 的 repack 数据
r = gguf.GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}
t = T["blk.0.time_mix_key.weight"]
rep = dict(repack_tensor(t))
rep["shape"] = list(rep["shape"])[::-1]  # physical
O, K = int(rep["shape"][0]), int(rep["shape"][1])
codes = rep["codes"]          # [N] u8
scales = rep["scales"].astype(np.float32)  # [N/32]
zp = rep["zp"].astype(np.float32)          # [N/32]
N = codes.shape[0]
G = scales.shape[0]
print(f"[t] weight [{O}x{K}] type={rep['type']} codes={N} scales={G}", flush=True)

# ground truth: x @ dequant(W)
x = np.random.randn(1, K).astype(np.float16)
nb = N // 256
raw = np.frombuffer(t.data, dtype=np.uint8)
gt_w = gguf.quants.Q4_K.dequantize_blocks(raw.reshape(-1, 144)[:nb].copy()).reshape(O, K).astype(np.float32)
gt_y = (x.astype(np.float32) @ gt_w.T).astype(np.float32)  # [1, O]

# 建图：自定义 op
x_ph = ops.parameter(ov.Shape([1, K]), ov.Type.f16, name="x")
codes_ph = ops.parameter(ov.Shape([N]), ov.Type.u8, name="codes")
scales_ph = ops.parameter(ov.Shape([G]), ov.Type.f32, name="scales")
zp_ph = ops.parameter(ov.Shape([G]), ov.Type.f32, name="zp")

print("[t] trying ops.extension.RwkvKQuantMatMul ...", flush=True)
try:
    node = ops.extension.RwkvKQuantMatMul(x_ph, codes_ph, scales_ph, zp_ph)
    print("[t] ops.extension create OK", flush=True)
except Exception as e:
    print(f"[t] ops.extension failed: {e}", flush=True)
    # fallback: opset13 extension via get_extension opset
    try:
        from openvino import opset_extension
        node = opset_extension.RwkvKQuantMatMul(x_ph, codes_ph, scales_ph, zp_ph)
        print("[t] opset_extension create OK", flush=True)
    except Exception as e2:
        print(f"[t] opset_extension failed: {e2}", flush=True)
        node = None

if node is None:
    print("[t] FAIL: cannot create custom op via Python API", flush=True)
    sys.exit(1)

res = ops.result(node, name="y")
model = ov.Model([res], [x_ph, codes_ph, scales_ph, zp_ph])
comp = core.compile_model(model, "CPU")
req = comp.create_infer_request()
print("[t] compiled, infer ...", flush=True)
t0 = time.time()
req.infer({x_ph: x, codes_ph: codes, scales_ph: scales, zp_ph: zp})
el = time.time() - t0
y = np.array(req.get_output_tensor(0).data)
err = float(np.max(np.abs(y - gt_y)))
print(f"[t] infer {el*1000:.1f} ms, max|err| vs ground truth = {err:.3e}", flush=True)
print(f"[t] {'PASS' if err < 1e-2 else 'FAIL'}", flush=True)
