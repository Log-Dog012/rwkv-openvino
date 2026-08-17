#!/usr/bin/env python3
"""验证 rwkv_kquant.dll：通过 XML read_model 加载自定义 op + 推理正确性（对比 gguf ground truth）。

Python 侧无 opset.extension/create API，自定义 op 节点走官方路径：XML IR 里
<op type="RwkvKQuantMatMul" version="rwkv">，read_model 时 add_extension 的 dll 解析构造。
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gguf_kquant_repack import repack_tensor

import gguf
import openvino as ov

GGUF = "C:/Users/Mcsof/models/gguf/rwkv7-g1i-13.3b-Q4_K_M.gguf"
DLL = "scripts/kquant_op/build/rwkv_kquant.dll"
XML = "scripts/kquant_op/test_kq.xml"

core = ov.Core()
core.add_extension(DLL)
print("[t] extension loaded OK", flush=True)

# repack 数据
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
print(f"[t] weight [{O}x{K}] codes={N} scales={G}", flush=True)

# ground truth
x = np.random.randn(1, K).astype(np.float16)
nb = N // 256
raw = np.frombuffer(t.data, dtype=np.uint8)
gt_w = gguf.quants.Q4_K.dequantize_blocks(raw.reshape(-1, 144)[:nb].copy()).reshape(O, K).astype(np.float32)
gt_y = (x.astype(np.float32) @ gt_w.T).astype(np.float32)

# 最小 XML（含自定义 op）
xml = f'''<net name="kq" version="11">
<layers>
<layer id="0" name="x" type="Parameter" version="opset1">
  <data element_type="f16" shape="1,{K}"/>
  <output><port id="0" precision="FP16" names="x"/></output>
</layer>
<layer id="1" name="codes" type="Parameter" version="opset1">
  <data element_type="u8" shape="{N}"/>
  <output><port id="0" precision="U8" names="codes"/></output>
</layer>
<layer id="2" name="scales" type="Parameter" version="opset1">
  <data element_type="f32" shape="{G}"/>
  <output><port id="0" precision="FP32" names="scales"/></output>
</layer>
<layer id="3" name="zp" type="Parameter" version="opset1">
  <data element_type="f32" shape="{G}"/>
  <output><port id="0" precision="FP32" names="zp"/></output>
</layer>
<layer id="4" name="kq" type="RwkvKQuantMatMul" version="rwkv">
  <input>
    <port id="0" precision="FP16"/><port id="1" precision="U8"/>
    <port id="2" precision="FP32"/><port id="3" precision="FP32"/>
  </input>
  <output><port id="0" precision="FP32" names="y"/></output>
</layer>
<layer id="5" name="y" type="Result" version="opset1">
  <input><port id="0" precision="FP32"/></input>
</layer>
</layers>
<edges>
<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
<edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
<edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
<edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
<edge from-layer="4" from-port="0" to-layer="5" to-port="0"/>
</edges>
</net>'''
open(XML, "w").write(xml)

print("[t] read_model ...", flush=True)
try:
    model = core.read_model(XML)
    print("[t] read_model OK, inputs:", [i.any_name for i in model.inputs], flush=True)
except Exception as e:
    print(f"[t] read_model FAILED: {e}", flush=True)
    sys.exit(1)

# 禁用 CPU snippets：自定义 op 不能被包进 SnippetsOpset::Subgraph JIT（其 evaluate 回退失败），
# 必须走 reference evaluate（我们的 AVX2 kernel）
try:
    core.set_property("CPU", {"SNIPPETS_MODE": "DISABLE"})
    print("[t] snippets disabled OK", flush=True)
except Exception as e:
    print(f"[t] warn: cannot disable snippets: {e}", flush=True)

comp = core.compile_model(model, "CPU")
req = comp.create_infer_request()
ins = {i.any_name: i for i in comp.inputs}
print("[t] compiled inputs:", list(ins.keys()), flush=True)
t0 = time.time()
res = req.infer({ins["x"]: x, ins["codes"]: codes, ins["scales"]: scales, ins["zp"]: zp})
el = time.time() - t0
outs = {o.any_name: o for o in comp.outputs}
print("[t] outputs:", list(outs.keys()), flush=True)
y = np.array(res[list(outs.keys())[0]])
err = float(np.max(np.abs(y - gt_y)))
print(f"[t] infer {el*1000:.1f} ms, max|err| vs ground truth = {err:.3e}", flush=True)
print(f"[t] {'PASS' if err < 1e-2 else 'FAIL'}", flush=True)
