#!/usr/bin/env python3
"""逐 op 数值二分器：L=1 小图全中间节点加 Result，CPU vs GPU(f16) 逐节点对比，
找第一个 GPU 跑偏的 op。内存占用小（L=1 图 ~百 MB 级）。
用法: python scripts/opwise_diff.py <gguf> [--layers 1] [--max-report 15]
"""
import argparse, sys, os
import numpy as np
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rwkv7_ov import build_rwkv7_ov

ap = argparse.ArgumentParser()
ap.add_argument("gguf")
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--device", default="GPU")
ap.add_argument("--max-report", type=int, default=15)
args = ap.parse_args()

model, (V, C, L, H, N) = build_rwkv7_ov(args.gguf, max_layers=args.layers)
# 拷贝图并给每个 op 的全部输出加 Result
from openvino import opset13 as _ops
model2 = model.clone()
outputs = []
for op in model2.get_ops():
    if op.get_type_name() in ("Parameter", "Constant", "Result"):
        continue
    for i in range(len(op.outputs())):
        try:
            outputs.append(op.output(i))
        except Exception:
            pass
model2.add_results([_ops.result(out) for out in outputs])
print(f"[diff] ops={len(model2.get_ops())} probe outputs={len(outputs)}", flush=True)

core = ov.Core()
res = {}
for tag, dev, props in (("CPU", "CPU", {}), ("DEV", args.device, {})):
    core.set_property(dev, props)
    comp = core.compile_model(model2, dev)
    ins = {i.any_name: i for i in comp.inputs}
    req = comp.create_infer_request()
    o = req.infer({ins["idx"]: np.array([15338], np.int64),
                   ins["s_att_x"]: np.zeros((L, C), np.float16),
                   ins["s_kv"]: np.zeros((L, H, N, N), np.float16),
                   ins["s_ffn"]: np.zeros((L, C), np.float16)})
    res[tag] = [np.asarray(o[i]) for i in range(len(outputs))]
    del comp, req

names = [out.get_node().friendly_name or out.get_node().name for out in outputs]
types = [out.get_node().get_type_name() for out in outputs]
diffs = []
for i, (a, b) in enumerate(zip(res["CPU"], res["DEV"])):
    a = a.astype(np.float32); b = b.astype(np.float32)
    if a.size == 0:
        continue
    d = np.abs(a - b)
    rel = d.max() / (np.abs(a).max() + 1e-9)
    diffs.append((d.max(), rel, i))

diffs.sort(key=lambda t: -t[0])
print(f"[diff] top {args.max_report} divergent nodes (maxabs, rel):")
for d, rel, i in diffs[:args.max_report]:
    print(f"  {d:12.6f} rel={rel:9.2e}  [{types[i]:14s}] {names[i]}")

first = None
for d, rel, i in reversed(diffs):
    if d > 0.05:
        first = (d, rel, i)
print(f"\n[diff] LAST bad node (图执行顺序上游最晚分叉点需结合拓扑): "
      f"{names[first[2]]} [{types[first[2]]}] maxabs={first[0]:.4f}" if first else "[diff] no node >0.05")

# 拓扑序找第一个分叉 >0.05 的节点
order = {op.get_name(): k for k, op in enumerate(model2.get_ordered_ops())}
bad = [(order.get(names[i], 1 << 30), d, i) for d, rel, i in diffs if d > 0.05]
if bad:
    bad.sort()
    _, d, i = bad[0]
    print(f"[diff] FIRST divergent in topo order: {names[i]} [{types[i]}] maxabs={d:.4f}")
else:
    print("[diff] all nodes agree (CPU vs GPU)")
