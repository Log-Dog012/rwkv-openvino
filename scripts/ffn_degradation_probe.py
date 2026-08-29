#!/usr/bin/env python3
"""整图 FFN matmul 退化探针（复现 FUSE_INVESTIGATION task 5 方法，参数化任意 GGUF）。

构 L 层整图 -> CPU 编译（perf counter 开启）-> 二次推理后取 profiling：
汇总 ffk_/ffv_（FFN 16384 宽 matmul）单 op 耗时，对照 13B 退化序列
（L=2: 4.9ms / L=4: 4.8ms / L=8: 20ms / L=16: 34ms avg, 204ms max）。

用法:
  python scripts/ffn_degradation_probe.py <model.gguf> --layers 8 [--device CPU] [--threads 8]
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import openvino as ov
from rwkv7_ov import build_rwkv7_ov


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--layers", type=int, required=True)
    A.add_argument("--device", default="CPU")
    A.add_argument("--threads", type=int, default=8)
    A.add_argument("--iters", type=int, default=3, help="推理次数（首暖后取后几次均值）")
    args = A.parse_args()

    t0 = time.time()
    model, (V, C, L, H, N) = build_rwkv7_ov(args.gguf, max_layers=args.layers)
    print(f"[probe] built L={L} C={C} in {time.time()-t0:.1f}s", flush=True)

    core = ov.Core()
    core.set_property(args.device, {ov.properties.inference_num_threads: args.threads,
                                    ov.properties.enable_profiling: True})
    tc = time.time()
    comp = core.compile_model(model, args.device)
    print(f"[probe] compiled in {time.time()-tc:.1f}s", flush=True)
    ins = {i.any_name: i for i in comp.inputs}
    req = comp.create_infer_request()
    req.infer({ins["idx"]: np.array([1], np.int64),
               ins["s_att_x"]: np.zeros((L, C), np.float16),
               ins["s_kv"]: np.zeros((L, H, N, N), np.float16),
               ins["s_ffn"]: np.zeros((L, C), np.float16)})  # warmup

    times = []
    for _ in range(args.iters):
        t1 = time.time()
        req.infer({ins["idx"]: np.array([1], np.int64),
                   ins["s_att_x"]: np.zeros((L, C), np.float16),
                   ins["s_kv"]: np.zeros((L, H, N, N), np.float16),
                   ins["s_ffn"]: np.zeros((L, C), np.float16)})
        times.append((time.time() - t1) * 1000)
    print(f"[probe] infer steady: {np.median(times):.1f} ms (all: {[f'{t:.1f}' for t in times]})", flush=True)

    prof = req.get_profiling_info()
    ffk = [p for p in prof if p.node_name.startswith("ffk_")]
    ffv = [p for p in prof if p.node_name.startswith("ffv_")]
    att = [p for p in prof if p.node_name.startswith(("r_", "k_", "v_", "o_"))
           or "time_mix_output" in p.node_name]
    ms = lambda xs: np.array([p.real_time.total_seconds() * 1000 for p in xs])
    for tag, group in (("ffk", ffk), ("ffv", ffv)):
        if not group:
            print(f"[probe] {tag}: none"); continue
        us = sorted(ms(group))
        print(f"[probe] {tag}: n={len(us)} avg={np.mean(us):.1f}ms "
              f"med={np.median(us):.1f}ms max={max(us):.1f}ms", flush=True)
    # 最慢 5 个 op，看退化落在哪
    top = sorted(prof, key=lambda p: -p.microseconds)[:5]
    for p in top:
        print(f"[probe] top: {p.node_name} [{p.node_type}] {p.real_time.total_seconds()*1000:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
