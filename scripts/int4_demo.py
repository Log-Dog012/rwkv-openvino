#!/usr/bin/env python3
"""RWKV-7 -> OpenVINO IR 的 NNCF 权重量化演示（INT8 / INT4）。

关键修正（2026-08-12）：NNCF `compress_weights` 默认 `group_size=128`，
而 RWKV 大量权重矩阵通道维仅 32/64，导致整层被跳过、INT4 名存实亡。
对 INT4 必须显式传 `group_size<=最小通道维`（这里 32），才能真压到 4-bit。

用法:
  python3 scripts/int4_demo.py models/rwkv7-g1d-0.1b_step_fp32.xml --mode INT4_SYM --group-size 32
  python3 scripts/int4_demo.py models/rwkv7-g1d-0.1b_step_fp32.xml --mode INT8_SYM
产出: out/<name>_<mode>_gs<gs>.{xml,bin}  + 体积/首步diff/生成样本
"""
import argparse, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import openvino as ov
import nncf
from rwkv_tokenizer import TRIE_TOKENIZER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", help="FP32 IR (xml)")
    ap.add_argument("--mode", default="INT4_SYM", choices=["INT4_SYM", "INT4_ASYM", "INT8_SYM"])
    ap.add_argument("--group-size", type=int, default=32, help="INT4 必须 <= 最小权重通道维(此处 32)")
    ap.add_argument("--n", type=int, default=24, help="生成 token 数")
    ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pth", default=None, help="RWKV pth 路径，用于读取 L/H/N/C 维度（IR 的 state 为动态形状）")
    A = ap.parse_args()

    core = ov.Core()
    base = core.read_model(A.xml)
    p0 = os.path.join(tempfile.mkdtemp(), "fp32.xml")
    ov.save_model(base, p0, compress_to_fp16=False)
    fp32_mb = (os.path.getsize(p0) + os.path.getsize(p0[:-4] + ".bin")) / 1e6
    print(f"[int4] FP32 baseline: {fp32_mb:.1f} MB")

    mode = getattr(nncf.CompressWeightsMode, A.mode)
    kw = {} if not A.mode.startswith("INT4") else {"group_size": A.group_size}
    if A.mode.startswith("INT4"):
        print(f"[int4] compressing {A.mode} (group_size={A.group_size}) ...")
    cm = nncf.compress_weights(base, mode=mode, **kw)

    outp = A.out or (os.path.splitext(A.xml)[0] + f"_{A.mode.lower()}_gs{A.group_size}.xml")
    ov.save_model(cm, outp, compress_to_fp16=False)
    q_mb = (os.path.getsize(outp) + os.path.getsize(outp[:-4] + ".bin")) / 1e6
    print(f"[int4] {A.mode} gs={A.group_size}: {q_mb:.1f} MB  (压缩比 {fp32_mb/q_mb:.2f}x) -> {outp}")

    # 编译 + 贪心生成（复用 export_big 已验证的喂数据逻辑）
    comp = core.compile_model(outp, "CPU")
    req = comp.create_infer_request()
    ins = [i.any_name for i in comp.inputs]
    L = H = N = C = None
    if A.pth:
        from rwkv7_torch import RWKV7
        _m = RWKV7(A.pth)
        L, H, N, C = _m.n_layer, _m.n_head, _m.head_size, _m.n_embd
    else:
        for i in comp.inputs:
            ps, nm = i.get_partial_shape(), i.any_name
            if ps.is_static:
                s = tuple(ps.to_shape())
                if nm == "s_att_x":
                    L, C = s[0], s[1]
                elif nm == "s_kv":
                    L, H, N, _ = s
    assert None not in (L, H, N, C), "无法从 IR 推导维度，请用 --pth 指定 pth"
    tok = TRIE_TOKENIZER(os.path.join(os.path.dirname(__file__), "rwkv_vocab_v20230424.txt"))
    ids = tok.encode(A.prompt)
    sa = np.zeros((L, C), np.float32)
    sk = np.zeros((L, H, N, N), np.float32)
    sf = np.zeros((L, C), np.float32)
    g = []
    for t in ids + [None] * A.n:
        if t is None:
            t = int(np.argmax(o[0]))
        o = req.infer({ins[0]: np.array([t], np.int64), ins[1]: sa, ins[2]: sk, ins[3]: sf})
        sa = np.asarray(o[1]).reshape(L, C)
        sk = np.asarray(o[2]).reshape(L, H, N, N)
        sf = np.asarray(o[3]).reshape(L, C)
        if t is not None:
            g.append(t)
    print(f"[int4] gen: {tok.decode(g)}")


if __name__ == "__main__":
    main()
