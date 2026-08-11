"""
把纯 torch 的 RWKV7 单步递推导出为 OpenVINO IR（FP32 / FP16），并在 CPU 上验证：
  * 与 torch 参考逐 token logits 对比（max abs diff）；
  * 贪心生成文本对比；
  * 编译后单 step 延迟与吞吐量。

用法:
  python3 export_ov.py --fp32          # 导出 fp32 IR + 编译测试
  python3 export_ov.py --fp16          # 导出 fp16 IR + 编译测试（默认）
  python3 export_ov.py --both          # 两个都来
"""
import os, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("RWKV_JIT_ON", "1"); os.environ["RWKV_CUDA_ON"] = "0"

import torch, openvino as ov
from rwkv_tokenizer import TRIE_TOKENIZER
from rwkv7_torch import RWKV7

MODEL = os.path.join(ROOT, "models", "rwkv7-g1d-0.1b")
OUT = os.path.join(ROOT, "out")
PROMPT = "The Eiffel Tower is located in the city of"


def build_example(m: RWKV7):
    idx = torch.zeros((1,), dtype=torch.int64)   # (1,) 一维，避免 0-d 被 OV 编译成动态 shape
    sa, sk, sf = m.zero_state()
    return (idx, sa, sk, sf), {"idx": idx, "s_att_x": sa, "s_kv": sk, "s_ffn": sf}


def export_one(m: RWKV7, fp16: bool):
    tag = "fp16" if fp16 else "fp32"
    xml = os.path.join(OUT, f"rwkv7_g1d_0.1b_step_{tag}.xml")
    print(f"\n=== convert_model -> {tag} ===")
    args, ex = build_example(m)
    t0 = time.time()
    ov_model = ov.convert_model(m, example_input=ex)   # trace 单步图
    print(f"  traced in {time.time()-t0:.1f}s ; nodes={len(ov_model.get_ops())}")
    ov.save_model(ov_model, xml, compress_to_fp16=fp16)
    size = os.path.getsize(xml) + os.path.getsize(xml.replace(".xml", ".bin"))
    print(f"  saved {xml}  ({size/1e6:.1f} MB, compress_to_fp16={fp16})")
    return ov_model, xml


def verify(ov_model, m: RWKV7, tok, n=20):
    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    req = compiled.create_infer_request()
    # 自动生成的输入名（按 forward 参数顺序）
    inames = [i.any_name for i in compiled.inputs]

    ids = tok.encode(PROMPT)
    s_att_x, s_kv, s_ffn = [x.numpy() for x in m.zero_state()]
    ref_sa, ref_sk, ref_sf = m.zero_state()
    L, H, N, C = m.n_layer, m.n_head, m.head_size, m.n_embd
    maxd = 0.0; nsteps = 0
    t0 = time.time()
    with torch.no_grad():
        # prefill
        for t in ids:
            out = req.infer({inames[0]: np.array([t], dtype=np.int64), inames[1]: s_att_x, inames[2]: s_kv, inames[3]: s_ffn})
            rlog, ref_sa, ref_sk, ref_sf = m(torch.tensor([t]), ref_sa, ref_sk, ref_sf)
            maxd = max(maxd, float((torch.from_numpy(out[0]).float() - rlog.float()).abs().max()))
            s_att_x = np.asarray(out[1]).reshape(L, C)
            s_kv = np.asarray(out[2]).reshape(L, H, N, N)
            s_ffn = np.asarray(out[3]).reshape(L, C)
            nsteps += 1
        # generate
        gen_ids = []
        for _ in range(n):
            nxt = int(torch.argmax(torch.from_numpy(out[0])))
            out = req.infer({inames[0]: np.array([nxt], dtype=np.int64),
                             inames[1]: s_att_x, inames[2]: s_kv, inames[3]: s_ffn})
            s_att_x = np.asarray(out[1]).reshape(L, C)
            s_kv = np.asarray(out[2]).reshape(L, H, N, N)
            s_ffn = np.asarray(out[3]).reshape(L, C)
            gen_ids.append(nxt); nsteps += 1
    dt = time.time() - t0
    print(f"\n[ov] {nsteps} steps in {dt:.2f}s -> {nsteps/dt:.1f} tok/s (prefill+gen)")
    print(f"[ov] max|logits diff vs torch| = {maxd:.3e}  -> {'OK' if maxd < 1.0 else 'CHECK'}")
    print(f"[ov] greedy out: {tok.decode(gen_ids)!r}")
    return maxd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="fp16")
    a = ap.parse_args(); modes = ["fp32", "fp16"] if a.mode == "both" else [a.mode]
    tok = TRIE_TOKENIZER(os.path.join(HERE, "rwkv_vocab_v20230424.txt"))
    m = RWKV7(MODEL + ".pth", dtype=torch.float32).eval()
    print(f"loaded torch ref L={m.n_layer} C={m.n_embd} H={m.n_head} N={m.head_size}")
    for mode in modes:
        ov_model, xml = export_one(m, fp16=(mode == "fp16"))
        verify(ov_model, m, tok, n=20)
        print(f"  IR xml: {xml}")


if __name__ == "__main__":
    main()
