"""
大模型验证脚本（参数化模型路径）：
  加载 RWKV7 系列 .pth -> 导出 OpenVINO IR (FP32/FP16) -> CPU 编译
  -> 既做 torch 贪婪生成（参考）又做 OV 贪婪生成，对齐 token 序列，证明 CPU 端跑通且正确。
用法:
  python3 export_big.py <model.pth> [--n 16] [--mode fp32|fp16] [--load-fp16]
  --load-fp16 : 以 FP16 加载权重（13.3B 时把常驻显存/RAM 从 ~53GB 降到 ~26GB），mode 应配 fp16
"""
import os, sys, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("RWKV_JIT_ON", "1"); os.environ["RWKV_CUDA_ON"] = "0"
import torch, openvino as ov, numpy as np
from rwkv_tokenizer import TRIE_TOKENIZER
from rwkv7_torch import RWKV7

PROMPT = "The Eiffel Tower is located in the city of"


def greedy_torch(m, n, tok):
    ids = tok.encode(PROMPT)
    sa, sk, sf = m.zero_state()
    gen = []
    for t in ids + [None] * n:
        if t is None:
            with torch.no_grad():
                logits, sa, sk, sf = m(torch.tensor([gen[-1]]), sa, sk, sf)
                t = int(torch.argmax(logits[0]).item())
        else:
            with torch.no_grad():
                _, sa, sk, sf = m(torch.tensor([t]), sa, sk, sf)
        if t is not None:
            gen.append(t)
    return gen


def greedy_ov(comp, ins, m, n, tok):
    ids = tok.encode(PROMPT)
    L, H, N, C = m.n_layer, m.n_head, m.head_size, m.n_embd
    s_att_x, s_kv, s_ffn = [x.numpy() for x in m.zero_state()]
    gen = []
    for t in ids + [None] * n:
        if t is None:
            t = int(np.argmax(o[0]))
        o = comp.infer({ins[0]: np.array([t], dtype=np.int64),
                       ins[1]: s_att_x, ins[2]: s_kv, ins[3]: s_ffn})
        s_att_x = np.asarray(o[1]).reshape(L, C)
        s_kv = np.asarray(o[2]).reshape(L, H, N, N)
        s_ffn = np.asarray(o[3]).reshape(L, C)
        if t is not None:
            gen.append(t)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--mode", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--load-fp16", action="store_true",
                    help="以 FP16 加载权重（大模型省 RAM），需配合 --mode fp16")
    a = ap.parse_args()

    ldtype = torch.float16 if a.load_fp16 else torch.float32
    t0 = time.time()
    m = RWKV7(a.model, dtype=ldtype).eval()
    print(f"[big] loaded L={m.n_layer} C={m.n_embd} H={m.n_head} N={m.head_size} "
          f"dtype={ldtype} in {time.time()-t0:.1f}s")

    fp16 = (a.mode == "fp16")
    idx = torch.zeros((1,), dtype=torch.int64)
    sa, sk, sf = m.zero_state()
    t0 = time.time()
    ovm = ov.convert_model(m, example_input=(idx, sa, sk, sf))
    print(f"[big] traced IR ({a.mode}) in {time.time()-t0:.1f}s ; nodes={len(ovm.get_ops())}")
    outxml = a.model.replace(".pth", f"_step_{a.mode}.xml")
    ov.save_model(ovm, outxml, compress_to_fp16=fp16)
    tot = os.path.getsize(outxml) + os.path.getsize(outxml.replace('.xml', '.bin'))
    print(f"[big] saved {outxml} ({tot/1e6:.0f} MB)")

    tok = TRIE_TOKENIZER(os.path.join(HERE, "rwkv_vocab_v20230424.txt"))
    print(f"[big] torch greedy ({len(tok.encode(PROMPT))+a.n} steps) ...")
    t0 = time.time()
    gt = greedy_torch(m, a.n, tok)
    dt = time.time() - t0
    print(f"[big]   torch out: {tok.decode(gt)!r}  ({dt:.1f}s, {len(gt)/dt:.2f} tok/s)")

    core = ov.Core()
    comp = core.compile_model(outxml, "CPU")
    req = comp.create_infer_request()
    ins = [i.any_name for i in comp.inputs]
    t0 = time.time()
    gv = greedy_ov(comp, ins, m, a.n, tok)
    dt = time.time() - t0
    print(f"[big]   OV    out: {tok.decode(gv)!r}  ({dt:.1f}s, {len(gv)/dt:.2f} tok/s)")

    match = "EXACT MATCH" if gt == gv else \
        f"DIFF @ first {next((i for i,(x,y) in enumerate(zip(gt,gv)) if x!=y), -1)}"
    print(f"[big] TORCH vs OV: {match}  (torch_len={len(gt)} ov_len={len(gv)})")


if __name__ == "__main__":
    main()
