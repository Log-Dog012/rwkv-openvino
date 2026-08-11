"""
大模型权重量化演示：导出 FP32 IR -> NNCF INT8_SYM (W8A32) -> CPU 验证生成。
证明"压缩"在大模型上同样可行（0.1B 已证明 4x 压缩，这里在 0.4B 复现）。
用法: python3 quantize_big.py <model.pth> [--n 16]
"""
import os, sys, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("RWKV_JIT_ON", "1"); os.environ["RWKV_CUDA_ON"] = "0"
import torch, openvino as ov, nncf, numpy as np
from rwkv_tokenizer import TRIE_TOKENIZER
from rwkv7_torch import RWKV7

PROMPT = "The Eiffel Tower is located in the city of"


def run(comp, req, ins, m, n=16):
    tok = TRIE_TOKENIZER(os.path.join(HERE, "rwkv_vocab_v20230424.txt"))
    ids = tok.encode(PROMPT)
    L, H, N, C = m.n_layer, m.n_head, m.head_size, m.n_embd
    s_att_x, s_kv, s_ffn = [x.numpy() for x in m.zero_state()]
    gen = []
    for t in ids + [None] * n:
        if t is None:
            t = int(np.argmax(o[0]))
        o = req.infer({ins[0]: np.array([t], dtype=np.int64),
                        ins[1]: s_att_x, ins[2]: s_kv, ins[3]: s_ffn})
        s_att_x = np.asarray(o[1]).reshape(L, C)
        s_kv = np.asarray(o[2]).reshape(L, H, N, N)
        s_ffn = np.asarray(o[3]).reshape(L, C)
        if t is not None:
            gen.append(t)
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    m = RWKV7(a.model, dtype=torch.float32).eval()
    print(f"[qb] loaded L={m.n_layer} C={m.n_embd} H={m.n_head} N={m.n_head and m.head_size}")

    idx = torch.zeros((1,), dtype=torch.int64); sa, sk, sf = m.zero_state()
    print("[qb] exporting FP32 IR ...")
    ovm = ov.convert_model(m, example_input=(idx, sa, sk, sf))
    fp32xml = a.model.replace(".pth", "_step_fp32.xml")
    ov.save_model(ovm, fp32xml, compress_to_fp16=False)
    sz32 = os.path.getsize(fp32xml) + os.path.getsize(fp32xml.replace(".xml", ".bin"))
    print(f"[qb] FP32 IR: {sz32/1e6:.0f} MB  ({fp32xml})")

    print("[qb] NNCF INT8_SYM (W8A32) ...")
    t0 = time.time()
    qm = nncf.compress_weights(ovm, mode=nncf.CompressWeightsMode.INT8_SYM)
    int8xml = a.model.replace(".pth", "_step_int8_sym.xml")
    ov.save_model(qm, int8xml, compress_to_fp16=False)
    sz8 = os.path.getsize(int8xml) + os.path.getsize(int8xml.replace(".xml", ".bin"))
    print(f"[qb] INT8 IR: {sz8/1e6:.0f} MB  ({sz8/sz32:.2f}x of FP32)  quantized in {time.time()-t0:.1f}s")

    core = ov.Core(); comp = core.compile_model(int8xml, "CPU")
    req = comp.create_infer_request(); ins = [i.any_name for i in comp.inputs]
    t0 = time.time()
    out = run(comp, req, ins, m, a.n)
    dt = time.time() - t0
    print(f"[qb] INT8 gen ({len(out)} tok / {dt:.1f}s = {len(out)/dt:.1f} tok/s):\n  {out!r}")


if __name__ == "__main__":
    main()
