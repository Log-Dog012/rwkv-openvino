#!/usr/bin/env python3
"""OVMS REST (KServe v2) 推理验证客户端。

验证 OVMS 服务化 RWKV 单步 IR 的"加载 -> REST 推理 -> 递推状态 -> 生成"全链路。
- FP32 模型（如 rwkv32）：用标准 JSON 数值列表即可（datatype=FP32）。
- FP16 模型（如 rwkv）：JSON 协议无法表达 FP16 数值，需 binary/multipart REST
  （OVMS 特性），本脚本暂以 FP32 模型做端到端验证；FP16 在服务器侧已确认 AVAILABLE。

用法:
    python3 scripts/ovms_validate.py --url http://localhost:9001 --model rwkv32 \
        --pth models/rwkv7-g1d-0.1b.pth --prompt "The Eiffel Tower is located in the city of" --n 24
"""
import argparse, json, sys, urllib.request
import numpy as np

sys.path.insert(0, "scripts")
from rwkv_tokenizer import TRIE_TOKENIZER

REST = "{url}/v2/models/{model}/infer"


def main():
    A = argparse.ArgumentParser()
    A.add_argument("--url", default="http://localhost:9001")
    A.add_argument("--model", default="rwkv32")
    A.add_argument("--pth", default="models/rwkv7-g1d-0.1b.pth")
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=24)
    A = A.parse_args()

    import torch
    z = torch.load(A.pth, map_location="cpu")
    C = int(z["blocks.0.ln1.weight"].shape[0])
    H, N = tuple(z["blocks.0.att.r_k"].shape)
    L = 1 + max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks."))
    print(f"[ovms] L={L} H={H} N={N} C={C}")

    tok = TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
    ids = tok.encode(A.prompt)
    sa = np.zeros((L, C), np.float32)
    sk = np.zeros((L, H, N, N), np.float32)
    sf = np.zeros((L, C), np.float32)
    gen = []

    def infer(t):
        body = {"inputs": [
            {"name": "idx", "shape": [1], "datatype": "INT64", "data": [int(t)]},
            {"name": "s_att_x", "shape": [L, C], "datatype": "FP32", "data": sa.flatten().tolist()},
            {"name": "s_kv", "shape": [L, H, N, N], "datatype": "FP32", "data": sk.flatten().tolist()},
            {"name": "s_ffn", "shape": [L, C], "datatype": "FP32", "data": sf.flatten().tolist()},
        ]}
        req = urllib.request.Request(REST.format(url=A.url, model=A.model),
                                    data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)["outputs"]
        except urllib.error.HTTPError as e:
            print("[ovms] HTTP", e.code, "BODY:", e.read().decode()[:400])
            raise
        om = {o["name"]: o for o in out}
        logits = np.array(om["out_0"]["data"], dtype=np.float32).reshape(1, -1)
        nsa = np.array(om["out_1"]["data"], dtype=np.float32).reshape(L, C)
        nsk = np.array(om["out_2"]["data"], dtype=np.float32).reshape(L, H, N, N)
        nsf = np.array(om["out_3"]["data"], dtype=np.float32).reshape(L, C)
        return logits, nsa, nsk, nsf

    last_logits = None
    for t in list(ids) + [None] * A.n:
        logits, sa, sk, sf = infer(t if t is not None else int(np.argmax(last_logits)))
        last_logits = logits
        if t is None:
            t = int(np.argmax(logits))
            gen.append(t)
    print("[ovms] generated:", tok.decode(gen))


if __name__ == "__main__":
    main()
