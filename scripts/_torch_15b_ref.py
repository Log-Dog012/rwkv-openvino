#!/usr/bin/env python3
"""1.5B GGUF 的 torch 基线生成（可信参考，用于与 OV chunked 执行器对比）。

从 GGUF 经 build_state(fp16) 注入 torch 复刻类 RWKV7，greedy 生成文本。
输出: 文本 + token id 序列 + 首个生成 token 的 top-10 logits。

用法: python3 _torch_15b_ref.py <model.gguf> [--prompt "..."] [--n 16]
"""
import os, sys, time, argparse, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["RWKV_CUDA_ON"] = "0"

import gguf
from gguf_to_ov import build_state
from rwkv7_torch import RWKV7
from rwkv_tokenizer import TRIE_TOKENIZER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default=None, help="保存结果 json 的路径")
    args = ap.parse_args()

    tok = TRIE_TOKENIZER(os.path.join(HERE, "rwkv_vocab_v20230424.txt"))
    ids = tok.encode(args.prompt)
    print(f"prompt({len(ids)} tok): {args.prompt!r}", flush=True)

    t0 = time.time()
    r = gguf.GGUFReader(args.gguf)
    z, n_head, head_size = build_state(r, torch.float16)
    del r
    m = RWKV7.from_state(z, n_head, head_size, dtype=torch.float16).eval()
    print(f"[torch] L={m.n_layer} C={m.n_embd} H={m.n_head} N={m.head_size} "
          f"V={m.vocab_size} load {time.time()-t0:.1f}s", flush=True)

    sa, sk, sf = m.zero_state()
    logits = None
    t0 = time.time()
    with torch.no_grad():
        for t in ids:
            logits, sa, sk, sf = m(torch.tensor([t]), sa, sk, sf)
    print(f"[torch] prefill {len(ids)} tok in {time.time()-t0:.1f}s", flush=True)

    # 首个生成 token 的 top-10（留作数值对比）
    top = torch.topk(logits[0], 10)
    topk = [(int(i), float(v)) for i, v in zip(top.indices.tolist(), top.values.tolist())]

    gen, t0 = [], time.time()
    with torch.no_grad():
        for _ in range(args.n):
            nxt = int(torch.argmax(logits))
            gen.append(nxt)
            logits, sa, sk, sf = m(torch.tensor([nxt]), sa, sk, sf)
    dt = time.time() - t0
    text = tok.decode(gen)
    print(f"[torch] gen {args.n} tok in {dt:.1f}s ({args.n/dt:.1f} tok/s)", flush=True)
    print(f"[torch] ids: {gen}", flush=True)
    print(f"[torch] RWKV7-Torch gen: {text!r}", flush=True)
    print(f"[torch] top10(first gen): {topk}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"prompt": args.prompt, "ids": ids, "gen_ids": gen,
                       "text": text, "topk_first": topk}, f, ensure_ascii=False, indent=1)
        print(f"[torch] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
