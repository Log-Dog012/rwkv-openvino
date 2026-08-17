#!/usr/bin/env python3
"""OV 单图常驻推理 benchmark：load 全图 IR → 编译一次 → 推理循环跑 N token。

与 llamacpp llama-bench 同口径：单图常驻、权重一次性加载、推理循环测稳态 tg t/s。
绕开分块执行器每 token 重载全 chunk 的架构代价，反映 OV 真实推理速度。

用法:
  python scripts/ov_single_graph_bench.py <gguf> --ir out/rwkv7-13.3b-q4k_ov.xml \
      --device CPU --threads 8 --prompt "The Eiffel Tower is located in the city of" --n 16
"""
import argparse, os, sys, time
import numpy as np
import openvino as ov


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf", help="GGUF 模型（读 V/C/L/H/N 元数据，不构全图）")
    A.add_argument("--ir", required=True, help="已落盘的全图 IR xml 路径")
    A.add_argument("--device", default="CPU")
    A.add_argument("--threads", type=int, default=8)
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=16, help="生成 token 数")
    A.add_argument("--warmup", type=int, default=2, help="预热推理次数（不计稳态）")
    args = A.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gguf
    from rwkv7_ov import _dequant_np
    from rwkv_tokenizer import TRIE_TOKENIZER

    # ---- 超参（从 GGUF 元数据，不构全图）----
    r = gguf.GGUFReader(args.gguf)
    T = {t.name: t for t in r.tensors}
    emb_raw = _dequant_np(T["token_embd.weight"])
    V, C = emb_raw.shape
    L = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk."))
    H, N = C // 64, 64
    print(f"[bench] V={V} C={C} L={L} H={H} N={N} dev={args.device} threads={args.threads}", flush=True)

    # ---- load IR + 编译一次 ----
    core = ov.Core()
    try:
        props = {ov.properties.inference_num_threads: args.threads}
        if args.device == "CPU":
            props[ov.properties.cache_dir] = "out/ov_cache"
        core.set_property(args.device, props)
    except Exception as e:
        print(f"[bench] warn: set_property failed ({e})", flush=True)
    t0 = time.time()
    model = core.read_model(args.ir)
    comp = core.compile_model(model, args.device)
    print(f"[bench] read+compile in {time.time()-t0:.1f}s", flush=True)

    ins = {i.any_name: i for i in comp.inputs}
    req = comp.create_infer_request()
    print(f"[bench] inputs: {list(ins.keys())}", flush=True)

    # ---- tokenizer + 初始状态 ----
    tok = TRIE_TOKENIZER(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rwkv_vocab_v20230424.txt"))
    ids = tok.encode(args.prompt)
    print(f"[bench] prompt({len(ids)} tok): {args.prompt!r}", flush=True)

    sa = np.zeros((L, C), np.float16)
    sk = np.zeros((L, H, N, N), np.float16)
    sf = np.zeros((L, C), np.float16)

    # ---- prompt 扫描（吃 prompt，建状态）----
    last = None
    for t in ids:
        req.infer({ins["idx"]: np.array([t], np.int64),
                   ins["s_att_x"]: sa, ins["s_kv"]: sk, ins["s_ffn"]: sf})
        last = np.array(req.get_output_tensor(0).data)
        sa = np.asarray(req.get_output_tensor(1).data).reshape(L, C)
        sk = np.asarray(req.get_output_tensor(2).data).reshape(L, H, N, N)
        sf = np.asarray(req.get_output_tensor(3).data).reshape(L, C)
    print(f"[bench] prompt sweep done, first gen token id={int(np.argmax(last[0]))}", flush=True)

    # ---- 预热（不计稳态）----
    for _ in range(args.warmup):
        t = int(np.argmax(last))
        req.infer({ins["idx"]: np.array([t], np.int64),
                   ins["s_att_x"]: sa, ins["s_kv"]: sk, ins["s_ffn"]: sf})
        last = np.array(req.get_output_tensor(0).data)
        sa = np.asarray(req.get_output_tensor(1).data).reshape(L, C)
        sk = np.asarray(req.get_output_tensor(2).data).reshape(L, H, N, N)
        sf = np.asarray(req.get_output_tensor(3).data).reshape(L, C)

    # ---- 稳态生成测速 ----
    gen = []
    tb = time.time()
    for g in range(args.n):
        t = int(np.argmax(last))
        gen.append(t)
        req.infer({ins["idx"]: np.array([t], np.int64),
                   ins["s_att_x"]: sa, ins["s_kv"]: sk, ins["s_ffn"]: sf})
        last = np.array(req.get_output_tensor(0).data)
        sa = np.asarray(req.get_output_tensor(1).data).reshape(L, C)
        sk = np.asarray(req.get_output_tensor(2).data).reshape(L, H, N, N)
        sf = np.asarray(req.get_output_tensor(3).data).reshape(L, C)
    elapsed = time.time() - tb
    tps = args.n / elapsed
    print(f"[bench] === {args.device} tg: {tps:.2f} t/s ({args.n} tokens in {elapsed:.2f}s) ===", flush=True)
    print(f"[bench] gen ids: {gen}", flush=True)
    print(f"[bench] gen text: {tok.decode(gen)!r}", flush=True)


if __name__ == "__main__":
    main()
