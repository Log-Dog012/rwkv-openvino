#!/usr/bin/env python3
"""RWKV7-OV 分块生成接力器：绕开单进程跑不完端到端（prompt sweep + 生成总超 bash 290s）。

两段：
  phase=sweep : 跑 prompt sweep，存最终 logits + 全 L 层 states + stream/vf + n_out 到 npz。
  phase=gen   : 从 npz 加载 states + last_logits，只跑生成阶段（每 token 重载全 chunk）。

复用 rwkv7_ov_layerwise 的 F/R/_l2norm/_dequant_np/C_DTYPE、build_chunk、repack 等。
共用缓存 out/ov_cache（编译缓存启用后每 chunk ~3s）。
"""
import argparse, os, sys, time, gc, glob
import numpy as np
import openvino as ov
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_ov import _dequant_np
from rwkv7_ov_layerwise import build_chunk


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--phase", required=True, choices=["sweep", "gen"])
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=8)
    A.add_argument("--threads", type=int, default=8)
    A.add_argument("--chunk", type=int, default=8)
    A.add_argument("--ir-dir", default="out/chunks_13.3b")
    A.add_argument("--state", default="out/gen_state.npz")
    args = A.parse_args()

    t0 = time.time()
    r = gguf.GGUFReader(args.gguf)
    T = {t.name: t for t in r.tensors}
    emb_raw = _dequant_np(T["token_embd.weight"])
    V, C = emb_raw.shape
    L = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk."))
    H, N = C // 64, 64
    print(f"[接力] V={V} C={C} L={L} H={H} N={N} phase={args.phase}", flush=True)

    ln0_w = _dequant_np(T["token_embd_norm.weight"]).reshape(1, -1)
    ln0_b = _dequant_np(T["token_embd_norm.bias"]).reshape(1, -1)
    emb_ln = ((emb_raw - emb_raw.mean(-1, keepdims=True)) /
              np.sqrt(emb_raw.var(-1, keepdims=True) + 1e-5) * ln0_w + ln0_b)
    emb_ln = np.ascontiguousarray(emb_ln.astype(np.float16))
    del emb_raw

    core = ov.Core()
    try:
        core.set_property("CPU", {ov.properties.inference_num_threads: args.threads,
                                  ov.properties.cache_dir: "out/ov_cache"})
    except Exception:
        pass

    chunks = [(i, min(i + args.chunk, L)) for i in range(0, L, args.chunk)]
    states = [np.zeros((L, C), np.float16),
              np.zeros((L, H, N, N), np.float16),
              np.zeros((L, C), np.float16)]

    def build_or_load(ci):
        lo, hi = chunks[ci]
        fs = glob.glob(f"{args.ir_dir}/chunk{lo}_*.xml")
        if fs:
            return core.compile_model(fs[0], "CPU")
        m, _ = build_chunk(r, lo, hi, V, C, with_head=(hi == L))
        path = f"{args.ir_dir}/chunk{lo}_{hi}.xml"
        ov.save_model(m, path)
        del m; gc.collect()
        return core.compile_model(path, "CPU")

    def run_chunk(comp, ci, tok, x_in, vf_in):
        lo, hi = chunks[ci]
        feed = {}
        if lo == 0:
            feed["idx"] = np.array([tok], np.int64)
            feed["emb_table"] = emb_ln
        else:
            feed["x_in"] = np.ascontiguousarray(np.asarray(x_in, np.float16).reshape(1, C))
            feed["v_first"] = np.ascontiguousarray(np.asarray(vf_in, np.float16).reshape(1, C))
        feed["s_att"] = states[0][lo:hi]
        feed["s_kv"] = states[1][lo:hi]
        feed["s_ffn"] = states[2][lo:hi]
        req = comp.create_infer_request()
        req.infer(feed)
        off = 1 if hi == L else 0
        states[0][lo:hi] = np.array(req.get_output_tensor(off).data)
        states[1][lo:hi] = np.array(req.get_output_tensor(off + 1).data)
        states[2][lo:hi] = np.array(req.get_output_tensor(off + 2).data)
        x_out = np.array(req.get_output_tensor(off + 3).data)
        vf_out = np.array(req.get_output_tensor(off + 4).data)
        lg = np.array(req.get_output_tensor(0).data) if off == 1 else None
        del req
        return x_out, vf_out, lg

    from rwkv_tokenizer import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rwkv_vocab_v20230424.txt"))

    if args.phase == "sweep":
        ids = tok.encode(args.prompt)
        print(f"[接力] prompt({len(ids)} tok): {args.prompt!r}", flush=True)
        last_logits = None
        stream = list(ids); vf_stream = None
        for ci in range(len(chunks)):
            comp = build_or_load(ci)
            new_stream, new_vf = [], []
            for i, x_in in enumerate(stream):
                tokid = x_in if ci == 0 else None
                vf_in = None if ci == 0 else vf_stream[i]
                x_out, vf_out, lg = run_chunk(comp, ci, tokid, x_in if ci > 0 else None, vf_in)
                new_stream.append(x_out); new_vf.append(vf_out)
                if lg is not None:
                    last_logits = lg
            del comp; gc.collect()
            print(f"[接力] sweep chunk {ci} ({chunks[ci][0]}-{chunks[ci][1]-1}) done in {time.time()-t0:.1f}s", flush=True)
            stream, vf_stream = new_stream, new_vf
        # 存状态接力
        np.savez(args.state,
                 last_logits=np.asarray(last_logits, np.float16),
                 s_att=states[0], s_kv=states[1], s_ffn=states[2],
                 stream=np.asarray(stream, np.float16),
                 vf_stream=np.asarray([np.asarray(x, np.float16) for x in vf_stream]) if vf_stream else np.zeros((0,)),
                 n_out=args.n, prompt=args.prompt)
        am = int(np.argmax(last_logits[0])) if last_logits is not None else -1
        print(f"[接力] sweep done in {time.time()-t0:.1f}s, saved state -> {args.state}, next token id={am}", flush=True)

    else:  # phase=gen
        st = np.load(args.state, allow_pickle=True)
        states[0] = st["s_att"]; states[1] = st["s_kv"]; states[2] = st["s_ffn"]
        last_logits = st["last_logits"].astype(np.float16)
        stream = [np.asarray(x, np.float16) for x in st["stream"]]
        vf_stream = [np.asarray(x, np.float16) for x in st["vf_stream"]] if st["vf_stream"].size else None
        n_out = int(st["n_out"])
        print(f"[接力] loaded state from {args.state}, gen {n_out} tokens", flush=True)
        gen = []
        tb = time.time()
        for g in range(n_out):
            t = int(np.argmax(last_logits[0]))
            gen.append(t)
            x_in, vf_in = None, None
            for ci in range(len(chunks)):
                comp = build_or_load(ci)
                tokid = t if ci == 0 else None
                x_out, vf_out, lg = run_chunk(comp, ci, tokid, x_in if ci > 0 else None, vf_in if ci > 0 else None)
                x_in, vf_in = x_out, vf_out
                if lg is not None:
                    last_logits = lg
                del comp; gc.collect()
            print(f"[接力] gen {g+1}/{n_out} tok in {time.time()-tb:.1f}s: ids={gen} text={tok.decode(gen)!r}", flush=True)
        print(f"[接力] generated {n_out} tokens in {time.time()-tb:.1f}s", flush=True)
        print(f"[接力] gen ids: {gen}", flush=True)
        print(f"[接力] RWKV7-OV(chunked) gen: {tok.decode(gen)!r}", flush=True)


if __name__ == "__main__":
    main()
