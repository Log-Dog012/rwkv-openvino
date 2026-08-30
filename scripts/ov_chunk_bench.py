#!/usr/bin/env python3
"""分块 IR 常驻推理 benchmark（对标 llama-bench 口径）：任意设备 + 可选 f32 hint。

背景: GPU 压缩权重融合 FC 低精度累加 → 逐层分块 IR 每块单独编译时可用
INFERENCE_PRECISION_HINT=f32 修复数值（每块权重 f32 物化 ~1GB，内存受控；
单图整模 f32 hint 会物化 ~27GB 直接换页 250s/token，此路不通）。

用法:
  python scripts/ov_chunk_bench.py --outdir out/chunks_7.2b --device GPU [--f32-hint] \
      [--prompt "..."] [--n 16] [--warmup 2]
"""
import argparse, os, sys, time
import numpy as np
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--device", default="GPU")
ap.add_argument("--f32-hint", action="store_true")
ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--warmup", type=int, default=2)
args = ap.parse_args()

from rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(os.path.join(HERE, "rwkv_vocab_v20230424.txt"))

core = ov.Core()
if args.f32_hint:
    core.set_property(args.device, {ov.properties.hint.inference_precision: ov.Type.f32})

def load(p):
    return core.compile_model(p, args.device)

t0 = time.time()
emb = load(f"{args.outdir}/rwkv_chunk_emb.xml")
L = 1 + max(int(f.split("layer")[1].split(".")[0])
            for f in os.listdir(args.outdir) if f.startswith("rwkv_chunk_layer"))
lays = [load(f"{args.outdir}/rwkv_chunk_layer{i}.xml") for i in range(L)]
out = load(f"{args.outdir}/rwkv_chunk_out.xml")
models = [emb] + lays + [out]
reqs = [m.create_infer_request() for m in models]
print(f"[bench] chunks={L+2} (L={L}) load+compile {time.time()-t0:.1f}s dev={args.device} f32hint={args.f32_hint}", flush=True)

C = emb.output(0).shape[-1]
H, N = C // 64, 64

def feed(ci, **kv):
    m = {}
    for k, v in kv.items():
        et = models[ci].input(k).element_type
        m[k] = np.asarray(v, dtype=np.float16 if et == ov.Type.f16 else np.float32)
    return reqs[ci].infer(m)

def step(t, st):
    sa, sk, sf, vf = st
    o = feed(0, idx=np.array([t], np.int64))
    x = np.asarray(o[0]).reshape(C).astype(np.float32)
    for i in range(L):
        o = feed(1 + i, x=x, sa=sa, sk=sk, sf=sf, vf=vf)
        x = np.asarray(o[0]).reshape(C).astype(np.float32)
        sa = np.asarray(o[1]).reshape(C).astype(np.float32)
        sk = np.asarray(o[2]).reshape(H, N, N).astype(np.float32)
        sf = np.asarray(o[3]).reshape(C).astype(np.float32)
        vf = np.asarray(o[4]).reshape(C).astype(np.float32)
    logits = np.asarray(feed(1 + L, x=x)[0]).reshape(-1)
    return logits, (sa, sk, sf, vf)

st = (np.zeros(C, np.float32), np.zeros((H, N, N), np.float32),
      np.zeros(C, np.float32), np.zeros(C, np.float32))
ids = tok.encode(args.prompt)
gen, logits = [], None
t_all = 0.0; n_all = 0
for t in list(ids) + [None] * args.n:
    src = "sweep" if t is not None else "gen"
    if t is None:
        t = int(np.argmax(logits))
    t1 = time.time()
    logits, st = step(t, st)
    dt = time.time() - t1
    if t is not None:
        gen.append(t)
    else:
        n_all += 1; t_all += dt
        print(f"[bench] gen {n_all} in {dt:.3f}s ({n_all/t_all:.2f} t/s) {tok.decode(gen[-1:])!r}", flush=True)
print(f"[bench] === {args.device}{' f32' if args.f32_hint else ''} tg: {n_all/t_all:.2f} t/s ===", flush=True)
print("[bench] gen text:", tok.decode(gen), flush=True)
