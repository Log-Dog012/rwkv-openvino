"""
基线验证：
  1) 用复刻的 RWKV7 单步递推生成文本（greedy），确认输出连贯；
  2) 与官方 rwkv==0.8.32 包（cpu fp32）逐 token 对比 logits，给出 max/mean abs diff。

用法: python3 run_torch_baseline.py [--n 24] [--no-ref]
"""
import os, sys, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ["RWKV_CUDA_ON"] = "0"
os.environ["RWKV_V7_ON"] = "1"   # 必须：否则 rwkv 包的 RWKV 类会走 v6 分支

import torch
from rwkv_tokenizer import TRIE_TOKENIZER
from rwkv7_torch import RWKV7

MODEL = os.path.join(ROOT, "models", "rwkv7-g1d-0.1b")
PROMPT = "The Eiffel Tower is located in the city of"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()

    tok = TRIE_TOKENIZER(os.path.join(HERE, "rwkv_vocab_v20230424.txt"))
    ids = tok.encode(PROMPT)
    print(f"prompt: {PROMPT!r}\ntokens: {ids}\n")

    t0 = time.time()
    m = RWKV7(MODEL + ".pth", dtype=torch.float32).eval()
    print(f"[mine] loaded in {time.time()-t0:.1f}s  L={m.n_layer} C={m.n_embd} "
          f"H={m.n_head} N={m.head_size} V={m.vocab_size}")

    # ---------------- 我的复刻：prefill + greedy 生成 ----------------
    sa, sk, sf = m.zero_state()
    logits = None
    t0 = time.time()
    with torch.no_grad():
        for t in ids:
            logits, sa, sk, sf = m(torch.tensor(t), sa, sk, sf)
    pre_dt = time.time() - t0
    my_prefill_logits = logits.clone()

    out_ids, t0 = [], time.time()
    with torch.no_grad():
        for _ in range(args.n):
            nxt = int(torch.argmax(logits))
            out_ids.append(nxt)
            logits, sa, sk, sf = m(torch.tensor(nxt), sa, sk, sf)
    gen_dt = time.time() - t0
    print(f"\n[mine] prefill {len(ids)} tok in {pre_dt:.2f}s "
          f"({len(ids)/pre_dt:.1f} tok/s), gen {args.n} tok in {gen_dt:.2f}s "
          f"({args.n/gen_dt:.1f} tok/s)")
    print(f"[mine] greedy out: {tok.decode(out_ids)!r}")
    print(f"[mine] top5 after prompt: "
          f"{[tok.decode([i]) for i in torch.topk(my_prefill_logits, 5).indices.tolist()]}")

    if args.no_ref:
        return

    # ---------------- 官方包参考 ----------------
    sys.path.insert(0, os.path.join(ROOT, "temp", "rwkv-0.8.32", "src"))
    from rwkv.model import RWKV as RefRWKV
    ref = RefRWKV(model=MODEL, strategy="cpu fp32")

    ref_state, ref_logits = None, None
    t0 = time.time()
    for t in ids:
        ref_logits, ref_state = ref.forward([t], ref_state)
    ref_dt = time.time() - t0
    print(f"\n[ref ] prefill {len(ids)} tok in {ref_dt:.2f}s ({len(ids)/ref_dt:.1f} tok/s)")
    print(f"[ref ] top5 after prompt: "
          f"{[tok.decode([i]) for i in torch.topk(ref_logits, 5).indices.tolist()]}")

    d = (my_prefill_logits.float() - ref_logits.float()).abs()
    rel = d.max() / ref_logits.float().abs().max()
    print(f"\n=== logits diff (after {len(ids)} tok) ===")
    print(f"max abs = {d.max():.3e}   mean abs = {d.mean():.3e}   "
          f"max/|ref|max = {rel:.3e}")
    print(f"argmax mine={int(my_prefill_logits.argmax())} ref={int(ref_logits.argmax())} "
          f"-> {'MATCH' if int(my_prefill_logits.argmax())==int(ref_logits.argmax()) else 'MISMATCH'}")

    # 参考实现的 greedy 续写，作为文本级对照
    ref_out, rl, rs = [], ref_logits, ref_state
    for _ in range(args.n):
        nxt = int(torch.argmax(rl))
        ref_out.append(nxt)
        rl, rs = ref.forward([nxt], rs)
    print(f"[ref ] greedy out: {tok.decode(ref_out)!r}")
    print(f"TEXT {'MATCH' if ref_out == out_ids else 'DIFFER'}")


if __name__ == "__main__":
    main()
