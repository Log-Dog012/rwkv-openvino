#!/usr/bin/env python3
"""增量生成：绕过沙箱 120s Bash 限制。
mode=prompt: 每调用处理至多 --chunk 个 prompt token 并更新状态（不生成）。
mode=gen   : 每调用生成 --chunk 个 token（要求 prompt 已处理完）。
状态（sa/sk/sf/last/pidx/gen）落盘 resume。"""
import os, sys, json, time, argparse, gc
import numpy as np, torch
torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import RWKV7
from rwkv_tokenizer import TRIE_TOKENIZER
from gguf_to_ov import build_state, repair_v12

GGUF="models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
PTH="models/rwkv7-g1i-1.5b.pth"
PROMPT="The Eiffel Tower is located in the city of"
ST_DIR="temp/incr_state"; os.makedirs(ST_DIR, exist_ok=True)
ST_TOK=os.path.join(ST_DIR,"tokens.json"); ST_SA=os.path.join(ST_DIR,"sa.npy")
ST_SK=os.path.join(ST_DIR,"sk.npy"); ST_SF=os.path.join(ST_DIR,"sf.npy")
ST_LAST=os.path.join(ST_DIR,"last.npy"); ST_PIDX=os.path.join(ST_DIR,"pidx.json")

A=argparse.ArgumentParser()
A.add_argument("--mode",choices=["prompt","gen"],required=True)
A.add_argument("--chunk",type=int,default=4)
A=A.parse_args()
tok=TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
ids=tok.encode(PROMPT)

print(f"[incr] building GGUF-repaired model ...",flush=True)
import gguf as _g
reader=_g.GGUFReader(GGUF)
z,nh,hs=build_state(reader,torch.float16)
repair_v12(z,PTH)
m=RWKV7.from_state(z,nh,hs,dtype=torch.float16).eval()
del z; gc.collect()

with torch.no_grad():
    m(*[torch.tensor([0],dtype=torch.int64),*m.zero_state()])  # warmup
    # 载入已有状态
    if os.path.exists(ST_PIDX):
        pidx=json.load(open(ST_PIDX)); gen=json.load(open(ST_TOK))
        sa=torch.from_numpy(np.load(ST_SA)); sk=torch.from_numpy(np.load(ST_SK))
        sf=torch.from_numpy(np.load(ST_SF)); last=torch.from_numpy(np.load(ST_LAST))
        print(f"[incr] resumed pidx={pidx} gen={len(gen)}",flush=True)
    else:
        pidx=0; gen=[]; sa,sk,sf=m.zero_state(); last=None

    if A.mode=="prompt":
        end=min(pidx+A.chunk, len(ids))
        for t in ids[pidx:end]:
            idx=torch.tensor([t],dtype=torch.int64)
            lg,sa,sk,sf=m(idx,sa,sk,sf); last=lg; gen.append(t)
        pidx=end
        print(f"[incr] prompt {pidx}/{len(ids)} processed",flush=True)
    else:
        assert pidx>=len(ids), f"prompt not done: {pidx}/{len(ids)}"
        t0=time.time()
        for _ in range(A.chunk):
            idx=torch.tensor([int(torch.argmax(last))],dtype=torch.int64)
            lg,sa,sk,sf=m(idx,sa,sk,sf); last=lg
            gen.append(int(torch.argmax(lg)))
        print(f"[incr] +{A.chunk} steps in {time.time()-t0:.1f}s",flush=True)

    np.save(ST_SA,sa.numpy()); np.save(ST_SK,sk.numpy()); np.save(ST_SF,sf.numpy())
    np.save(ST_LAST,last.numpy()); json.dump(gen,open(ST_TOK,"w")); json.dump(pidx,open(ST_PIDX,"w"))
    print(f"[incr] total {len(gen)} tokens; text: {tok.decode(gen)}",flush=True)
print("[incr] DONE",flush=True)
