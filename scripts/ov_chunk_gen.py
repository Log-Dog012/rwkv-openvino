#!/usr/bin/env python3
"""仅加载已导出的分块 OV IR 并生成（不重建权重，省 42s）。"""
import os, sys, argparse
import numpy as np
import openvino as ov
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_tokenizer import TRIE_TOKENIZER

A=argparse.ArgumentParser()
A.add_argument("--outdir", required=True)
A.add_argument("--n", type=int, default=16)
A=A.parse_args()
tok=TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")

core=ov.Core()
embs=[core.compile_model(f"{A.outdir}/rwkv_chunk_emb.xml","CPU")]
lays=[core.compile_model(f"{A.outdir}/rwkv_chunk_layer{i}.xml","CPU") for i in range(24)]
outs=[core.compile_model(f"{A.outdir}/rwkv_chunk_out.xml","CPU")]
reqs=[c.create_infer_request() for c in embs+lays+outs]
nins=[[i.any_name for i in c.inputs] for c in embs+lays+outs]
L=len(lays)
C=embs[0].output(0).shape[-1]
H,N=32,64

def run(ci,feed):
    c=embs+lays+outs; out={}
    for k,v in feed.items():
        et=c[ci].input(k).element_type
        out[k]=np.asarray(v,dtype=np.float16 if et==ov.Type.f16 else np.float32)
    return reqs[ci].infer(out)

ids=tok.encode("The Eiffel Tower is located in the city of")
gen=[]; logits=None
for t in list(ids)+[None]*A.n:
    if t is None: t=int(np.argmax(logits))
    o=run(0,{nins[0][0]:np.array([t],np.int64)})
    x=np.asarray(o[0]).reshape(C).astype(np.float32)
    sa=np.zeros(C,np.float32); sk=np.zeros((H,N,N),np.float32); sf=np.zeros(C,np.float32); vf=np.zeros(C,np.float32)
    for i in range(L):
        o=run(1+i,{nins[1+i][0]:x,nins[1+i][1]:sa,nins[1+i][2]:sk,nins[1+i][3]:sf,nins[1+i][4]:vf})
        x=np.asarray(o[0]).reshape(C).astype(np.float32)
        sa=np.asarray(o[1]).reshape(C).astype(np.float32)
        sk=np.asarray(o[2]).reshape(H,N,N).astype(np.float32)
        sf=np.asarray(o[3]).reshape(C).astype(np.float32)
        vf=np.asarray(o[4]).reshape(C).astype(np.float32)
    logits=np.asarray(run(1+L,{nins[1+L][0]:x})[0]).reshape(-1)
    if t is not None: gen.append(t)
print("[ov-chunk] gen:", tok.decode(gen))
