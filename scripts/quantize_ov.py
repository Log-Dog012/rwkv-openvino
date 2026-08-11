"""
NNCF 权重量化（W8A32 INT8）：直接把 IR 权重压成 INT8，无需校准集。
对比 FP32/FP16 的体积与 OV 推理精度（logits diff + 生成文本）。
"""
import os, sys, time, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE)
os.environ.setdefault("RWKV_JIT_ON","1"); os.environ["RWKV_CUDA_ON"]="0"

import torch, openvino as ov, nncf
from rwkv_tokenizer import TRIE_TOKENIZER
from rwkv7_torch import RWKV7

MODEL=os.path.join(ROOT,"models","rwkv7-g1d-0.1b.pth")
OUT=os.path.join(ROOT,"out")
PROMPT="The Eiffel Tower is located in the city of"

def load_compiled(xml):
    core=ov.Core(); comp=core.compile_model(xml,"CPU")
    return comp, comp.create_infer_request(), [i.any_name for i in comp.inputs]

def run(comp, req, ins, m, n=20):
    tok=TRIE_TOKENIZER(os.path.join(HERE,"rwkv_vocab_v20230424.txt"))
    ids=tok.encode(PROMPT)
    L,H,N,C=m.n_layer,m.n_head,m.head_size,m.n_embd
    s_att_x,s_kv,s_ffn=[x.numpy() for x in m.zero_state()]
    ref_sa,ref_sk,ref_sf=m.zero_state(); maxd=0.0; nsteps=0; t0=time.time()
    with torch.no_grad():
        for t in ids:
            o=req.infer({ins[0]:np.array([t],dtype=np.int64),ins[1]:s_att_x,ins[2]:s_kv,ins[3]:s_ffn})
            rl,ref_sa,ref_sk,ref_sf=m(torch.tensor([t]),ref_sa,ref_sk,ref_sf)
            maxd=max(maxd,float((torch.from_numpy(o[0]).float()-rl.float()).abs().max()))
            s_att_x=np.asarray(o[1]).reshape(L,C); s_kv=np.asarray(o[2]).reshape(L,H,N,N); s_ffn=np.asarray(o[3]).reshape(L,C); nsteps+=1
        gen=[]
        for _ in range(n):
            nx=int(torch.argmax(torch.from_numpy(o[0])))
            o=req.infer({ins[0]:np.array([nx],dtype=np.int64),ins[1]:s_att_x,ins[2]:s_kv,ins[3]:s_ffn})
            s_att_x=np.asarray(o[1]).reshape(L,C); s_kv=np.asarray(o[2]).reshape(L,H,N,N); s_ffn=np.asarray(o[3]).reshape(L,C)
            gen.append(nx); nsteps+=1
    dt=time.time()-t0
    return maxd,nsteps,dt,tok.decode(gen)

def main():
    m=RWKV7(MODEL,dtype=torch.float32).eval()
    for src,tag in [("rwkv7_g1d_0.1b_step_fp32.xml","FP32"),
                    ("rwkv7_g1d_0.1b_step_fp16.xml","FP16")]:
        xml=os.path.join(OUT,src)
        comp,req,ins=load_compiled(xml)
        maxd,nsteps,dt,text=run(comp,req,ins,m)
        sz=os.path.getsize(xml)+os.path.getsize(xml.replace(".xml",".bin"))
        print(f"[{tag}] {src}\n  size={sz/1e6:.1f}MB  maxdiff={maxd:.3e}  {nsteps}steps/{dt:.2f}s={nsteps/dt:.1f}tok/s\n  out={text!r}\n")

    # NNCF 权重量化：INT8（W8A32）与 INT4（W4，体积接近 GGUF-Q4）对比
    ovm=ov.Core().read_model(os.path.join(OUT,"rwkv7_g1d_0.1b_step_fp32.xml"))
    for mode,tag in [("INT8_SYM","INT8 (W8A32)"),("INT4_SYM","INT4_SYM (W4)")]:
        print(f"=== NNCF 权重量化 {tag} ===")
        t0=time.time()
        qm=nncf.compress_weights(ovm, mode=getattr(nncf.CompressWeightsMode, mode))
        print(f"  quantized in {time.time()-t0:.1f}s")
        qxml=os.path.join(OUT,f"rwkv7_g1d_0.1b_step_{mode.lower()}.xml")
        ov.save_model(qm,qxml,compress_to_fp16=False)
        sz=os.path.getsize(qxml)+os.path.getsize(qxml.replace(".xml",".bin"))
        comp,req,ins=load_compiled(qxml)
        maxd,nsteps,dt,text=run(comp,req,ins,m)
        print(f"[{tag}] size={sz/1e6:.1f}MB  maxdiff={maxd:.3e}  {nsteps}steps/{dt:.2f}s={nsteps/dt:.1f}tok/s\n  out={text!r}")

if __name__=="__main__":
    main()
