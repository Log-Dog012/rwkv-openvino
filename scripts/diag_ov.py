"""逐 token 诊断 OV vs torch：打印前 3 token 的 logits max-diff 与状态形状，定位漂移来源。"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); os.environ.setdefault("RWKV_JIT_ON","1"); os.environ["RWKV_CUDA_ON"]="0"
import torch, openvino as ov
from rwkv_tokenizer import TRIE_TOKENIZER
from rwkv7_torch import RWKV7
MODEL=os.path.join(ROOT,"models","rwkv7-g1d-0.1b"); OUT=os.path.join(ROOT,"out")
tok=TRIE_TOKENIZER(os.path.join(HERE,"rwkv_vocab_v20230424.txt"))
m=RWKV7(MODEL+".pth",dtype=torch.float32).eval()
core=ov.Core()
xml=os.path.join(OUT,"rwkv7_g1d_0.1b_step_fp32.xml")
comp=core.compile_model(xml,"CPU"); req=comp.create_infer_request()
inames=[i.any_name for i in comp.inputs]
print("INPUTS :", inames, "| N outputs =", len(comp.outputs))
ids=tok.encode("The Eiffel Tower is located in the city of")
sa,sk,sf=m.zero_state()
s_att_x,s_kv,s_ffn=[x.numpy() for x in (sa,sk,sf)]
for step,t in enumerate(ids[:4]):
    out=req.infer({inames[0]:t,inames[1]:s_att_x,inames[2]:s_kv,inames[3]:s_ffn})
    ov_log=torch.from_numpy(out[0]).float()
    with torch.no_grad(): rlog,sa,sk,sf=m(torch.tensor(t),sa,sk,sf)
    d=(ov_log-rlog.float()).abs()
    print(f"step {step} token {t}: maxdiff={d.max():.3e} mean={d.mean():.3e} "
          f"argmax {'M' if int(ov_log.argmax())==int(rlog.argmax()) else 'X'} "
          f"| s_att_x={tuple(out[1].shape)} s_kv={tuple(out[2].shape)} s_ffn={tuple(out[3].shape)}")
    s_att_x,s_kv,s_ffn=out[1],out[2],out[3]
