"""二分定位：OV(idx=0,零状态) vs torch 逐层 pre-residual 输出，找第一个分歧层。"""
import os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); os.environ.setdefault("RWKV_JIT_ON","1"); os.environ["RWKV_CUDA_ON"]="0"
import torch, openvino as ov
from rwkv7_torch import RWKV7

m=RWKV7(os.path.join(ROOT,"models","rwkv7-g1d-0.1b.pth"),dtype=torch.float32).eval()
core=ov.Core(); comp=core.compile_model(os.path.join(ROOT,"out","rwkv7_g1d_0.1b_step_fp32.xml"),"CPU")
req=comp.create_infer_request(); inames=[i.any_name for i in comp.inputs]
idx=0
ov_out=req.infer({inames[0]:idx, inames[1]:m.zero_state()[0].numpy(),
                  inames[2]:m.zero_state()[1].numpy(), inames[3]:m.zero_state()[2].numpy()})
ov_logits=ov_out[0]

# torch 逐层
H,N,C,L=m.n_head,m.head_size,m.n_embd,m.n_layer; w=m.w
x=w("emb.weight")[idx]
if x.dim()==2: x=x.reshape(C)
v_first=torch.zeros_like(x)
# 重建 torch 的 att/ffn 输出需复刻层内逻辑；这里直接调用模型内部做分层
import torch.nn.functional as F
def layer_i(i):
    b,f"blocks.{i}." if False else f"blocks.{i}."
att=f"blocks.{i}.att."; ffn=f"blocks.{i}.ffn."; bb=f"blocks.{i}."
    xa=F.layer_norm(x,(C,),weight=w(bb+"ln1.weight"),bias=w(bb+"ln1.bias"))
    d=s_att_x_i - xa
    xr=xa+d*w(att+"x_r"); xw=xa+d*w(att+"x_w"); xk=xa+d*w(att+"x_k")
    xv=xa+d*w(att+"x_v"); xa2=xa+d*w(att+"x_a"); xg=xa+d*w(att+"x_g")
    r=xr@w(att+"receptance.weight")
    wl=torch.tanh(xw@w(att+"w1"))@w(att+"w2")
    k=xk@w(att+"key.weight"); v=xv@w(att+"value.weight")
    a=torch.sigmoid(w(att+"a0")+(xa2@w(att+"a1"))@w(att+"a2"))
    g=torch.sigmoid(xg@w(att+"g1"))@w(att+"g2")
    kk=F.normalize((k*w(att+"k_k")).view(H,N),dim=-1,p=2.0).view(H*N)
    k=k*(1+(a-1)*w(att+"k_a"))
    if i==0: v_first2=v
    else: v2=v+(v_first-(v))*(torch.sigmoid(w(att+"v0")+(xv@w(att+"v1"))@w(att+"v2")))
    decay=torch.exp(-0.606531*torch.sigmoid(w(att+"w0")+wl))
    st=torch.zeros(H,N,N)
    vk=v.view(H,N,1)@k.view(H,1,N); ab=(-kk).view(H,N,1)@(kk*a).view(H,1,N)
    st=st*decay.view(H,1,N)+st@ab+vk
    o=st@r.view(H,N,1)
    o=F.group_norm(o.view(1,H*N),num_groups=H,weight=w(att+"ln_x.weight"),bias=w(att+"ln_x.bias"),eps=64e-5).view(H*N)
    o=o+((r*k*w(att+"r_k")).view(H,N).sum(-1,keepdim=True)*v.view(H,N)).view(H*N)
    x2=x+(o*g)@w(att+"output.weight")
    return x2

# 简化：逐层调用模型并 capture pre-residual（用 forward 改造不便，这里用参考脚本的 RWKV_x070_TMix_one）
from rwkv.model import RWKV as RefRWKV  # 仅拿底层函数不易，改为直接对比 OV vs eager 全量
print("OV logits argmax =", int(torch.from_numpy(ov_logits).argmax()))
# eager
sa,sk,sf=m.zero_state()
with torch.no_grad(): elog,_,_,_=m(torch.tensor(idx),sa,sk,sf)
print("EAGER logits argmax =", int(elog.argmax()))
print("logits maxdiff =",(torch.from_numpy(ov_logits).float()-elog.float()).abs().max().item())
