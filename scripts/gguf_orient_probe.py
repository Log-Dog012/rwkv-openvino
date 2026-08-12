import gguf, numpy as np, torch
torch.set_num_threads(4)
GGUF='models/rwkv7-g1i-1.5b-Q4_K_M.gguf'
PTH='models/rwkv7-g1i-1.5b.pth'
r=gguf.GGUFReader(GGUF)
T={t.name:t for t in r.tensors}
def dq(n):
    t=T[n]
    if t.tensor_type.name in ('F32','F16'): return t.data.astype(np.float32)
    return gguf.dequantize(t.data,t.tensor_type).astype(np.float32)
raw=torch.load(PTH,map_location='cpu',mmap=True)
def corr(a,b):
    a=a.reshape(-1); b=b.reshape(-1); n=min(len(a),len(b)); a,b=a[:n],b[:n]
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else float('nan')
for gg in ['w1','a1','g1','v1','w2','a2','g2','v2']:
    g=dq(f'blk.0.time_mix_{gg}.weight')
    p=raw[f'blocks.0.att.{gg}'].to(torch.float32).numpy()
    print(f'{gg}: GGUF{g.shape} pth{p.shape} | raw_corr={corr(g,p):.4f} T_corr={corr(g.T,p):.4f}', flush=True)
print('DONE', flush=True)
