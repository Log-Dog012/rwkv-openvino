#!/usr/bin/env python3
"""POC: 验证 Q6_K 在 OV 上的 bit-exact 在线解码 (照搬 gguf 源码 reshape 常量)。"""
import numpy as np
import torch
import openvino as ov
from gguf import GGUFReader, dequantize

GGUF = "models/rwkv7-g1i-7.2b-Q4_K_M.gguf"
r = GGUFReader(GGUF)
qt = [t for t in r.tensors if t.tensor_type.name == "Q6_K"][0]
raw = np.frombuffer(qt.data, dtype=np.uint8)
nblk = min(raw.shape[0] // 210, 4)  # POC 只取前几个 block 验证解码正确性
raw = raw[: nblk * 210].reshape(nblk, 210)

ql = raw[:, :128].astype(np.uint8)
qh = raw[:, 128:192].astype(np.uint8)
scales = raw[:, 192:208].view(np.int8).astype(np.float32)   # (n,16)
d = raw[:, 208:210].view(np.float16).astype(np.float32)     # (n,)
d_scaled = (d * scales).astype(np.float32)                   # (n,16)  -- 预计算，绕开 f16 view

ref = dequantize(qt.data[: nblk * 210], qt.tensor_type).reshape(nblk, 256)


class M(torch.nn.Module):
    def forward(self, ds, ql, qh):
        n = ds.shape[0]
        ql = ql.reshape(n, -1, 1, 64) >> torch.tensor([0, 4], dtype=torch.uint8).reshape(1, 1, 2, 1)
        ql = (ql & 0x0F).reshape(n, -1, 32)
        qh = qh.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(1, 1, 4, 1)
        qh = (qh & 0x03).reshape(n, -1, 32)
        q = (ql | (qh << torch.tensor(4, dtype=torch.uint8))).to(torch.int8) - torch.tensor(32, dtype=torch.int8)
        q = q.reshape(n, 16, -1).to(torch.float32)
        return (ds.reshape(n, 16, 1) * q).reshape(n, 256)


m = M()
ds_t = torch.from_numpy(d_scaled)
ql_t = torch.from_numpy(ql)
qh_t = torch.from_numpy(qh)
ovm = ov.convert_model(m, example_input=(ds_t, ql_t, qh_t))
comp = ov.Core().compile_model(ovm, "CPU")
ins = {i.any_name: t.numpy() for i, t in zip(ovm.inputs, [ds_t, ql_t, qh_t])}
o = comp(ins)[0]
err = np.abs(o - ref).max()
print(f"[poc] Q6K max|err| vs gguf.dequantize = {err:.6e}  -> {'BIT-EXACT OK' if err < 1e-2 else 'MISMATCH'}")
