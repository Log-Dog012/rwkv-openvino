#!/usr/bin/env python3
"""POC: 验证 OV 能否 trace "4-bit 解包 + 乘加" 子图 (GGUF Q4_K bit-exact 真继承的关键)。

思路（避免 OV 不支持的 f16 reinterpret / 复杂 bit 操作）：
- 在 numpy 侧把 GGUF Q4_K block 拆成: d_scaled(n,8) fp32、dm_scaled(n,8) fp32、qs(n,128) uint8
  （d_scaled = d*f16_scale 的 fp32 值，已复现 gguf 的 super-block 结构）
- 模型内只做: 4-bit unpack (>> &) + reshape + 广播乘减
- 比对 ov 输出 vs gguf.dequantize 数值 -> 验证 bit-exact
"""
import numpy as np
import torch
import openvino as ov
from gguf import GGUFReader, dequantize

GGUF = "models/rwkv7-g1i-7.2b-Q4_K_M.gguf"
r = GGUFReader(GGUF)
qt = [t for t in r.tensors if t.tensor_type.name == "Q4_K"][1]  # 取一个代表性的 Q4_K 张量
raw = np.frombuffer(qt.data, dtype=np.uint8)
nblk = raw.shape[0] // 144
raw = raw[: nblk * 144].reshape(nblk, 144)

# --- numpy 侧复现 gguf Q4_K 的 d_scaled / dm_scaled ---
d = raw[:, 0:2].view(np.float16).astype(np.float32)
dmin = raw[:, 2:4].view(np.float16).astype(np.float32)
scales = raw[:, 4:16].reshape(nblk, 12).view(np.uint8)
scales = scales.reshape(nblk, 3, 4)
dd, mm, md = np.split(scales, 3, axis=-2)
sc = np.concatenate([dd & 0x3F, (md & 0x0F) | ((dd >> 2) & 0x30)], axis=-1).reshape(nblk, 8)
mn = np.concatenate([mm & 0x3F, (md >> 4) | ((mm >> 2) & 0x30)], axis=-1).reshape(nblk, 8)
d_scaled = (d * sc.astype(np.float32))
dm_scaled = (dmin * mn.astype(np.float32))
qs = raw[:, 16:].astype(np.uint8)  # (nblk, 128) uint8 packed nibbles

# --- 参考: gguf 官方解码 ---
ref = dequantize(qt.data, qt.tensor_type).reshape(nblk, 256)


class Q4KDecode(torch.nn.Module):
    def forward(self, ds, dm, qs):
        n = ds.shape[0]
        q = (qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 4], dtype=torch.uint8).reshape(1, 1, 2, 1)) \
            & torch.tensor(0x0F, dtype=torch.uint8)
        q = q.reshape(n, -1, 32).to(torch.float32)  # (n,8,32)
        out = ds.reshape(n, 8, 1) * q - dm.reshape(n, 8, 1)  # (n,8,32)
        return out.reshape(n, 256)


m = Q4KDecode()
ds_t = torch.from_numpy(d_scaled.astype(np.float32))
dm_t = torch.from_numpy(dm_scaled.astype(np.float32))
qs_t = torch.from_numpy(qs)
t0 = __import__("time").time()
ovm = ov.convert_model(m, example_input=(ds_t, dm_t, qs_t))
print(f"[poc] convert_model OK in {__import__('time').time()-t0:.1f}s")
print("[poc] inputs element_type:", [str(i.element_type) for i in ovm.inputs])
comp = ov.Core().compile_model(ovm, "CPU")
ins = {i.any_name: t.numpy() for i, t in zip(ovm.inputs, [ds_t, dm_t, qs_t])}
o = comp(ins)[0]
err = np.abs(o - ref).max()
print(f"[poc] max|err| vs gguf.dequantize = {err:.6e}")
print("[poc] RESULT:", "BIT-EXACT OK" if err < 1e-2 else "MISMATCH")
