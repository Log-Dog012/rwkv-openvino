"""
RWKV-7 (x070) 单步推理的纯 PyTorch 复刻，逐行对齐 pip 包 rwkv==0.8.32 的 RWKV_x070。

设计要点（为 OpenVINO 导出而定）：
  * 只实现 token-by-token 的 forward_one 递推，不用任何 CUDA 自定义算子；
  * state 打包为 3 个定长张量（而非 3*L 个 list 元素），convert_model 更友好：
        s_att_x : (L, C)        每层 att 的上一时刻 ln1 输出
        s_kv    : (L, H, N, N)  每层 att 的 KV 记忆矩阵
        s_ffn   : (L, C)        每层 ffn 的上一时刻 ln2 输出
  * 权重全部 register_buffer，trace 时固化成 IR 常量；
  * v_first 是"同一 token 内 layer0 的 v"，不跨 token，故不进 state。

只支持非 DeepEmbed 变体（权重里无 ffn.s_emb）——rwkv7-g1/g1d 皆如此。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_T_KEYS = ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")


class RWKV7(nn.Module):
    def __init__(self, path: str, dtype=torch.float32):
        super().__init__()
        raw = torch.load(path, map_location="cpu", mmap=True)
        assert not any("s_emb" in k for k in raw), "DeepEmbed 变体暂不支持"

        self.n_head, self.head_size = raw["blocks.0.att.r_k"].shape
        self.vocab_size, self.n_embd = raw["emb.weight"].shape
        self.n_layer = max(int(k.split(".")[1]) for k in raw if k.startswith("blocks.")) + 1
        self.dtype = dtype

        z = {}
        for k, v in raw.items():
            t = v
            if any(s in k for s in _T_KEYS):
                t = t.t()          # 官方装载时把这几类矩阵转置，forward 里用 x @ W
            t = t.squeeze()        # (1,1,C) -> (C,)
            if k.endswith("att.r_k"):
                t = t.flatten()    # (H,N) -> (H*N,)
            z[k] = t.to(dtype).contiguous()

        # emb 预先做 ln0，省掉每步一次 layer_norm
        z["emb.weight"] = F.layer_norm(
            z["emb.weight"], (self.n_embd,),
            weight=z["blocks.0.ln0.weight"], bias=z["blocks.0.ln0.bias"],
        )

        # buffer 名不能含 '.'，做一层映射
        self._map = {}
        for i, (k, v) in enumerate(z.items()):
            bn = f"b{i}"
            self.register_buffer(bn, v, persistent=False)
            self._map[k] = bn

        self._raw = raw
        self._z = z
        self.free_source()  # 权重已固化为 buffer，释放原始引用降峰值

    def w(self, k: str) -> torch.Tensor:
        return getattr(self, self._map[k])

    # 内存卫生：权重已 register_buffer 固化进 IR 常量，及时释放原始 state_dict 与中间字典，
    # 降低大模型 convert_model 时的常驻峰值（本沙箱 cgroup 仅 8GB）。
    def free_source(self):
        import gc
        if hasattr(self, "_raw"):
            del self._raw
        if hasattr(self, "_z"):
            del self._z
        gc.collect()

    def zero_state(self):
        L, C, H, N = self.n_layer, self.n_embd, self.n_head, self.head_size
        return (
            torch.zeros(L, C, dtype=self.dtype),
            torch.zeros(L, H, N, N, dtype=self.dtype),
            torch.zeros(L, C, dtype=self.dtype),
        )

    def forward(self, idx, s_att_x, s_kv, s_ffn):
        """idx: 形状 (1,) 的 int64 tensor（便于 OV 静态化）；返回 (logits[V], 新 3 state)"""
        H, N, C, L = self.n_head, self.head_size, self.n_embd, self.n_layer
        w = self.w

        x = w("emb.weight")[idx].reshape(C)   # idx: (1,) -> (C,)，保持数据依赖（真 Gather）
        if x.dim() == 2:                      # idx 为 shape [1] 时压平
            x = x.reshape(C)
        v_first = torch.zeros_like(x)

        new_att_x, new_kv, new_ffn = [], [], []

        for i in range(L):
            b, att, ffn = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
            x = x.reshape(C)   # 保证 x 始终是 (C,)，状态也是 (C,)；避免 batch 维混入

            # ---------------- time mix ----------------
            xa_in = F.layer_norm(x, (C,), weight=w(b + "ln1.weight"), bias=w(b + "ln1.bias"))
            d = s_att_x[i] - xa_in
            xr = xa_in + d * w(att + "x_r")
            xw = xa_in + d * w(att + "x_w")
            xk = xa_in + d * w(att + "x_k")
            xv = xa_in + d * w(att + "x_v")
            xa = xa_in + d * w(att + "x_a")
            xg = xa_in + d * w(att + "x_g")

            r = xr @ w(att + "receptance.weight")
            wl = torch.tanh(xw @ w(att + "w1")) @ w(att + "w2")
            k = xk @ w(att + "key.weight")
            v = xv @ w(att + "value.weight")
            a = torch.sigmoid(w(att + "a0") + (xa @ w(att + "a1")) @ w(att + "a2"))
            g = torch.sigmoid(xg @ w(att + "g1")) @ w(att + "g2")

            kk = F.normalize((k * w(att + "k_k")).view(H, N), dim=-1, p=2.0).view(H * N)
            k = k * (1 + (a - 1) * w(att + "k_a"))
            if i == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(
                    w(att + "v0") + (xv @ w(att + "v1")) @ w(att + "v2")
                )
            decay = torch.exp(-0.606531 * torch.sigmoid(w(att + "w0") + wl))  # exp(-0.5)=0.606531

            st = s_kv[i]
            vk = v.view(H, N, 1) @ k.view(H, 1, N)
            ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
            st = st * decay.view(H, 1, N) + st @ ab + vk

            # per-head 归一化（ln_x），逐通道 affine（数学等价于 F.group_norm）。
            # 不用 F.group_norm：OpenVINO 2026.3 的 torch->IR 前端转换该算子有数值偏差。
            # 为数值稳定，var/sqrt 在 float32 算，结果 cast 回 x.dtype，
            # 这样 --load-fp16（Half 权重）下后续矩阵乘 dtype 一致，FP32 路径则 identity 不受影响。
            _x = (st @ r.view(H, N, 1)).view(1, H, N).to(torch.float32)
            _mu = _x.mean(-1, keepdim=True)
            _var = _x.var(-1, unbiased=False, keepdim=True)
            _y = (_x - _mu) / torch.sqrt(_var + float(64e-5))
            o = _y.view(1, H * N).to(x.dtype) * w(att + "ln_x.weight") + w(att + "ln_x.bias")
            o = o + ((r * k * w(att + "r_k")).view(H, N).sum(dim=-1, keepdim=True)
                     * v.view(H, N)).view(H * N)

            x = x + (o * g) @ w(att + "output.weight")
            new_att_x.append(xa_in)
            new_kv.append(st)

            # ---------------- channel mix ----------------
            xf_in = F.layer_norm(x, (C,), weight=w(b + "ln2.weight"), bias=w(b + "ln2.bias"))
            kf = xf_in + (s_ffn[i] - xf_in) * w(ffn + "x_k")
            kf = torch.relu(kf @ w(ffn + "key.weight")) ** 2
            x = x + kf @ w(ffn + "value.weight")
            new_ffn.append(xf_in)

        x = F.layer_norm(x, (C,), weight=w("ln_out.weight"), bias=w("ln_out.bias"))
        logits = x @ w("head.weight")
        return logits, torch.stack(new_att_x), torch.stack(new_kv), torch.stack(new_ffn)
