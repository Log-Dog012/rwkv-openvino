#!/usr/bin/env python3
"""逐层分块导出 RWKV7 单步模型为多个小 OV IR（解决 >RAM 转换 + trace OOM）。

原理：RWKV 的递推逐层顺序执行，state 只在相邻层间流动。把整个单步 forward 拆成：
  emb 子模型(idx -> x) + L 个 layer 子模型(x, sa_i, sk_i, sf_i, v_first -> 新状态) + out 子模型(x -> logits)
每层只含该层权重（1.5B 每层 ~125MB fp16，trace 峰值 ~250MB，13.3B 每层 ~0.4GB 同样稳），
权重逐层流式装载（pth 用 mmap；gguf 逐张量反量化），整体内存恒定。

driver 在 Python 侧串起 L+2 个子模型完成逐 token 生成。

用法:
  python3 scripts/export_chunked.py <model.pth|model.gguf> --outdir models --n 16
  源为 .pth：mmap 逐层取权重；源为 .gguf：gguf 包逐层反量化。
"""
import argparse, gc, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import _T_KEYS


# ---------------- 逐层权重提取 ----------------

def _post(t, k, dtype):
    """与 rwkv7_torch loader 相同的后处理（转置/squeeze/flatten）。"""
    if any(s in k for s in _T_KEYS):
        t = t.t()
    t = t.squeeze()
    if k.endswith("att.r_k"):
        t = t.flatten()
    return t.to(dtype).contiguous()


class PthSource:
    """pth 源：mmap 后按层取键。"""
    def __init__(self, path, dtype):
        import torch as _t
        self.raw = _t.load(path, map_location="cpu", mmap=True)
        self.dtype = dtype
        self.H, self.N = self.raw["blocks.0.att.r_k"].shape
        self.C = self.raw["blocks.0.ln1.weight"].shape[0]
        self.L = 1 + max(int(k.split(".")[1]) for k in self.raw if k.startswith("blocks."))
        self.V = self.raw["emb.weight"].shape[0]

    def layer_keys(self, i):
        return [k for k in self.raw if k.startswith(f"blocks.{i}.")]

    def get(self, k):
        return _post(self.raw[k], k, self.dtype)


class GgufSource:
    """gguf 源：逐层反量化（内存恒定）。

    repair_pth: 可选参考 pth（bf16, mmap 懒加载）。shoumenchougou 的 RWKV7-G1i GGUF
    中 time_mix_v1/v2 被错写成 rank 96（与 a/w 混秩混淆）且值为垃圾；官方 pth 的
    v1/v2 为 rank 64 且语义正确。设为该 pth 后，get() 对 v1/v2 返回参考 pth 的值。
    """
    def __init__(self, path, dtype, repair_pth=None):
        import gguf
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.dtype = dtype
        self.repair_pth = repair_pth
        self._repair_raw = None
        emb = self._dq("token_embd.weight")
        self.C = int(emb.shape[1])
        self.V = int(emb.shape[0])
        self.L = max(int(n.split(".")[1]) for n in self.tensors if n.startswith("blk.")) + 1
        self.H, self.N = self.C // 64, 64

    def _repair_v(self, i, which):
        if self._repair_raw is None:
            self._repair_raw = torch.load(self.repair_pth, map_location="cpu", mmap=True)
        arr = self._repair_raw[f"blocks.{i}.att.{which}"].to(torch.float32).numpy()
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.dtype)

    def _dq(self, name):
        import gguf
        t = self.tensors[name]
        if t.tensor_type.name == "F32":
            return t.data.astype(np.float32)
        if t.tensor_type.name == "F16":
            return t.data.astype(np.float32)
        return gguf.dequantize(t.data, t.tensor_type).astype(np.float32)

    def _to(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.dtype)

    def get(self, k):  # k 为 pth 风格键
        if k.startswith("emb.weight"):
            return self._to(self._dq("token_embd.weight"))
        if k.startswith("ln_out."):
            return self._to(self._dq("output_norm." + k.split(".")[1]))
        if k.startswith("head.weight"):
            return self._to(self._dq("output.weight").T)
        if k.startswith("blocks.0.ln0."):
            return self._to(self._dq("token_embd_norm." + k.split(".")[-1]))
        i = int(k.split(".")[1])
        rest = k.split(".", 2)[2]  # 如 "att.w1" / "att.key.weight" / "ffn.x_k"
        b = f"blk.{i}."
        if rest.startswith("ln1."):
            return self._to(self._dq(b + "attn_norm." + rest.split(".")[1]))
        if rest.startswith("ln2."):
            return self._to(self._dq(b + "attn_norm_2." + rest.split(".")[1]))
        if rest.startswith("att.ln_x."):
            return self._to(self._dq(b + "time_mix_ln." + rest.split(".")[2]))  # weight/bias
        if rest.startswith("att.x_"):
            lf = self._dq(b + "time_mix_lerp_fused.weight").reshape(6, -1)
            order = {"x_r": 0, "x_w": 1, "x_k": 2, "x_v": 3, "x_a": 4, "x_g": 5}
            return self._to(lf[order[rest[4:]]])
        if rest.startswith("ffn.x_k"):
            return self._to(self._dq(b + "channel_mix_lerp_k.weight"))
        name = rest.split(".")[1] if "." in rest else rest  # w1 / key / key.weight
        # 方阵 att.{key,value,receptance,output}.weight（GGUF 同朝向，不转置）
        if rest in ("att.key.weight", "att.value.weight", "att.receptance.weight", "att.output.weight"):
            gg = rest.split(".")[1]
            return self._to(self._dq(b + f"time_mix_{gg}.weight"))
        # FFN 方阵 ffn.{key,value}.weight（GGUF 存转置，需 .T）
        if rest in ("ffn.key.weight", "ffn.value.weight"):
            gg = "key" if rest.startswith("ffn.key") else "value"
            return self._to(self._dq(b + f"channel_mix_{gg}.weight").T)
        # LoRA / 向量（无 .weight，GGUF 存转置）
        gg = {"w1": "w1", "w2": "w2", "a1": "a1", "a2": "a2", "g1": "g1", "g2": "g2",
              "v1": "v1", "v2": "v2", "w0": "w0", "a0": "a0", "k_a": "k_a", "k_k": "k_k",
              "v0": "v0", "r_k": "r_k"}[name]
        # 修复：v1/v2 用参考 pth 的正确 rank-64 值覆盖 GGUF 的坏 rank-96 张量
        if name in ("v1", "v2") and self.repair_pth is not None:
            return self._repair_v(i, name)
        arr = self._dq(b + f"time_mix_{gg}.weight")
        if name in ("w1", "w2", "a1", "a2", "g1", "g2", "v1", "v2"):
            arr = arr.T
        return self._to(arr)

    def layer_keys(self, i):
        return None  # get() 按需生成，无需枚举


# ---------------- 层子模型 ----------------

class EmbSub(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, idx):
        return self.emb[idx].reshape(-1)  # (C,)


class OutSub(nn.Module):
    def __init__(self, ln_w, ln_b, head):
        super().__init__()
        self.register_buffer("ln_w", ln_w, persistent=False)
        self.register_buffer("ln_b", ln_b, persistent=False)
        self.register_buffer("head", head, persistent=False)

    def forward(self, x):
        C = x.shape[-1]
        x = F.layer_norm(x, (C,), weight=self.ln_w, bias=self.ln_b)
        return x @ self.head  # (V,)


class LayerSub(nn.Module):
    """第 i 层子模型。输入 (x, sa_i, sk_i, sf_i, v_first) -> (x, nsa, nsk, nsf, v_first)。"""
    def __init__(self, i, H, N, C, wget):
        super().__init__()
        self.H, self.N, self.C = H, N, C
        self.i = i
        self._map = {}
        names = (
            "ln1.weight", "ln1.bias", "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g",
            "att.receptance.weight", "att.w1", "att.w2", "att.key.weight", "att.value.weight",
            "att.a0", "att.a1", "att.a2", "att.g1", "att.g2", "att.k_k", "att.k_a",
            "att.v0", "att.v1", "att.v2", "att.w0", "att.ln_x.weight", "att.ln_x.bias",
            "att.r_k", "att.output.weight", "ln2.weight", "ln2.bias",
            "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
        )
        for j, nm in enumerate(names):
            bn = f"b{j}"
            self.register_buffer(bn, wget(f"blocks.{i}.{nm}").contiguous(), persistent=False)
            self._map[nm] = bn

    def w(self, k):
        return getattr(self, self._map[k])

    def forward(self, x, sa_i, sk_i, sf_i, v_first):
        H, N, C = self.H, self.N, self.C
        w = self.w
        x = x.reshape(C)
        # time mix
        xa_in = F.layer_norm(x, (C,), weight=w("ln1.weight"), bias=w("ln1.bias"))
        d = sa_i - xa_in
        xr = xa_in + d * w("att.x_r"); xw = xa_in + d * w("att.x_w")
        xk = xa_in + d * w("att.x_k"); xv = xa_in + d * w("att.x_v")
        xa = xa_in + d * w("att.x_a"); xg = xa_in + d * w("att.x_g")
        r = xr @ w("att.receptance.weight")
        wl = torch.tanh(xw @ w("att.w1")) @ w("att.w2")
        k = xk @ w("att.key.weight")
        v = xv @ w("att.value.weight")
        a = torch.sigmoid(w("att.a0") + (xa @ w("att.a1")) @ w("att.a2"))
        g = torch.sigmoid(xg @ w("att.g1")) @ w("att.g2")
        kk = F.normalize((k * w("att.k_k")).view(H, N), dim=-1, p=2.0).view(H * N)
        k = k * (1 + (a - 1) * w("att.k_a"))
        if self.i == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(w("att.v0") + (xv @ w("att.v1")) @ w("att.v2"))
        decay = torch.exp(-0.606531 * torch.sigmoid(w("att.w0") + wl))
        st = sk_i
        vk = v.view(H, N, 1) @ k.view(H, 1, N)
        ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
        st = st * decay.view(H, 1, N) + st @ ab + vk
        _x = (st @ r.view(H, N, 1)).view(1, H, N).to(torch.float32)
        _mu = _x.mean(-1, keepdim=True)
        _var = _x.var(-1, unbiased=False, keepdim=True)
        _y = (_x - _mu) / torch.sqrt(_var + float(64e-5))
        o = _y.view(1, H * N).to(x.dtype) * w("att.ln_x.weight") + w("att.ln_x.bias")
        o = o + ((r * k * w("att.r_k")).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
        x = x + (o * g) @ w("att.output.weight")
        nsa = xa_in
        nsk = st
        # channel mix
        xf_in = F.layer_norm(x, (C,), weight=w("ln2.weight"), bias=w("ln2.bias"))
        kf = xf_in + (sf_i - xf_in) * w("ffn.x_k")
        kf = torch.relu(kf @ w("ffn.key.weight")) ** 2
        x = x + kf @ w("ffn.value.weight")
        nsf = xf_in
        return x, nsa, nsk, nsf, v_first


# ---------------- 导出与 driver ----------------

def export(src_path, outdir, n, dtype=torch.float16, repair_v12=None, start=0, end=None, do_gen=True):
    src = GgufSource(src_path, dtype, repair_pth=repair_v12) if src_path.endswith(".gguf") else PthSource(src_path, dtype)
    H, N, C, L = src.H, src.N, src.C, src.L
    end = L if end is None else min(end, L)
    print(f"[chunk] L={L} C={C} H={H} N={N} V={src.V} range=[{start},{end})", flush=True)
    os.makedirs(outdir, exist_ok=True)
    import openvino as ov
    core = ov.Core()

    # emb 子模型（含 ln0 预归一：emb 权重直接取 loader 预处理后的值）
    raw_emb = src.get("emb.weight")
    ln0w, ln0b = src.get("blocks.0.ln0.weight"), src.get("blocks.0.ln0.bias")
    emb_normed = F.layer_norm(raw_emb.float(), (C,), weight=ln0w.float(), bias=ln0b.float()).to(dtype)
    m_emb = EmbSub(emb_normed.contiguous())
    idx = torch.zeros((1,), dtype=torch.int64)
    om = ov.convert_model(m_emb, example_input=idx)
    ov.save_model(om, f"{outdir}/rwkv_chunk_emb.xml", compress_to_fp16=(dtype == torch.float16))
    del m_emb; gc.collect()

    # 逐层导出（支持 --start/--end 分批，跨 120s Bash 限制）
    for i in range(start, end):
        m = LayerSub(i, H, N, C, src.get)
        x = torch.zeros(C, dtype=dtype)
        sa = torch.zeros(C, dtype=dtype)
        sk = torch.zeros(H, N, N, dtype=dtype)
        sf = torch.zeros(C, dtype=dtype)
        vf = torch.zeros(C, dtype=dtype)
        t0 = time.time()
        ovm = ov.convert_model(m, example_input=(x, sa, sk, sf, vf))
        ov.save_model(ovm, f"{outdir}/rwkv_chunk_layer{i}.xml", compress_to_fp16=(dtype == torch.float16))
        print(f"[chunk] layer{i} exported in {time.time()-t0:.1f}s", flush=True)
        del m, ovm; gc.collect()

    # out 子模型
    m_out = OutSub(src.get("ln_out.weight"), src.get("ln_out.bias"), src.get("head.weight"))
    x = torch.zeros(C, dtype=dtype)
    ovm = ov.convert_model(m_out, example_input=x)
    ov.save_model(ovm, f"{outdir}/rwkv_chunk_out.xml", compress_to_fp16=(dtype == torch.float16))
    del m_out, ovm; gc.collect()
    print("[chunk] all exported", flush=True)

    # driver：逐 token 生成（--no-gen 时跳过，仅导出 IR）
    if do_gen:
        sys.path.insert(0, "scripts")
        from rwkv_tokenizer import TRIE_TOKENIZER
        tok = TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
        embs = [core.compile_model(f"{outdir}/rwkv_chunk_emb.xml", "CPU")]
        lays = [core.compile_model(f"{outdir}/rwkv_chunk_layer{i}.xml", "CPU") for i in range(L)]
        outs = [core.compile_model(f"{outdir}/rwkv_chunk_out.xml", "CPU")]
        reqs = [c.create_infer_request() for c in embs + lays + outs]
        nins = [[i.any_name for i in c.inputs] for c in embs + lays + outs]

        def run(core_i, feed):
            # 按编译模型输入的 element_type 自动 cast，避免 fp16 模型喂 float32 报错
            c = embs + lays + outs
            out = {}
            for k, v in feed.items():
                et = c[core_i].input(k).element_type
                if et == ov.Type.f16:
                    out[k] = np.asarray(v, dtype=np.float16)
                else:
                    out[k] = np.asarray(v, dtype=np.float32)
            return reqs[core_i].infer(out)

        ids = tok.encode("The Eiffel Tower is located in the city of")
        gen = []
        logits = None
        for t in list(ids) + [None] * n:
            if t is None:
                t = int(np.argmax(logits))
            o = run(0, {nins[0][0]: np.array([t], np.int64)})
            x = np.asarray(o[0]).reshape(C).astype(np.float32)
            sa = np.zeros(C, np.float32); sk = np.zeros((H, N, N), np.float32); sf = np.zeros(C, np.float32)
            vf = np.zeros(C, np.float32)
            for i in range(L):
                o = run(1 + i, {nins[1 + i][0]: x, nins[1 + i][1]: sa, nins[1 + i][2]: sk,
                                nins[1 + i][3]: sf, nins[1 + i][4]: vf})
                x = np.asarray(o[0]).reshape(C).astype(np.float32)
                sa = np.asarray(o[1]).reshape(C).astype(np.float32)
                sk = np.asarray(o[2]).reshape(H, N, N).astype(np.float32)
                sf = np.asarray(o[3]).reshape(C).astype(np.float32)
                vf = np.asarray(o[4]).reshape(C).astype(np.float32)
            logits = np.asarray(run(1 + L, {nins[1 + L][0]: x})[0]).reshape(-1)
            if t is not None:
                gen.append(t)
        print("[chunk] gen:", tok.decode(gen))


if __name__ == "__main__":
    A = argparse.ArgumentParser()
    A.add_argument("src")
    A.add_argument("--outdir", default="models")
    A.add_argument("--n", type=int, default=16)
    A.add_argument("--dtype", default="fp16")
    A.add_argument("--repair-v12", default=None, help="参考 pth，用其正确 rank-64 的 v1/v2 覆盖 GGUF 坏张量")
    A.add_argument("--start", type=int, default=0, help="分层导出起始层（含），跨 120s 限制分批")
    A.add_argument("--end", type=int, default=None, help="分层导出结束层（不含）")
    A.add_argument("--no-gen", action="store_true", help="仅导出 IR，不跑生成")
    A = A.parse_args()
    torch.set_num_threads(8)
    export(A.src, A.outdir, A.n, torch.float16 if A.dtype == "fp16" else torch.float32,
           repair_v12=A.repair_v12, start=A.start, end=A.end, do_gen=not A.no_gen)
