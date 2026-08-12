#!/usr/bin/env python3
"""OpenVINO 原生 RWKV7 单步推理图（压缩权重继承路线）。

与 gguf_to_ov.py 的 export 子命令（先解量化 fp16 再 convert_model，7.2B 会 OOM）不同，
本脚本直接用 OV opset13 手写计算图：每个 Linear 权重走 build_compressed_weight()
产出的"原生压缩权重子图"（Constant u4/i8 -> Convert f16 -> [Subtract zp] -> Multiply
scale），由 CPU/GPU 插件融合成 int4/int8 内核，权重**不落 fp16**，7.2B Q4 仅占 ~2.4GB。

参考:
  - rwkv7_torch.py 的 forward()（逐行对齐）
  - gguf_to_ov.py 的 build_state()（GGUF 张量名 -> pth 风格键名映射，corr=1.0 已验证）
  - gguf_to_ov_compressed.build_compressed_weight()（bit-exact 已验证 Q4_K/Q6_K）

权重方向（关键，_orient_test.py 实测定论）:
  gguf_to_ov.build_state() 对 key/value/receptance/output 及全部 LoRA 对/w1/w2/ffn 都做了 orient="T"（.T 转置），
  => 即 torch 前向里的 matmul 操作数 W_torch = GGUF_raw.T（已用 x@raw 与 x@bs 对比确认 8 类权重全部 bs==raw.T）。
  因此 OV 直接吃 GGUF 原生布局 W_gguf 时，必须用 MatMul(x[1,in], W_gguf[d0,d1], transpose_b=True)
  = x @ W_gguf.T = x @ W_torch = torch 结果（bit-exact 对齐前提）。
  状态更新里的 vk=v@k、ab=(-kk)@(kk*a)、st@ab、st@r 是真实矩阵乘积（操作数非权值），保持 transpose_b=False。
  emb 用 Gather（无转置），ln0 预归一在 Gather 前固化到 emb_const。

用法:
  python3 rwkv7_ov.py <model.gguf> [--device CPU] [--prompt "..."] [--n 16]
"""
import argparse, os, sys, time
import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_kquant_repack import repack_tensor
from gguf_to_ov_compressed import build_compressed_weight
import openvino as ov
from openvino import opset13 as ops

C_DTYPE = ov.Type.f16


def F(v, dtype=C_DTYPE, name=None):
    """标量常量。默认 f16 匹配 fp16 图；group norm 内部段传 ov.Type.f32。"""
    return ops.constant(np.array(v, dtype=(np.float16 if dtype == C_DTYPE else np.float32)), dtype, name=name)


def R(node, shape, name):
    """Node 没有 .reshape 方法，统一走 ops.reshape（special_zero=True 支持 -1 推断）。"""
    return ops.reshape(node, shape, special_zero=True, name=name)


def _dequant_np(t):
    """GGUF 张量 -> float32 numpy（量化走 gguf.dequantize）。"""
    if t.tensor_type.name in ("F16", "F32"):
        return np.asarray(t.data, dtype=np.float32)
    return gguf.dequantize(t.data, t.tensor_type).astype(np.float32)


def _layer_norm(x, w, b, eps=1e-5, name=""):
    """x:[1,C] 全连接层归一，w,b:[C]（自动 broadcast）。"""
    w = ops.reshape(w, [1, -1], special_zero=True, name=name + "_w_r")
    b = ops.reshape(b, [1, -1], special_zero=True, name=name + "_b_r")
    mean = ops.reduce_mean(x, [1], keep_dims=True, name=name + "_mean")
    xc = ops.subtract(x, mean, name=name + "_xc")
    var = ops.reduce_mean(ops.power(xc, F(2.0, name=name + "_pw")), [1], keep_dims=True, name=name + "_var")
    std = ops.sqrt(ops.add(var, F(eps, name=name + "_eps")), name=name + "_std")
    xn = ops.divide(xc, std, name=name + "_xn")
    return ops.add(ops.multiply(xn, w, name=name + "_mul"), b, name=name + "_out")


def _l2norm(x, axes, eps=1e-6, name=""):
    """x / sqrt(sum(x^2, axes))。"""
    sq = ops.power(x, F(2.0, name=name + "_sq"))
    ss = ops.reduce_sum(sq, axes, keep_dims=True, name=name + "_ss")
    norm = ops.sqrt(ops.add(ss, F(eps, name=name + "_eps")), name=name + "_norm")
    return ops.divide(x, norm, name=name + "_out")


def build_rwkv7_ov(gguf_path, max_layers=None):
    r = gguf.GGUFReader(gguf_path)
    T = {t.name: t for t in r.tensors}

    # ---- 超参 ----
    emb_raw = _dequant_np(T["token_embd.weight"])          # (V, C)
    V, C = emb_raw.shape
    L_full = max(int(n.split(".")[1]) for n in T if n.startswith("blk.")) + 1
    L = min(L_full, max_layers) if max_layers else L_full
    H, N = C // 64, 64
    print(f"[ov] V={V} C={C} L={L}(/{L_full}) H={H} N={N}", flush=True)

    # ---- emb + ln0 预归一化（固化为 f16 常量，Gather 不支持压缩常量）----
    ln0_w = _dequant_np(T["token_embd_norm.weight"]).reshape(1, -1)
    ln0_b = _dequant_np(T["token_embd_norm.bias"]).reshape(1, -1)
    mu = emb_raw.mean(-1, keepdims=True)
    va = emb_raw.var(-1, keepdims=True)
    emb_ln = (emb_raw - mu) / np.sqrt(va + 1e-5) * ln0_w + ln0_b
    emb_const = ops.constant(np.ascontiguousarray(emb_ln.astype(np.float16)), C_DTYPE, name="emb")  # [V,C]

    # ---- 参数 ----
    idx = ops.parameter([1], ov.Type.i64, "idx"); idx.friendly_name = "idx"
    s_att_x = ops.parameter([L, C], C_DTYPE, "s_att_x"); s_att_x.friendly_name = "s_att_x"
    s_kv = ops.parameter([L, H, N, N], C_DTYPE, "s_kv"); s_kv.friendly_name = "s_kv"
    s_ffn = ops.parameter([L, C], C_DTYPE, "s_ffn"); s_ffn.friendly_name = "s_ffn"

    # ---- 权重访问 ----
    def cw(name):
        """压缩权重 matmul 用: 返回解量化子图 [d0,d1]（OV 融合 int4/int8）。

        方向关键: GGUF 物理存储为逻辑转置, gguf.dequantize 真实形状 = reversed(tensor.shape),
        而 build_state 对 matmul 权重做 orient="T" => torch 操作数 W_torch = graw.T。
        故这里把 repack 的最终重塑形状翻转为 reversed(tensor.shape) (= graw 物理形状),
        使 cw == graw; 外层 MatMul 统一 transpose_b=True => x @ cw.T == x @ graw.T == x @ W_torch
        (方阵/非方阵/head 全部 bit-exact 对齐 torch, 见 _cw_vs_bs_test / _orient_test)。
        """
        rep = dict(repack_tensor(T[name]))
        rep["shape"] = list(rep["shape"])[::-1]   # reversed = gguf.dequantize 的物理形状
        return build_compressed_weight(rep, name)

    def cf(name, reshape=None):
        """fp16 常量（向量/affine 小权重）。reshape: 目标形状元组。"""
        arr = _dequant_np(T[name])
        if reshape is not None:
            arr = arr.reshape(reshape)
        return ops.constant(np.ascontiguousarray(arr.astype(np.float16)), C_DTYPE, name=name)

    # ---- embedding ----
    x = ops.gather(emb_const, idx, axis=0, name="x")       # [1,C]

    new_att_x, new_kv, new_ffn = [], [], []
    v_first = None

    for i in range(L):
        b = f"blk.{i}."
        x = ops.reshape(x, [1, C], special_zero=True, name=f"x_{i}")
        xa_in = _layer_norm(x, cf(b + "attn_norm.weight"), cf(b + "attn_norm.bias"), name=f"ln1_{i}")

        # state 取本层: s_att_x[i] -> [1,C]; s_kv[i] -> [1,H,N,N]
        sax = ops.gather(s_att_x, ops.constant([i], np.int64), axis=0, name=f"sax_{i}")
        skv = ops.gather(s_kv, ops.constant([i], np.int64), axis=0, name=f"skv_{i}")     # [1,H,N,N]
        sff = ops.gather(s_ffn, ops.constant([i], np.int64), axis=0, name=f"sff_{i}")
        skv = ops.reshape(skv, [H, N, N], special_zero=True, name=f"skv_r_{i}")

        d = ops.subtract(sax, xa_in, name=f"d_{i}")

        # lerp_fused (6,C) -> x_r/x_w/x_k/x_v/x_a/x_g
        lerp = cf(b + "time_mix_lerp_fused.weight", reshape=(6, C))   # [6,C]
        xr_w = ops.gather(lerp, ops.constant([0], np.int64), axis=0, name=f"xr_w_{i}")
        xw_w = ops.gather(lerp, ops.constant([1], np.int64), axis=0, name=f"xw_w_{i}")
        xk_w = ops.gather(lerp, ops.constant([2], np.int64), axis=0, name=f"xk_w_{i}")
        xv_w = ops.gather(lerp, ops.constant([3], np.int64), axis=0, name=f"xv_w_{i}")
        xa_w = ops.gather(lerp, ops.constant([4], np.int64), axis=0, name=f"xa_w_{i}")
        xg_w = ops.gather(lerp, ops.constant([5], np.int64), axis=0, name=f"xg_w_{i}")

        def mix(inp, w):
            return ops.add(xa_in, ops.multiply(d, w, name=f"mix_{inp}_{i}"), name=f"xm_{inp}_{i}")

        xr = mix("r", xr_w)
        xw = mix("w", xw_w)
        xk = mix("k", xk_w)
        xv = mix("v", xv_w)
        xa = mix("a", xa_w)
        xg = mix("g", xg_w)

        # att matmul（统一 transpose_b=False）
        r = ops.matmul(xr, cw(b + "time_mix_receptance.weight"), transpose_a=False, transpose_b=True, name=f"r_{i}")
        wl = ops.tanh(ops.matmul(xw, cw(b + "time_mix_w1.weight"), transpose_a=False, transpose_b=True, name=f"wl1_{i}"), name=f"wlt_{i}")
        wl = ops.matmul(wl, cw(b + "time_mix_w2.weight"), transpose_a=False, transpose_b=True, name=f"wl2_{i}")
        k = ops.matmul(xk, cw(b + "time_mix_key.weight"), transpose_a=False, transpose_b=True, name=f"k_{i}")
        v = ops.matmul(xv, cw(b + "time_mix_value.weight"), transpose_a=False, transpose_b=True, name=f"v_{i}")
        a = ops.sigmoid(ops.add(cf(b + "time_mix_a0.weight"),
                                ops.matmul(ops.matmul(xa, cw(b + "time_mix_a1.weight"), transpose_a=False, transpose_b=True, name=f"a1_{i}"),
                                           cw(b + "time_mix_a2.weight"), transpose_a=False, transpose_b=True, name=f"a2_{i}"),
                               name=f"a_{i}"))
        g = ops.matmul(ops.sigmoid(ops.matmul(xg, cw(b + "time_mix_g1.weight"), transpose_a=False, transpose_b=True, name=f"g1_{i}"),
                                   name=f"gsig_{i}"),
                       cw(b + "time_mix_g2.weight"), transpose_a=False, transpose_b=True, name=f"g_{i}")

        k_k = cf(b + "time_mix_k_k.weight")
        k_a = cf(b + "time_mix_k_a.weight")
        kk = _l2norm(R(ops.multiply(k, k_k, name=f"kk_in_{i}"), [H, N], f"kk_v_{i}"),
                     axes=[1], name=f"kk_{i}")
        kk = R(kk, [H * N], f"kk_f_{i}")
        k = ops.multiply(k, ops.add(F(1.0), ops.multiply(ops.subtract(a, F(1.0), name=f"a1m_{i}"), k_a, name=f"ak_{i}"), name=f"akm_{i}"), name=f"kdec_{i}")

        if i == 0:
            v_first = v
        else:
            v1mm = ops.matmul(xv, cw(b + "time_mix_v1.weight"), transpose_a=False, transpose_b=True, name=f"v1_{i}")
            v2mm = ops.matmul(v1mm, cw(b + "time_mix_v2.weight"), transpose_a=False, transpose_b=True, name=f"v2_{i}")
            vsig = ops.sigmoid(ops.add(cf(b + "time_mix_v0.weight"), v2mm, name=f"vsig_{i}"), name=f"vsig_a_{i}")
            v = ops.add(v, ops.multiply(ops.subtract(v_first, v, name=f"vv_{i}"), vsig, name=f"vfix_{i}"), name=f"v_after_vfix_{i}")

        decay = R(ops.exp(ops.multiply(F(-0.606531), ops.sigmoid(ops.add(cf(b + "time_mix_w0.weight"), wl, name=f"decay_in_{i}"), name=f"decay_s_{i}"), name=f"decay_e_{i}"),
                            name=f"decay_{i}"), [H, 1, N], f"decay_r_{i}")

        # 递归状态更新
        st = skv                                               # [H,N,N]
        vk = ops.matmul(R(v, [H, N, 1], f"v_v_{i}"),
                        R(k, [H, 1, N], f"k_v_{i}"), transpose_a=False, transpose_b=False, name=f"vk_{i}")   # [H,N,N]
        ab = ops.matmul(R(ops.multiply(F(-1.0), R(kk, [H, N], f"nkk_{i}"), name=f"nkk2_{i}"), [H, N, 1], f"nkk3_{i}"),
                        R(ops.multiply(R(kk, [H, N], f"kka_{i}"), R(a, [H, N], f"ka_{i}"), name=f"kka2_{i}"), [H, 1, N], f"kka3_{i}"),
                        transpose_a=False, transpose_b=False, name=f"ab_{i}")                        # [H,N,N]
        st = ops.add(ops.add(ops.multiply(st, decay, name=f"st_dec_{i}"), ops.matmul(st, ab, transpose_a=False, transpose_b=False, name=f"st_ab_{i}"), name=f"st_ab2_{i}"), vk, name=f"st_new_{i}")

        # per-head ln_x（group norm 等价）
        st_r = ops.matmul(st, R(r, [H, N, 1], f"r_v_{i}"), transpose_a=False, transpose_b=False, name=f"str_{i}")  # [H,N,1]
        _x = ops.convert(ops.reshape(st_r, [1, H, N], special_zero=True, name=f"xrsh_{i}"), ov.Type.f32, name=f"xf32_{i}")
        mu = ops.reduce_mean(_x, [2], keep_dims=True, name=f"xmu_{i}")
        var = ops.reduce_mean(ops.power(ops.subtract(_x, mu, name=f"xc_{i}"), F(2.0, ov.Type.f32), name=f"xpw_{i}"), [2], keep_dims=True, name=f"xvar_{i}")
        y = ops.divide(ops.subtract(_x, mu, name=f"xsub_{i}"),
                       ops.sqrt(ops.add(var, np.float32(64e-5), name=f"xeps_{i}"), name=f"xstd_{i}"), name=f"xn_{i}")
        o = ops.add(ops.multiply(ops.convert(ops.reshape(y, [1, H * N], special_zero=True, name=f"yr_{i}"), C_DTYPE, name=f"yf16_{i}"),
                                 cf(b + "time_mix_ln.weight"), name=f"o_aff_{i}"),
                   cf(b + "time_mix_ln.bias"), name=f"o_{i}")
        r_k = R(cf(b + "time_mix_r_k.weight"), [H, N], f"rk_{i}")
        rr = R(r, [H, N], f"rr_{i}")
        kk2 = R(k, [H, N], f"kk_r_{i}")
        gate = ops.reduce_sum(ops.multiply(ops.multiply(rr, kk2, name=f"rrk_{i}"), r_k, name=f"rrkr_{i}"), [1], keep_dims=True, name=f"gate_{i}")  # [H,1]
        vv = R(v, [H, N], f"vv_r_{i}")
        o = ops.add(o, R(ops.multiply(gate, vv, name=f"gatev_{i}"), [1, H * N], f"gatev_r_{i}"), name=f"o2_{i}")

        x = ops.add(x, ops.matmul(ops.multiply(o, g, name=f"og_{i}"),
                                  cw(b + "time_mix_output.weight"), transpose_a=False, transpose_b=True, name=f"out_mm_{i}"),
                    name=f"x_after_att_{i}")
        new_att_x.append(xa_in)
        new_kv.append(ops.reshape(st, [1, H, N, N], special_zero=True, name=f"kv_out_{i}"))

        # ---- channel mix ----
        xf_in = _layer_norm(x, cf(b + "attn_norm_2.weight"), cf(b + "attn_norm_2.bias"), name=f"ln2_{i}")
        kff = ops.add(xf_in, ops.multiply(ops.subtract(sff, xf_in, name=f"ffd_{i}"), cf(b + "channel_mix_lerp_k.weight"), name=f"ffm_{i}"), name=f"kff_{i}")
        kff = ops.power(ops.relu(ops.matmul(kff, cw(b + "channel_mix_key.weight"), transpose_a=False, transpose_b=True, name=f"ffk_{i}"), name=f"ffrelu_{i}"), F(2.0), name=f"ffsq_{i}")
        x = ops.add(x, ops.matmul(kff, cw(b + "channel_mix_value.weight"), transpose_a=False, transpose_b=True, name=f"ffv_{i}"), name=f"x_after_ffn_{i}")
        new_ffn.append(xf_in)

    x = _layer_norm(x, cf("output_norm.weight"), cf("output_norm.bias"), name="ln_out")
    logits = ops.matmul(x, cw("output.weight"), transpose_a=False, transpose_b=True, name="logits")

    new_att_x_n = ops.concat([ops.reshape(t, [1, C], special_zero=True, name=f"nax_{i}") for i, t in enumerate(new_att_x)], axis=0, name="new_att_x")
    new_kv_n = ops.concat(new_kv, axis=0, name="new_kv")
    new_ffn_n = ops.concat([ops.reshape(t, [1, C], special_zero=True, name=f"nff_{i}") for i, t in enumerate(new_ffn)], axis=0, name="new_ffn")

    model = ov.Model([ops.result(logits, "logits"), ops.result(new_att_x_n, "new_att_x"),
                      ops.result(new_kv_n, "new_kv"), ops.result(new_ffn_n, "new_ffn")],
                     [idx, s_att_x, s_kv, s_ffn], name="rwkv7_ov")
    return model, (V, C, L, H, N)


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--device", default="CPU")
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=16)
    A.add_argument("--out", default=None, help="保存 IR xml 路径（可选）")
    A.add_argument("--threads", type=int, default=4, help="CPU 线程数（降低以压低编译/推理峰值内存，cgroup 8GB 受限时设小）")
    A.add_argument("--layers", type=int, default=0, help="仅构建前 N 层（0=全部）。用于单层/少层验证，编译量远小于 8GB")
    A.add_argument("--no-compile", action="store_true", help="只导出 IR，不编译不推理")
    args = A.parse_args()

    t0 = time.time()
    model, (V, C, L, H, N) = build_rwkv7_ov(args.gguf, max_layers=args.layers or None)
    print(f"[ov] built graph in {time.time()-t0:.1f}s, devices={args.device} threads={args.threads}", flush=True)

    # 先落盘 IR（不依赖编译）：即便后续编译 OOM 也已保住产物，可换机/低内存方式编译
    if args.out:
        ov.save_model(model, args.out, compress_to_fp16=True)
        print(f"[ov] saved IR -> {args.out}")

    if args.no_compile:
        print("[ov] --no-compile: 仅导出 IR，跳过编译与推理。")
        return

    core = ov.Core()
    import gc, subprocess
    gc.collect()
    # 回收 build 阶段读 GGUF 留下的 page cache（计入 cgroup memory.current），压低编译峰值
    try:
        subprocess.run("echo 3 > /proc/sys/vm/drop_caches", shell=True, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        core.set_property("CPU", {ov.properties.inference_num_threads: args.threads,
                                  ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY})
    except Exception as e:
        print(f"[ov] warn: set_property failed ({e}); fallback default threads")
    tc = time.time()
    comp = core.compile_model(model, args.device)
    print(f"[ov] compiled in {time.time()-tc:.1f}s, running ...", flush=True)
    ins = {i.any_name: i for i in comp.inputs}
    req = comp.create_infer_request()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rwkv_tokenizer import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER("rwkv_vocab_v20230424.txt")

    sa = np.zeros((L, C), np.float16)
    sk = np.zeros((L, H, N, N), np.float16)
    sf = np.zeros((L, C), np.float16)
    g = []
    for t in tok.encode(args.prompt) + [None] * args.n:
        if t is None:
            t = int(np.argmax(o[0]))
        req.infer({ins["idx"]: np.array([t], np.int64),
                    ins["s_att_x"]: sa, ins["s_kv"]: sk, ins["s_ffn"]: sf})
        o = np.array(req.get_output_tensor(0).data)
        sa = np.asarray(req.get_output_tensor(1).data).reshape(L, C)
        sk = np.asarray(req.get_output_tensor(2).data).reshape(L, H, N, N)
        sf = np.asarray(req.get_output_tensor(3).data).reshape(L, C)
        if t is not None:
            g.append(t)
    print(f"[ov] RWKV7-OV gen: {tok.decode(g)}")


if __name__ == "__main__":
    main()
