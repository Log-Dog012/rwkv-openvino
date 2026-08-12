#!/usr/bin/env python3
"""RWKV7 OV 分层执行器（8GB cgroup 友好）。

全 24 层单图编译会超 8GB（实测 RC=137 OOM：int4→fp16 展开 + 图表示开销）。
本脚本把每层构建为独立单层中图（压缩权重继承 K-quant，reversed 形状 + transpose_b=True
已验证 bit-exact 对齐 torch），由 Python 循环逐层推进 3 个状态张量，端到端生成文本。

内存账（1.5B Q4）:
  24 个单层中图: 每层压缩权 ~40MB + 编译期 fp16 展开 ~80MB ≈ 1.9GB
  emb 表(共享输入, ln0 预归一) [V,C] f16 ≈ 268MB; 状态 ~30MB
  峰值 ≈ 2.3GB < 8GB  ✅

用法:
  python3 rwkv7_ov_layerwise.py <model.gguf> [--prompt "..."] [--n 16] [--threads 4]
"""
import argparse, os, sys, time, gc, mmap
import numpy as np
import gguf
import openvino as ov
from openvino import opset13 as ops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_ov import F, R, _layer_norm, _l2norm, _dequant_np, C_DTYPE
from gguf_kquant_repack import repack_tensor
from gguf_to_ov_compressed import build_compressed_weight


def _cw(T, name):
    """压缩权重 matmul 用（reversed 形状 + 外层 transpose_b=True 对齐 torch）。"""
    rep = dict(repack_tensor(T[name]))
    rep["shape"] = list(rep["shape"])[::-1]   # reversed = gguf.dequantize 的物理形状
    return build_compressed_weight(rep, name)


def build_layer(r, li, V, C, with_head=False):
    """构建第 li 层的单层中图。emb 表以共享输入传入（避免每层 268MB 常量拷贝）。
    T 为共享的张量字典(由调用方用单个 GGUFReader 构建一次), 避免重复 mmap 8.4GB 文件。

    输入: idx[1] i64, emb_table[V,C] f16(已 ln0 预归一), s_att_x[1,C], s_kv[1,H,N,N], s_ffn[1,C]
    输出: (logits[1,V] 仅 with_head) , new_att_x[1,C], new_kv[1,H,N,N], new_ffn[1,C]
    """
    H, N = C // 64, 64
    T = {t.name: t for t in r.tensors}
    b = f"blk.{li}."

    idx = ops.parameter([1], ov.Type.i64, "idx"); idx.friendly_name = "idx"
    emb_table = ops.parameter([V, C], C_DTYPE, "emb_table"); emb_table.friendly_name = "emb_table"
    s_att_x = ops.parameter([1, C], C_DTYPE, "s_att_x"); s_att_x.friendly_name = "s_att_x"
    s_kv = ops.parameter([1, H, N, N], C_DTYPE, "s_kv"); s_kv.friendly_name = "s_kv"
    s_ffn = ops.parameter([1, C], C_DTYPE, "s_ffn"); s_ffn.friendly_name = "s_ffn"

    def cf(name, reshape=None):
        arr = _dequant_np(T[name])
        if reshape is not None:
            arr = arr.reshape(reshape)
        return ops.constant(np.ascontiguousarray(arr.astype(np.float16)), C_DTYPE, name=name)

    x = ops.gather(emb_table, idx, axis=0, name="x")                       # [1,C]
    xa_in = _layer_norm(x, cf(b + "attn_norm.weight"), cf(b + "attn_norm.bias"), name="ln1")

    d = ops.subtract(s_att_x, xa_in, name="d")

    lerp = cf(b + "time_mix_lerp_fused.weight", reshape=(6, C))
    xr_w = ops.gather(lerp, ops.constant([0], np.int64), axis=0, name="xr_w")
    xw_w = ops.gather(lerp, ops.constant([1], np.int64), axis=0, name="xw_w")
    xk_w = ops.gather(lerp, ops.constant([2], np.int64), axis=0, name="xk_w")
    xv_w = ops.gather(lerp, ops.constant([3], np.int64), axis=0, name="xv_w")
    xa_w = ops.gather(lerp, ops.constant([4], np.int64), axis=0, name="xa_w")
    xg_w = ops.gather(lerp, ops.constant([5], np.int64), axis=0, name="xg_w")

    def mix(inp, w):
        return ops.add(xa_in, ops.multiply(d, w, name=f"mix_{inp}"), name=f"xm_{inp}")

    xr = mix("r", xr_w); xw = mix("w", xw_w); xk = mix("k", xk_w)
    xv = mix("v", xv_w); xa = mix("a", xa_w); xg = mix("g", xg_w)

    r_w = ops.matmul(xr, _cw(T, b + "time_mix_receptance.weight"), transpose_a=False, transpose_b=True, name="r")
    wl = ops.tanh(ops.matmul(xw, _cw(T, b + "time_mix_w1.weight"), transpose_a=False, transpose_b=True, name="wl1"), name="wlt")
    wl = ops.matmul(wl, _cw(T, b + "time_mix_w2.weight"), transpose_a=False, transpose_b=True, name="wl2")
    k = ops.matmul(xk, _cw(T, b + "time_mix_key.weight"), transpose_a=False, transpose_b=True, name="k")
    v = ops.matmul(xv, _cw(T, b + "time_mix_value.weight"), transpose_a=False, transpose_b=True, name="v")
    a = ops.sigmoid(ops.add(cf(b + "time_mix_a0.weight"),
                            ops.matmul(ops.matmul(xa, _cw(T, b + "time_mix_a1.weight"), transpose_a=False, transpose_b=True, name="a1"),
                                       _cw(T, b + "time_mix_a2.weight"), transpose_a=False, transpose_b=True, name="a2"),
                            name="a"))
    g = ops.matmul(ops.sigmoid(ops.matmul(xg, _cw(T, b + "time_mix_g1.weight"), transpose_a=False, transpose_b=True, name="g1"), name="gsig"),
                   _cw(T, b + "time_mix_g2.weight"), transpose_a=False, transpose_b=True, name="g")

    k_k = cf(b + "time_mix_k_k.weight"); k_a = cf(b + "time_mix_k_a.weight")
    kk = _l2norm(R(ops.multiply(k, k_k, name="kk_in"), [H, N], "kk_v"), axes=[1], name="kk")
    kk = R(kk, [H * N], "kk_f")
    k = ops.multiply(k, ops.add(F(1.0), ops.multiply(ops.subtract(a, F(1.0), name="a1m"), k_a, name="ak"), name="akm"), name="kdec")

    decay = R(ops.exp(ops.multiply(F(-0.606531), ops.sigmoid(ops.add(cf(b + "time_mix_w0.weight"), wl, name="decay_s"), name="decay_e"), name="decay")),
              [H, 1, N], "decay_r")

    st = s_kv
    vk = ops.matmul(R(v, [H, N, 1], "v_v"), R(k, [H, 1, N], "k_v"), transpose_a=False, transpose_b=False, name="vk")
    ab = ops.matmul(R(ops.multiply(F(-1.0), R(kk, [H, N], "nkk"), name="nkk2"), [H, N, 1], "nkk3"),
                    R(ops.multiply(R(kk, [H, N], "kka"), R(a, [H, N], "ka"), name="kka2"), [H, 1, N], "kka3"),
                    transpose_a=False, transpose_b=False, name="ab")
    st = ops.add(ops.add(ops.multiply(st, decay, name="st_dec"), ops.matmul(st, ab, transpose_a=False, transpose_b=False, name="st_ab"), name="st_ab2"), vk, name="st_new")

    st_r = ops.matmul(st, R(r_w, [H, N, 1], "r_v"), transpose_a=False, transpose_b=False, name="str")
    _x = ops.convert(ops.reshape(st_r, [1, H, N], special_zero=True, name="xrsh"), ov.Type.f32, name="xf32")
    mu = ops.reduce_mean(_x, [2], keep_dims=True, name="xmu")
    var = ops.reduce_mean(ops.power(ops.subtract(_x, mu, name="xc"), F(2.0, ov.Type.f32), name="xpw"), [2], keep_dims=True, name="xvar")
    y = ops.divide(ops.subtract(_x, mu, name="xsub"), ops.sqrt(ops.add(var, np.float32(64e-5), name="xeps"), name="xstd"), name="xn")
    o = ops.add(ops.multiply(ops.convert(ops.reshape(y, [1, H * N], special_zero=True, name="yr"), C_DTYPE, name="yf16"),
                             cf(b + "time_mix_ln.weight"), name="o_aff"), cf(b + "time_mix_ln.bias"), name="o")
    r_k = R(cf(b + "time_mix_r_k.weight"), [H, N], "rk")
    rr = R(r_w, [H, N], "rr"); kk2 = R(k, [H, N], "kk_r"); vv = R(v, [H, N], "vv_r")
    gate = ops.reduce_sum(ops.multiply(ops.multiply(rr, kk2, name="rrk"), r_k, name="rrkr"), [1], keep_dims=True, name="gate")
    o = ops.add(o, R(ops.multiply(gate, vv, name="gatev"), [1, H * N], "gatev_r"), name="o2")

    x = ops.add(x, ops.matmul(ops.multiply(o, g, name="og"), _cw(T, b + "time_mix_output.weight"), transpose_a=False, transpose_b=True, name="out_mm"), name="x_after_att")
    new_att_x = xa_in
    new_kv = ops.reshape(st, [1, H, N, N], special_zero=True, name="kv_out")

    xf_in = _layer_norm(x, cf(b + "attn_norm_2.weight"), cf(b + "attn_norm_2.bias"), name="ln2")
    kff = ops.add(xf_in, ops.multiply(ops.subtract(s_ffn, xf_in, name="ffd"), cf(b + "channel_mix_lerp_k.weight"), name="ffm"), name="kff")
    kff = ops.power(ops.relu(ops.matmul(kff, _cw(T, b + "channel_mix_key.weight"), transpose_a=False, transpose_b=True, name="ffk"), name="ffrelu"), F(2.0), name="ffsq")
    x = ops.add(x, ops.matmul(kff, _cw(T, b + "channel_mix_value.weight"), transpose_a=False, transpose_b=True, name="ffv"), name="x_after_ffn")
    new_ffn = xf_in

    outputs = [ops.result(new_att_x, "new_att_x"), ops.result(new_kv, "new_kv"), ops.result(new_ffn, "new_ffn")]
    if with_head:
        logits = ops.matmul(x, _cw(T, "output.weight"), transpose_a=False, transpose_b=True, name="logits")
        outputs = [ops.result(logits, "logits")] + outputs

    model = ov.Model(outputs, [idx, emb_table, s_att_x, s_kv, s_ffn], name=f"rwkv7_layer_{li}")
    return model, (V, C, H, N)


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=16)
    A.add_argument("--threads", type=int, default=4)
    A.add_argument("--layers", type=int, default=0, help="仅前 N 层(0=全部)")
    args = A.parse_args()

    t0 = time.time()
    r = gguf.GGUFReader(args.gguf)
    T = {t.name: t for t in r.tensors}
    L_full = max(int(n.split(".")[1]) for n in T if n.startswith("blk.")) + 1
    L = L_full if args.layers in (0, None) else min(args.layers, L_full)
    # 一次性反量化 emb（取 C/V 并做 ln0 预归一）, 避免构建每层时重复反量化
    emb_raw = _dequant_np(T["token_embd.weight"])
    V, C = emb_raw.shape
    H, N = C // 64, 64
    print(f"[lw] C={C} L={L}(/{L_full}) H={H} N={N}", flush=True)

    # ln0 预归一 emb 表（一次性, 共享输入）
    ln0_w = _dequant_np(T["token_embd_norm.weight"]).reshape(1, -1)
    ln0_b = _dequant_np(T["token_embd_norm.bias"]).reshape(1, -1)
    emb_ln = ((emb_raw - emb_raw.mean(-1, keepdims=True)) / np.sqrt(emb_raw.var(-1, keepdims=True) + 1e-5) * ln0_w + ln0_b)
    emb_ln = np.ascontiguousarray(emb_ln.astype(np.float16))
    del emb_raw
    print(f"[lw] emb_ln ready {emb_ln.shape} in {time.time()-t0:.1f}s", flush=True)

    # 构建并编译 24 个单层中图。GGUF 文件 8.4GB > 8GB cgroup, 且沙箱禁止 drop_caches:
    # 用单个共享 reader, 每编译完一层就 madvise(DONTNEED) 释放该层 fault 进来的文件页
    # (等价于 drop_caches, 无需 root), 使任意时刻 GGUF 常驻页仅 ~单层张量(~350MB),
    # 24 个编译模型(~2GB)+emb(~268MB) 稳过 8GB。
    def release_gguf_pages():
        try:
            r.data.base.madvise(mmap.MADV_DONTNEED)
        except Exception:
            pass

    core = ov.Core()
    try:
        core.set_property("CPU", {ov.properties.inference_num_threads: args.threads,
                                  ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY})
    except Exception:
        pass
    comps, reqs = [], []
    for li in range(L):
        m, _ = build_layer(r, li, V, C, with_head=(li == L - 1))
        comp = core.compile_model(m, "CPU")
        comps.append(comp); reqs.append(comp.create_infer_request())
        del m
        gc.collect()
        release_gguf_pages()   # 释放本层 fault 的 GGUF 文件页
        if (li + 1) % 8 == 0 or li == L - 1:
            print(f"[lw] compiled {li+1}/{L} layers in {time.time()-t0:.1f}s", flush=True)
    del r
    gc.collect()
    try:
        pass
    except Exception:
        pass

    tok = TRIE_TOKENIZER_safe("rwkv_vocab_v20230424.txt")
    ids = tok.encode(args.prompt)
    print(f"[lw] prompt({len(ids)} tok): {args.prompt!r}", flush=True)

    s_att = np.zeros((L, C), np.float16)
    s_kv = np.zeros((L, H, N, N), np.float16)
    s_ffn = np.zeros((L, C), np.float16)

    def run_token(t):
        """跑完所有层(逐层推进状态), 返回末层 logits[V]。"""
        for li in range(L):
            reqs[li].infer({"idx": np.array([t], np.int64), "emb_table": emb_ln,
                            "s_att_x": s_att[li:li + 1], "s_kv": s_kv[li:li + 1], "s_ffn": s_ffn[li:li + 1]})
            if li == L - 1:
                logits = np.array(reqs[li].get_output_tensor(0).data)
                att = np.array(reqs[li].get_output_tensor(1).data)
                kv = np.array(reqs[li].get_output_tensor(2).data)
                ffn = np.array(reqs[li].get_output_tensor(3).data)
            else:
                att = np.array(reqs[li].get_output_tensor(0).data)
                kv = np.array(reqs[li].get_output_tensor(1).data)
                ffn = np.array(reqs[li].get_output_tensor(2).data)
            s_att[li] = att.astype(np.float16); s_kv[li] = kv.astype(np.float16); s_ffn[li] = ffn.astype(np.float16)
        return logits

    gen = []
    tb = time.time()
    for t in ids:
        last_logits = run_token(t)
    for _ in range(args.n):
        t = int(np.argmax(last_logits))
        gen.append(t)
        last_logits = run_token(t)
    print(f"[lw] generated {args.n} tokens in {time.time()-tb:.1f}s", flush=True)
    print(f"[lw] RWKV7-OV(layerwise) gen: {tok.decode(gen)}", flush=True)


def TRIE_TOKENIZER_safe(vocab):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rwkv_tokenizer import TRIE_TOKENIZER
    return TRIE_TOKENIZER(vocab)


if __name__ == "__main__":
    main()
