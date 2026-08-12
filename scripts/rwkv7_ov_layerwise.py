#!/usr/bin/env python3
"""RWKV7 OV 分层分块执行器（8GB cgroup 友好 + 正确的层间组合）。

相对 rwkv7_ov.py 单图的关键差异:
1. 层间组合必须正确: 第 li 层的输入 x = 第 li-1 层的输出, 不能重新 embed。
   分块图 [lo,hi): lo==0 从 emb(已 ln0)+LN1 开始; lo>0 直接以 x_in 参数为输入。
   (旧版逐层脚本每层都 gather(emb_table,idx) 是组合 bug, 本版已修)
2. 每层递归状态 (att_x, kv, ffn) 跨 token 保持, 分块图内 K 层状态打包为
   [K,C]/[K,H,N,N]/[K,C] 三个输入/输出。
3. 8GB 墙: 实测本 OV 源码版 CPU 插件每编译一层常驻 ~0.45GB(疑似未融合 int4,
   权重解包成 f32 执行缓冲), 24 层单图 >8GB 必 OOM。分块后任意时刻只常驻
   一个 chunk(K=8 → ~3.6GB) + 状态(≈30MB) + emb(268MB) + 运行时, 峰值 ~6.5GB。
4. 分块图 IR 落盘(temp/ir_chunks), 生成阶段每 token 重载 chunk 时只需 load+compile,
   不需要重新 build/dequant。代价: 每个生成 token 要重载全部分块(编译 ~3s/层)。
5. 压缩权重方向: reversed(repack shape) + MatMul transpose_b=True (bit-exact 对齐 torch)。

用法:
  python3 rwkv7_ov_layerwise.py <model.gguf> [--prompt "..."] [--n 8] [--threads 4] [--chunk 8]
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


def _layer_body(x, s_att_x, s_kv, s_ffn, T, cf, li, C, v_first=None):
    """单层 RWKV7 前向（x[1,C] → x_out[1,C]，状态张量逐层独立）。
    v_first: RWKV7 g1 的跨层量——layer0 的原始 v, 同 token 内传给所有层;
    layer1+ 用它做 vfix: v += (v_first-v)*sigmoid(v0+xv@v1@v2)。
    li==0 时返回 (..., vf_out=raw v); 否则 vf_out=None。
    名字统一加 _l{li} 后缀, 避免分块图内多层节点重名。"""
    H, N = C // 64, 64
    b = f"blk.{li}."
    s = f"l{li}"

    xa_in = _layer_norm(x, cf(b + "attn_norm.weight"), cf(b + "attn_norm.bias"), name=f"ln1_{s}")
    d = ops.subtract(s_att_x, xa_in, name=f"d_{s}")

    lerp = cf(b + "time_mix_lerp_fused.weight", reshape=(6, C))
    xr_w = ops.gather(lerp, ops.constant([0], np.int64), axis=0, name=f"xr_w_{s}")
    xw_w = ops.gather(lerp, ops.constant([1], np.int64), axis=0, name=f"xw_w_{s}")
    xk_w = ops.gather(lerp, ops.constant([2], np.int64), axis=0, name=f"xk_w_{s}")
    xv_w = ops.gather(lerp, ops.constant([3], np.int64), axis=0, name=f"xv_w_{s}")
    xa_w = ops.gather(lerp, ops.constant([4], np.int64), axis=0, name=f"xa_w_{s}")
    xg_w = ops.gather(lerp, ops.constant([5], np.int64), axis=0, name=f"xg_w_{s}")

    def mix(inp, w):
        return ops.add(xa_in, ops.multiply(d, w, name=f"mix_{inp}_{s}"), name=f"xm_{inp}_{s}")

    xr = mix("r", xr_w); xw = mix("w", xw_w); xk = mix("k", xk_w)
    xv = mix("v", xv_w); xa = mix("a", xa_w); xg = mix("g", xg_w)

    r_w = ops.matmul(xr, _cw(T, b + "time_mix_receptance.weight"), transpose_a=False, transpose_b=True, name=f"r_{s}")
    wl = ops.tanh(ops.matmul(xw, _cw(T, b + "time_mix_w1.weight"), transpose_a=False, transpose_b=True, name=f"wl1_{s}"), name=f"wlt_{s}")
    wl = ops.matmul(wl, _cw(T, b + "time_mix_w2.weight"), transpose_a=False, transpose_b=True, name=f"wl2_{s}")
    k = ops.matmul(xk, _cw(T, b + "time_mix_key.weight"), transpose_a=False, transpose_b=True, name=f"k_{s}")
    v = ops.matmul(xv, _cw(T, b + "time_mix_value.weight"), transpose_a=False, transpose_b=True, name=f"v_{s}")
    a = ops.sigmoid(ops.add(cf(b + "time_mix_a0.weight"),
                            ops.matmul(ops.matmul(xa, _cw(T, b + "time_mix_a1.weight"), transpose_a=False, transpose_b=True, name=f"a1_{s}"),
                                       _cw(T, b + "time_mix_a2.weight"), transpose_a=False, transpose_b=True, name=f"a2_{s}"),
                            name=f"a_{s}"))
    g = ops.matmul(ops.sigmoid(ops.matmul(xg, _cw(T, b + "time_mix_g1.weight"), transpose_a=False, transpose_b=True, name=f"g1_{s}"), name=f"gsig_{s}"),
                   _cw(T, b + "time_mix_g2.weight"), transpose_a=False, transpose_b=True, name=f"g_{s}")

    k_k = cf(b + "time_mix_k_k.weight"); k_a = cf(b + "time_mix_k_a.weight")
    kk = _l2norm(R(ops.multiply(k, k_k, name=f"kk_in_{s}"), [H, N], f"kk_v_{s}"), axes=[1], name=f"kk_{s}")
    kk = R(kk, [H * N], f"kk_f_{s}")
    k = ops.multiply(k, ops.add(F(1.0), ops.multiply(ops.subtract(a, F(1.0), name=f"a1m_{s}"), k_a, name=f"ak_{s}"), name=f"akm_{s}"), name=f"kdec_{s}")

    # v_first / vfix (RWKV7 g1): layer0 的 v 即 v_first; layer1+ v += (v_first-v)*sigmoid(v0+xv@v1@v2)
    vf_out = None
    if li == 0:
        vf_out = v
    else:
        v1mm = ops.matmul(xv, _cw(T, b + "time_mix_v1.weight"), transpose_a=False, transpose_b=True, name=f"v1_{s}")
        v2mm = ops.matmul(v1mm, _cw(T, b + "time_mix_v2.weight"), transpose_a=False, transpose_b=True, name=f"v2_{s}")
        vsig = ops.sigmoid(ops.add(cf(b + "time_mix_v0.weight"), v2mm, name=f"vsig_{s}"), name=f"vsig_a_{s}")
        v = ops.add(v, ops.multiply(ops.subtract(v_first, v, name=f"vv_{s}"), vsig, name=f"vfix_{s}"), name=f"v_after_vfix_{s}")

    decay = R(ops.exp(ops.multiply(F(-0.606531), ops.sigmoid(ops.add(cf(b + "time_mix_w0.weight"), wl, name=f"decay_s_{s}"), name=f"decay_e_{s}"), name=f"decay_{s}")),
              [H, 1, N], f"decay_r_{s}")

    st = s_kv
    vk = ops.matmul(R(v, [H, N, 1], f"v_v_{s}"), R(k, [H, 1, N], f"k_v_{s}"), transpose_a=False, transpose_b=False, name=f"vk_{s}")
    ab = ops.matmul(R(ops.multiply(F(-1.0), R(kk, [H, N], f"nkk_{s}"), name=f"nkk2_{s}"), [H, N, 1], f"nkk3_{s}"),
                    R(ops.multiply(R(kk, [H, N], f"kka_{s}"), R(a, [H, N], f"ka_{s}"), name=f"kka2_{s}"), [H, 1, N], f"kka3_{s}"),
                    transpose_a=False, transpose_b=False, name=f"ab_{s}")
    st = ops.add(ops.add(ops.multiply(st, decay, name=f"st_dec_{s}"), ops.matmul(st, ab, transpose_a=False, transpose_b=False, name=f"st_ab_{s}"), name=f"st_ab2_{s}"), vk, name=f"st_new_{s}")

    st_r = ops.matmul(st, R(r_w, [H, N, 1], f"r_v_{s}"), transpose_a=False, transpose_b=False, name=f"str_{s}")
    _x = ops.convert(ops.reshape(st_r, [1, H, N], special_zero=True, name=f"xrsh_{s}"), ov.Type.f32, name=f"xf32_{s}")
    mu = ops.reduce_mean(_x, [2], keep_dims=True, name=f"xmu_{s}")
    var = ops.reduce_mean(ops.power(ops.subtract(_x, mu, name=f"xc_{s}"), F(2.0, ov.Type.f32), name=f"xpw_{s}"), [2], keep_dims=True, name=f"xvar_{s}")
    y = ops.divide(ops.subtract(_x, mu, name=f"xsub_{s}"), ops.sqrt(ops.add(var, np.float32(64e-5), name=f"xeps_{s}"), name=f"xstd_{s}"), name=f"xn_{s}")
    o = ops.add(ops.multiply(ops.convert(ops.reshape(y, [1, H * N], special_zero=True, name=f"yr_{s}"), C_DTYPE, name=f"yf16_{s}"),
                             cf(b + "time_mix_ln.weight"), name=f"o_aff_{s}"), cf(b + "time_mix_ln.bias"), name=f"o_{s}")
    r_k = R(cf(b + "time_mix_r_k.weight"), [H, N], f"rk_{s}")
    rr = R(r_w, [H, N], f"rr_{s}"); kk2 = R(k, [H, N], f"kk_r_{s}"); vv = R(v, [H, N], f"vv_r_{s}")
    gate = ops.reduce_sum(ops.multiply(ops.multiply(rr, kk2, name=f"rrk_{s}"), r_k, name=f"rrkr_{s}"), [1], keep_dims=True, name=f"gate_{s}")
    o = ops.add(o, R(ops.multiply(gate, vv, name=f"gatev_{s}"), [1, H * N], f"gatev_r_{s}"), name=f"o2_{s}")

    x = ops.add(x, ops.matmul(ops.multiply(o, g, name=f"og_{s}"), _cw(T, b + "time_mix_output.weight"), transpose_a=False, transpose_b=True, name=f"out_mm_{s}"), name=f"x_after_att_{s}")
    new_att_x = xa_in
    new_kv = ops.reshape(st, [1, H, N, N], special_zero=True, name=f"kv_out_{s}")

    xf_in = _layer_norm(x, cf(b + "attn_norm_2.weight"), cf(b + "attn_norm_2.bias"), name=f"ln2_{s}")
    kff = ops.add(xf_in, ops.multiply(ops.subtract(s_ffn, xf_in, name=f"ffd_{s}"), cf(b + "channel_mix_lerp_k.weight"), name=f"ffm_{s}"), name=f"kff_{s}")
    kff = ops.power(ops.relu(ops.matmul(kff, _cw(T, b + "channel_mix_key.weight"), transpose_a=False, transpose_b=True, name=f"ffk_{s}"), name=f"ffrelu_{s}"), F(2.0), name=f"ffsq_{s}")
    x = ops.add(x, ops.matmul(kff, _cw(T, b + "channel_mix_value.weight"), transpose_a=False, transpose_b=True, name=f"ffv_{s}"), name=f"x_after_ffn_{s}")
    new_ffn = xf_in
    return x, new_att_x, new_kv, new_ffn, vf_out


def build_chunk(r, lo, hi, V, C, with_head):
    """构建 [lo,hi) 层的分块中图。
    lo==0: 输入 idx[1]i64 + emb_table[V,C]f16(已 ln0 预归一) → gather 得到 x。
    lo>0 : 输入 x_in[1,C]f16 (上一层块输出的激活)。
    状态输入/输出: s_att[K,C], s_kv[K,H,N,N], s_ffn[K,C] (K=hi-lo 层打包)。
    输出: new_att/new_kv/new_ffn + x_out[1,C] (+ logits[1,V] 若 with_head)。"""
    K = hi - lo
    H, N = C // 64, 64
    T = {t.name: t for t in r.tensors}

    def cf(name, reshape=None):
        arr = _dequant_np(T[name])
        if reshape is not None:
            arr = arr.reshape(reshape)
        return ops.constant(np.ascontiguousarray(arr.astype(np.float16)), C_DTYPE, name=name)

    params = []
    if lo == 0:
        idx = ops.parameter([1], ov.Type.i64, "idx"); idx.friendly_name = "idx"
        emb_table = ops.parameter([V, C], C_DTYPE, "emb_table"); emb_table.friendly_name = "emb_table"
        x = ops.gather(emb_table, idx, axis=0, name="x_emb")                    # [1,C] 已 ln0
        params += [idx, emb_table]
        v_first_in = None
    else:
        x = ops.parameter([1, C], C_DTYPE, "x_in"); x.friendly_name = "x_in"
        params += [x]
        v_first_in = ops.parameter([1, C], C_DTYPE, "v_first"); v_first_in.friendly_name = "v_first"
        params += [v_first_in]

    s_att = ops.parameter([K, C], C_DTYPE, "s_att"); s_att.friendly_name = "s_att"
    s_kv = ops.parameter([K, H, N, N], C_DTYPE, "s_kv"); s_kv.friendly_name = "s_kv"
    s_ffn = ops.parameter([K, C], C_DTYPE, "s_ffn"); s_ffn.friendly_name = "s_ffn"
    params += [s_att, s_kv, s_ffn]

    att_outs, kv_outs, ffn_outs = [], [], []
    vf = v_first_in
    for li in range(lo, hi):
        j = li - lo
        sa = ops.slice(s_att, [j], [j + 1], [1], axes=[0], name=f"sa_{li}")
        sk = ops.slice(s_kv, [j], [j + 1], [1], axes=[0], name=f"sk_{li}")
        sf = ops.slice(s_ffn, [j], [j + 1], [1], axes=[0], name=f"sf_{li}")
        x, na, nk, nf, vf_out = _layer_body(x, sa, sk, sf, T, cf, li, C, v_first=vf)
        if vf_out is not None:
            vf = vf_out          # layer0 产出的 v_first, 供本 chunk 内后续层 + 输出
        att_outs.append(na); kv_outs.append(nk); ffn_outs.append(nf)
    vf_final = vf                # lo==0: layer0 的 v_first; lo>0: 输入 pass-through

    new_att = ops.concat(att_outs, axis=0, name="new_att")
    new_kv = ops.concat(kv_outs, axis=0, name="new_kv")
    new_ffn = ops.concat(ffn_outs, axis=0, name="new_ffn")
    outputs = [ops.result(new_att, "new_att"), ops.result(new_kv, "new_kv"),
               ops.result(new_ffn, "new_ffn"), ops.result(x, "x_out"),
               ops.result(vf_final, "v_first_out")]
    if with_head:
        # torch: 最后一层 x 先过 ln_out 归一, 再乘 head 权重
        x_ln = _layer_norm(x, cf("output_norm.weight"), cf("output_norm.bias"), name="ln_out")
        logits = ops.matmul(x_ln, _cw(T, "output.weight"), transpose_a=False, transpose_b=True, name="logits")
        outputs.insert(0, ops.result(logits, "logits"))

    model = ov.Model(outputs, params, name=f"rwkv7_chunk_{lo}_{hi}")
    return model, (V, C, H, N)


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=8)
    A.add_argument("--threads", type=int, default=4)
    A.add_argument("--chunk", type=int, default=8, help="每分块层数(常驻内存 ~0.45GB/层, 8→~6.5GB 峰值)")
    A.add_argument("--ir-dir", default="/workspace/rwkv-openvino/temp/ir_chunks")
    A.add_argument("--export-only", action="store_true",
                   help="仅构建并落盘各 chunk IR(不编译不生成), 用于交付/超大模型导出")
    args = A.parse_args()

    t0 = time.time()
    r = gguf.GGUFReader(args.gguf)
    T = {t.name: t for t in r.tensors}
    L_full = max(int(n.split(".")[1]) for n in T if n.startswith("blk.")) + 1
    L = L_full
    emb_raw = _dequant_np(T["token_embd.weight"])
    V, C = emb_raw.shape
    H, N = C // 64, 64
    print(f"[cw] C={C} L={L} H={H} N={N} chunk={args.chunk}", flush=True)

    # emb 表一次性 ln0 预归一 (共享输入, 仅块0用)
    ln0_w = _dequant_np(T["token_embd_norm.weight"]).reshape(1, -1)
    ln0_b = _dequant_np(T["token_embd_norm.bias"]).reshape(1, -1)
    emb_ln = ((emb_raw - emb_raw.mean(-1, keepdims=True)) / np.sqrt(emb_raw.var(-1, keepdims=True) + 1e-5) * ln0_w + ln0_b)
    emb_ln = np.ascontiguousarray(emb_ln.astype(np.float16))
    del emb_raw
    print(f"[cw] emb_ln ready {emb_ln.shape} in {time.time()-t0:.1f}s", flush=True)

    core = ov.Core()
    try:
        core.set_property("CPU", {ov.properties.inference_num_threads: args.threads,
                                  ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY})
    except Exception:
        pass

    def free_gguf_pages():
        # mmap 级: 释放当前进程 fault 的 GGUF 页
        try:
            r.data.base.madvise(mmap.MADV_DONTNEED)
        except Exception:
            pass
        # 文件级: 丢弃 cgroup 页缓存里该 GGUF 的所有文件页(旧进程残留也会清掉), 等效 drop_caches, 无需 root
        try:
            fd = os.open(args.gguf, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass

    chunks = [(i, min(i + args.chunk, L)) for i in range(0, L, args.chunk)]
    os.makedirs(args.ir_dir, exist_ok=True)

    if args.export_only:
        # 仅构建+保存各 chunk IR（不编译不生成）——超大模型交付用
        free_gguf_pages()   # 先清掉旧进程残留的 GGUF 页缓存, 避免 save_model 撞 8GB
        for ci, (lo, hi) in enumerate(chunks):
            path = f"{args.ir_dir}/chunk{lo}_{hi}.xml"
            if os.path.exists(path):
                print(f"[cw] export {ci+1}/{len(chunks)} chunk{lo}_{hi} exists, skip", flush=True)
                continue
            m, _ = build_chunk(r, lo, hi, V, C, with_head=(hi == L))
            ov.save_model(m, path)
            del m; gc.collect(); free_gguf_pages()
            print(f"[cw] export {ci+1}/{len(chunks)} chunk{lo}_{hi} saved in {time.time()-t0:.1f}s", flush=True)
        del r
        print(f"[cw] export-only done: {len(chunks)} chunk IRs -> {args.ir_dir}", flush=True)
        return

    def build_or_load(ci):
        """chunk ci: 有 IR 文件则直接 compile(快), 否则 build+save+compile。"""
        lo, hi = chunks[ci]
        path = f"{args.ir_dir}/chunk{lo}_{hi}.xml"
        if os.path.exists(path):
            return core.compile_model(path, "CPU")
        m, _ = build_chunk(r, lo, hi, V, C, with_head=(hi == L))
        ov.save_model(m, path)
        del m; gc.collect()
        return core.compile_model(path, "CPU")

    # 每层递归状态 [L,C]/[L,H,N,N]/[L,C] 跨 token 保持
    states = [np.zeros((L, C), np.float16),
              np.zeros((L, H, N, N), np.float16),
              np.zeros((L, C), np.float16)]

    def run_chunk(comp, ci, tok, x_in, vf_in):
        """跑一个 token 经过 chunk ci, 更新该 chunk 各层状态。返回 (x_out, vf_out, logits|None)。"""
        lo, hi = chunks[ci]
        feed = {}
        if lo == 0:
            feed["idx"] = np.array([tok], np.int64)
            feed["emb_table"] = emb_ln
        else:
            feed["x_in"] = np.ascontiguousarray(np.asarray(x_in, np.float16).reshape(1, C))
            feed["v_first"] = np.ascontiguousarray(np.asarray(vf_in, np.float16).reshape(1, C))
        feed["s_att"] = states[0][lo:hi]
        feed["s_kv"] = states[1][lo:hi]
        feed["s_ffn"] = states[2][lo:hi]
        req = comp.create_infer_request()
        req.infer(feed)
        # 输出顺序: 有 head [logits,new_att,new_kv,new_ffn,x_out,v_first_out];
        #          无 head [new_att,new_kv,new_ffn,x_out,v_first_out]
        off = 1 if hi == L else 0
        states[0][lo:hi] = np.array(req.get_output_tensor(off).data)
        states[1][lo:hi] = np.array(req.get_output_tensor(off + 1).data)
        states[2][lo:hi] = np.array(req.get_output_tensor(off + 2).data)
        x_out = np.array(req.get_output_tensor(off + 3).data)
        vf_out = np.array(req.get_output_tensor(off + 4).data)
        lg = np.array(req.get_output_tensor(0).data) if off == 1 else None
        del req
        return x_out, vf_out, lg

    tok = TRIE_TOKENIZER_safe("rwkv_vocab_v20230424.txt")
    ids = tok.encode(args.prompt)
    print(f"[cw] prompt({len(ids)} tok): {args.prompt!r}", flush=True)

    # ---------- prompt 扫描: 每个 chunk 处理全部 prompt token, 逐块推进 (x 与 v_first 同步线程) ----------
    last_logits = None
    stream = list(ids)
    vf_stream = None
    for ci in range(len(chunks)):
        comp = build_or_load(ci)
        new_stream, new_vf = [], []
        for i, x_in in enumerate(stream):
            tokid = x_in if ci == 0 else None
            vf_in = None if ci == 0 else vf_stream[i]
            x_out, vf_out, lg = run_chunk(comp, ci, tokid, x_in if ci > 0 else None, vf_in)
            new_stream.append(x_out); new_vf.append(vf_out)
            if lg is not None:
                last_logits = lg
        del comp; gc.collect(); free_gguf_pages()
        print(f"[cw] prompt sweep chunk {ci} ({chunks[ci][0]}-{chunks[ci][1]-1}) done, mem="
              f"{open('/sys/fs/cgroup/memory.current').read().strip()[:6]}B", flush=True)
        stream, vf_stream = new_stream, new_vf
    print(f"[cw] prompt sweep done in {time.time()-t0:.1f}s", flush=True)

    # ---------- 生成: 每 token 重载全部分块(内存限制, 编译 ~3s/层) ----------
    gen = []
    tb = time.time()
    for g in range(args.n):
        t = int(np.argmax(last_logits))
        gen.append(t)
        x_in, vf_in = None, None
        for ci in range(len(chunks)):
            comp = build_or_load(ci)
            tokid = t if ci == 0 else None
            x_out, vf_out, lg = run_chunk(comp, ci, tokid, x_in if ci > 0 else None, vf_in if ci > 0 else None)
            x_in, vf_in = x_out, vf_out
            if lg is not None:
                last_logits = lg
            del comp; gc.collect()
        free_gguf_pages()
        if (g + 1) % 2 == 0 or g == args.n - 1:
            print(f"[cw] gen {g+1}/{args.n} tok in {time.time()-tb:.1f}s: {tok.decode(gen)}", flush=True)
    print(f"[cw] generated {args.n} tokens in {time.time()-tb:.1f}s", flush=True)
    print(f"[cw] gen ids: {gen}", flush=True)
    print(f"[cw] RWKV7-OV(chunked) gen: {tok.decode(gen)!r}", flush=True)


def TRIE_TOKENIZER_safe(vocab):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rwkv_tokenizer import TRIE_TOKENIZER
    return TRIE_TOKENIZER(vocab)


if __name__ == "__main__":
    main()
