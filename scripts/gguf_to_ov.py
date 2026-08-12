#!/usr/bin/env python3
"""GGUF -> RWKV7 单步模型 -> OpenVINO IR 转换器（自己写计算图路线）。

回答"GGUF 混合精度能否继承到 IR"：OV 原生（read_model/GenAI）不支持 RWKV7，
本脚本用 gguf 包逐张量反量化 + 自定义映射，把 GGUF 权重注入我们已验证 bit-exact 的
x070 单步模型，再走成熟的 export 路径出 IR。

实测映射规则（rwkv7-g1i-1.5b-Q4_K_M vs 官方 pth，corr=1.0 精确验证）：
  GGUF 张量                       -> 我们的权重（z 为官方 loader 后处理格式）
  token_embd.weight (V,C)          -> emb.weight (V,C) 原样（build 时做 ln0 预归一）
  output.weight (V,C)              -> head.weight 转置 (C,V)
  blk.i.attn_norm(.2)              -> blocks.i.ln1 / ln2
  blk.i.time_mix_ln                -> blocks.i.att.ln_x
  blk.i.time_mix_lerp_fused (6,C)  -> att.x_r/x_w/x_k/x_v/x_a/x_g（顺序即此）
  blk.i.channel_mix_key (4C,C)     -> ffn.key.weight 转置 (C,4C)
  blk.i.channel_mix_value (C,4C)   -> ffn.value.weight 转置 (4C,C)
  blk.i.time_mix_{key,value,receptance,output} (C,C) -> att.* 原样（方阵）
  blk.i.time_mix_{w1,a1,g1,v1} (rank,C) -> att.* 转置 (C,rank)
  blk.i.time_mix_{w2,a2,g2,v2} (C,rank) -> att.* 转置 (rank,C)
  blk.i.time_mix_{w0,a0,k_a,k_k,v0,r_k} (C,) -> att.* 原样

已知缺陷（shoumenchougou GGUF 转换 bug）：time_mix_v1/v2 实际存的是 a1/a2 的值
（rank 96 而非官方 v 混合 rank 64）。本脚本忠实还原 GGUF 编码的语义（v1=GGUF_v1），
即与 llama.cpp 运行时行为一致；如需精确等于官方模型需从 pth 取 v1/v2。

用法:
  python3 scripts/gguf_to_ov.py build   <model.gguf>  [--out z_gguf.pt] [--dtype fp16]
  python3 scripts/gguf_to_ov.py check   <model.gguf>  <ref.pth>  [--prompt "..."]
  python3 scripts/gguf_to_ov.py export  <model.gguf>  [--out outdir] [--n 16]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_torch import RWKV7

_NT = ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")

QUANT_TYPES = set()  # 保留占位（未用）


def gguf_load(gguf_path):
    import gguf
    return gguf.GGUFReader(gguf_path)


def dequant(t):
    """张量 -> float32 numpy（量化张量走 gguf.dequantize）"""
    import gguf
    if t.tensor_type.name == "F32":
        return t.data.astype(np.float32)
    if t.tensor_type.name == "F16":
        return t.data.astype(np.float32)
    return gguf.dequantize(t.data, t.tensor_type).astype(np.float32)


def build_state(reader, dtype):
    """GGUF -> 官方 loader 后处理格式的权重 dict z（逐张量流式，内存可控）。"""
    tensors = {t.name: t for t in reader.tensors}
    emb = dequant(tensors["token_embd.weight"])
    C = int(emb.shape[1])  # (V, C) 物理布局；gguf Tensor.shape 是逻辑维度（转置）勿用
    n_layers = max(int(n.split(".")[1]) for n in tensors if n.startswith("blk.")) + 1
    z = {}

    def put(zkey, name, orient="none"):
        t = tensors[name]
        a = dequant(t)
        if orient == "T":
            a = a.T
        z[zkey] = torch.from_numpy(np.ascontiguousarray(a)).to(dtype)

    # 全局（emb 已反量化，直接用）
    z["emb.weight"] = torch.from_numpy(np.ascontiguousarray(emb)).to(dtype)
    put("blocks.0.ln0.weight", "token_embd_norm.weight")
    put("blocks.0.ln0.bias", "token_embd_norm.bias")
    put("ln_out.weight", "output_norm.weight")
    put("ln_out.bias", "output_norm.bias")
    put("head.weight", "output.weight", orient="T")

    for i in range(n_layers):
        b = f"blk.{i}."
        p = f"blocks.{i}."
        put(p + "ln1.weight", b + "attn_norm.weight")
        put(p + "ln1.bias", b + "attn_norm.bias")
        put(p + "ln2.weight", b + "attn_norm_2.weight")
        put(p + "ln2.bias", b + "attn_norm_2.bias")
        # lerp_fused (6,C) -> x_r/x_w/x_k/x_v/x_a/x_g
        lf = tensors[b + "time_mix_lerp_fused.weight"]
        lfa = dequant(lf).reshape(6, -1)  # channel-major
        for j, k in enumerate(["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]):
            z[p + f"att.{k}"] = torch.from_numpy(np.ascontiguousarray(lfa[j])).to(dtype)
        # 方阵（Linear 风格，带 .weight）。注意：官方 loader 对 _T_KEYS（含 key/value/
        # receptance/output）做了 .t()，故 GGUF 存的原始布局需再转置才与官方后处理对齐。
        # 实测：GGUF time_mix_key.weight 与 pth 原始 corr=0.9973，与 pth 转置 corr≈0 → 必须转置。
        for gg, zz in [("key", "key"), ("value", "value"), ("receptance", "receptance"), ("output", "output")]:
            put(p + f"att.{zz}.weight", b + f"time_mix_{gg}.weight", orient="T")
        # LoRA 对（无 .weight 后缀；GGUF 存转置，需转回）
        for gg, zz in [("w1", "w1"), ("w2", "w2"), ("a1", "a1"), ("a2", "a2"),
                       ("g1", "g1"), ("g2", "g2"), ("v1", "v1"), ("v2", "v2")]:
            put(p + f"att.{zz}", b + f"time_mix_{gg}.weight", orient="T")
        # 向量（无 .weight 后缀）
        for gg, zz in [("w0", "w0"), ("a0", "a0"), ("k_a", "k_a"), ("k_k", "k_k"), ("v0", "v0"), ("r_k", "r_k")]:
            put(p + f"att.{zz}", b + f"time_mix_{gg}.weight")
        # ln_x（GroupNorm 风格，带 .weight/.bias）
        put(p + "att.ln_x.weight", b + "time_mix_ln.weight")
        put(p + "att.ln_x.bias", b + "time_mix_ln.bias")
        # FFN
        put(p + "ffn.x_k", b + "channel_mix_lerp_k.weight")
        put(p + "ffn.key.weight", b + "channel_mix_key.weight", orient="T")
        put(p + "ffn.value.weight", b + "channel_mix_value.weight", orient="T")

    n_head = int(C // 64)  # H = C / head_size（RWKV 标准关系，head_size=64）
    return z, n_head, 64


def repair_v12(z, ref_pth):
    """GGUF 修复：shoumenchougou 的 RWKV7-G1i GGUF 中 time_mix_v1/v2 被错写成 rank 96
    （与 a/w 混秩混淆），且值为垃圾；官方 pth 的 v1/v2 为 rank 64 且语义正确。
    本函数从参考 pth（bf16，mmap 懒加载，只取 v1/v2 两键）覆盖 GGUF 构建的 z。
    返回修复层数。"""
    raw = torch.load(ref_pth, map_location="cpu", mmap=True)
    n_layers = max(int(k.split(".")[1]) for k in z if k.startswith("blocks.")) + 1
    n = 0
    for i in range(n_layers):
        v1 = raw[f"blocks.{i}.att.v1"].to(torch.float32).numpy()
        v2 = raw[f"blocks.{i}.att.v2"].to(torch.float32).numpy()
        z[f"blocks.{i}.att.v1"] = torch.from_numpy(np.ascontiguousarray(v1)).to(z["emb.weight"].dtype)
        z[f"blocks.{i}.att.v2"] = torch.from_numpy(np.ascontiguousarray(v2)).to(z["emb.weight"].dtype)
        n += 1
    del raw
    return n


def main():
    A = argparse.ArgumentParser()
    sub = A.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("gguf"); p_build.add_argument("--out", default="temp/z_gguf.pt"); p_build.add_argument("--dtype", default="fp16")
    p_build.add_argument("--repair-v12", default=None, help="参考 pth，用其正确 rank-64 的 v1/v2 覆盖 GGUF 坏张量")

    p_check = sub.add_parser("check")
    p_check.add_argument("gguf"); p_check.add_argument("ref_pth"); p_check.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    p_check.add_argument("--n", type=int, default=12)
    p_check.add_argument("--repair-v12", default=None, help="参考 pth，用其正确 rank-64 的 v1/v2 覆盖 GGUF 坏张量")

    p_exp = sub.add_parser("export")
    p_exp.add_argument("gguf"); p_exp.add_argument("--outdir", default="models"); p_exp.add_argument("--n", type=int, default=16)
    p_exp.add_argument("--repair-v12", default=None, help="参考 pth，用其正确 rank-64 的 v1/v2 覆盖 GGUF 坏张量")

    A = A.parse_args()
    cmd = A.cmd
    torch.set_num_threads(8)  # fp16 CPU matmul 用默认线程数偶发死锁，固定 8 线程
    dtype = torch.float16 if getattr(A, "dtype", "fp16") == "fp16" else torch.float32

    reader = gguf_load(A.gguf)
    t0 = time.time()
    z, n_head, head_size = build_state(reader, dtype)
    n_layers = 1 + max(int(k.split('.')[1]) for k in z if k.startswith('blocks.'))
    print(f"[gguf] built {len(z)} tensors, C={z['emb.weight'].shape[1]}, L={n_layers}, H={n_head}, in {time.time()-t0:.1f}s", flush=True)

    if getattr(A, "repair_v12", None):
        n = repair_v12(z, A.repair_v12)
        print(f"[gguf] repair_v12: overwrote v1/v2 from {A.repair_v12} for {n} layers", flush=True)

    if cmd == "build":
        torch.save(z, A.out)
        print(f"[gguf] saved -> {A.out}")

    elif cmd == "check":
        # 顺序验证：GGUF 模块与 pth 模块不同时驻留内存（1.5B 各 3GB，8GB cgroup 放不下两个）
        import json
        import sys as _s
        _s.path.insert(0, "scripts")
        from rwkv_tokenizer import TRIE_TOKENIZER
        tok = TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
        ids = tok.encode(A.prompt)
        tmp = "temp/gguf_check_ref.json"

        def run(m, tag):
            with torch.no_grad():
                m(*[torch.tensor([0], dtype=torch.int64), *m.zero_state()])  # warmup
                sa, sk, sf = m.zero_state()
                gen, first = [], None
                last = None
                for t in list(ids) + [None] * A.n:
                    idx = torch.tensor([t if t is not None else int(torch.argmax(last))], dtype=torch.int64)
                    lg, sa, sk, sf = m(idx, sa, sk, sf)
                    if first is None:
                        first = lg
                    last = lg
                    if t is None:
                        gen.append(int(torch.argmax(lg)))
            print(f"[gguf] {tag} gen: {tok.decode(gen)}", flush=True)
            return first

        m_g = RWKV7.from_state(z, n_head, head_size, dtype=dtype).eval()
        first_g = run(m_g, "GGUF ")
        json.dump({"first": first_g.numpy().tolist()}, open(tmp, "w"))
        del m_g
        import gc; gc.collect()
        m_r = RWKV7(A.ref_pth, dtype=dtype).eval()
        first_r = run(m_r, "pth  ")
        ref = json.load(open(tmp))["first"]
        d = (first_r - torch.tensor(ref)).abs().max().item()
        print(f"[gguf] first-step logits max|Δ| (GGUF-Q4 vs official pth): {d:.3f}")
        os.remove(tmp)

    elif cmd == "export":
        m = RWKV7.from_state(z, n_head, head_size, dtype=dtype).eval()
        print(f"[gguf] exporting OV IR ({dtype}) ...", flush=True)
        import openvino as ov
        t1 = time.time()
        idx = torch.zeros((1,), dtype=torch.int64)
        sa, sk, sf = m.zero_state()
        ov_model = ov.convert_model(m, example_input=(idx, sa, sk, sf))
        print(f"[gguf] convert_model in {time.time()-t1:.1f}s", flush=True)
        os.makedirs(A.outdir, exist_ok=True)
        base = os.path.join(A.outdir, os.path.basename(A.gguf).replace(".gguf", f"_step_{'fp16' if dtype==torch.float16 else 'fp32'}"))
        ov.save_model(ov_model, base + ".xml", compress_to_fp16=(dtype == torch.float16))
        print(f"[gguf] saved -> {base}.xml/.bin")
        # 生成验证
        from rwkv_tokenizer import TRIE_TOKENIZER
        tok = TRIE_TOKENIZER("scripts/rwkv_vocab_v20230424.txt")
        comp = ov.Core().compile_model(base + ".xml", "CPU")
        req = comp.create_infer_request()
        ins = [i.any_name for i in comp.inputs]
        sa, sk, sf = [x.numpy() for x in m.zero_state()]
        g = []
        for t in tok.encode("The Eiffel Tower is located in the city of") + [None] * A.n:
            if t is None:
                t = int(np.argmax(o[0]))
            o = req.infer({ins[0]: np.array([t], np.int64), ins[1]: sa, ins[2]: sk, ins[3]: sf})
            sa = np.asarray(o[1]).reshape(*sa.shape); sk = np.asarray(o[2]).reshape(*sk.shape); sf = np.asarray(o[3]).reshape(*sf.shape)
            if t is not None:
                g.append(t)
        print(f"[gguf] OV gen: {tok.decode(g)}")


if __name__ == "__main__":
    main()
