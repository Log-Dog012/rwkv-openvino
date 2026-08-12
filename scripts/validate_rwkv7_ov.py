#!/usr/bin/env python3
"""RWKV7 OV 图 vs numpy 单层参考：逐 token 头对头数值验证（单层编译量 < 8GB）。

为什么用 numpy 参考而非 torch:
  torch 全模型 fp16 权重常驻 ~3GB，与 OV 编译模型共存时推理循环峰值 > 8GB 被 OOM 杀。
  numpy 参考只抽取第 0 层权重(~50MB)，峰值 ~3.5GB，稳过 8GB cgroup。
  参考权重来自 gguf_to_ov.build_state（已验证方向正确），逐元素等价于已验证可生成正确文本的
  rwkv7_torch.py，故 OV 单层输出与 numpy 参考逐元素对齐即证明图正确。

比对对象（均为第 0 层）:
  - new_att_x (xa_in): ln1 输出   -> 验证 emb/ln0/ln1 预处理
  - new_kv    (st)   : wkv7 递归  -> 关键，验证 time_mix + WKV7 内核
  - new_ffn   (xf_in): ln2 输出   -> 验证 ln2 预处理
OV 用 f16，numpy 参考用 f32（权重均由 Q4_K 反量化，与 ov_weight_check 一致），
数值接近 fp16 精度即 PASS。
"""
import sys, os, time
import numpy as np
import openvino as ov

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_ov import build_rwkv7_ov
from rwkv7_ref_np import layer0_weights, block_forward
from rwkv_tokenizer import TRIE_TOKENIZER


def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else "../models/rwkv7-g1i-1.5b-Q4_K_M.gguf"
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The Eiffel Tower is located in the city of"

    # ---------- numpy 单层参考（第 0 层权重）----------
    print("[val] extracting layer-0 numpy reference (Q4_K dequant) ...", flush=True)
    t0 = time.time()
    w, C, H, N = layer0_weights(gguf_path)
    print(f"[val] numpy ref ready in {time.time()-t0:.1f}s  C={C} H={H} N={N}", flush=True)

    # ---------- OV 图（仅第 0 层；numpy 参考也只算第 0 层）----------
    n_layers = 1
    print(f"[val] building OV graph (max_layers={n_layers}) ...", flush=True)
    t0 = time.time()
    ov_model, (V, C2, L1, H2, N2) = build_rwkv7_ov(gguf_path, max_layers=n_layers)
    assert (C, H, N) == (C2, H2, N2), f"dim mismatch {(C,H,N)} vs {(C2,H2,N2)}"
    core = ov.Core()
    try:
        core.set_property("CPU", {ov.properties.inference_num_threads: 4,
                                  ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY})
    except Exception:
        pass
    comp = core.compile_model(ov_model, "CPU")
    ins = {i.any_name: i for i in comp.inputs}
    req = comp.create_infer_request()
    print(f"[val] OV compiled in {time.time()-t0:.1f}s", flush=True)

    tok = TRIE_TOKENIZER("rwkv_vocab_v20230424.txt")
    ids = tok.encode(prompt)
    print(f"[val] prompt({len(ids)} tok): {prompt!r}", flush=True)

    # 状态（零初始化）
    xa_np = np.zeros(C, np.float32)          # 上一 token 的 ln1 输出 (s_att_x)
    sk_np = np.zeros((H, N, N), np.float32)  # s_kv
    sf_np = np.zeros(C, np.float32)          # s_ffn
    sa_o = np.zeros((L1, C), np.float16)
    sk_o = np.zeros((L1, H, N, N), np.float16)
    sf_o = np.zeros((L1, C), np.float16)

    worst = {"att": 0.0, "kv": 0.0, "ffn": 0.0}
    for step, t in enumerate(ids):
        x = w["emb"][t].astype(np.float32)            # emb + ln0（build_state 已预归一）
        xa_ref, st_ref, xf_ref = block_forward(x, xa_np, sk_np, sf_np, w, H, N)
        xa_np, sk_np, sf_np = xa_ref.copy(), st_ref.copy(), xf_ref.copy()

        req.infer({ins["idx"]: np.array([t], np.int64),
                    ins["s_att_x"]: sa_o, ins["s_kv"]: sk_o, ins["s_ffn"]: sf_o})
        na_o = np.array(req.get_output_tensor(1).data).astype(np.float32)   # new_att_x [L1,C]
        nk_o = np.array(req.get_output_tensor(2).data).astype(np.float32)  # new_kv    [L1,H,N,N]
        nf_o = np.array(req.get_output_tensor(3).data).astype(np.float32)  # new_ffn   [L1,C]
        sa_o = na_o.astype(np.float16); sk_o = nk_o.astype(np.float16); sf_o = nf_o.astype(np.float16)

        for li in range(1):   # numpy 参考仅第 0 层，比对 OV 第 0 层
            att = float(np.max(np.abs(xa_ref - na_o[li])))
            kv = float(np.max(np.abs(st_ref - nk_o[li])))
            ffn = float(np.max(np.abs(xf_ref - nf_o[li])))
            worst["att"] = max(worst["att"], att)
            worst["kv"] = max(worst["kv"], kv)
            worst["ffn"] = max(worst["ffn"], ffn)
            print(f"  step {step:2d} layer {li} att_err={att:.3e} kv_err={kv:.3e} ffn_err={ffn:.3e}", flush=True)

    print(f"[val] WORST per-token abs err over {len(ids)} steps, {L1} layer(s):", flush=True)
    print(f"        att(xa_in) = {worst['att']:.3e}", flush=True)
    print(f"        kv (wkv7)  = {worst['kv']:.3e}", flush=True)
    print(f"        ffn(xf_in) = {worst['ffn']:.3e}", flush=True)
    ok = worst["kv"] < 0.1 and worst["att"] < 0.1 and worst["ffn"] < 0.1
    print(f"[val] {'PASS ✅' if ok else 'FAIL ❌'} (阈值 0.1)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
