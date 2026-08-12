"""
GGUF K-quant -> OpenVINO 原生压缩权重 repacker
==============================================
目标: 把 GGUF 的 K-quant 混合精度 raw 字节, 重排成 OpenVINO 可识别/可融合的
      原生压缩权重形态 (u4/u8 Constant + f16 scales + [zp]), 推理时走
      Convert -> Subtract(zp) -> Multiply(scale) 子图, 被 OV 的
      CompressedWeightsBlock 模式匹配并融合为原生 int4/int8 matmul.

为何不直接 dequant 成 fp16:
  - GGUF 已是混合精度(K-quant), dequant 会丢失结构并膨胀 3x (7B 14GB, 超 8GB cgroup)
  - 真·继承 = 以原始低比特 + scale 进 IR, 体积≈GGUF (7B≈4.6GB), 进 8GB

参考:
  - llama.cpp ggml-openvino-extra.cpp: ggml_openvino_get_extracted_layout()
    Q4_K->u4 per32 asym, Q6_K->u8 per16 sym, Q8_0->u8 per32 sym
  - openvino compressed_weights_block.cpp: Constant(u4/u8)->Convert->Sub(zp)->Mul(scale)
"""
import numpy as np
import gguf

QK_K = 256  # gguf.constants.QK_K


def _read_f16(arr, off):
    return np.frombuffer(arr[off:off + 2], dtype=np.float16).astype(np.float32)[0]


def get_scale_min_q4k(scales: np.ndarray):
    """与 gguf.quants.Q4_K.get_scale_min 完全一致. 输入 [nblk,12] uint8 -> (sc[8], mn[8])."""
    s = scales.reshape(-1, 3, 4).astype(np.uint8)
    d, m, m_d = np.split(s, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape(-1, 8).astype(np.float32), mn.reshape(-1, 8).astype(np.float32)


def unpack_q4_k_block(block: np.ndarray):
    """输入 144 字节 Q4_K block, 返回 (codes[256]uint8 0-15, scales[8]abs, mins[8]abs)."""
    d = _read_f16(block, 0)
    dmin = _read_f16(block, 2)
    sc, mn = get_scale_min_q4k(block[4:16].reshape(1, 12))
    sc, mn = sc[0], mn[0]
    scales = (d * sc).astype(np.float32)   # 绝对 scale [8]
    mins = (dmin * mn).astype(np.float32)  # 绝对 zp [8]
    qs = block[16:144]
    # 与 gguf 完全一致: reshape(-1,1,32) >> [0,4] & 0x0F -> (8,32) 逐行铺开
    qs_nib = (qs.reshape(4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 2, 1)) & 0x0F
    codes = qs_nib.reshape(-1).astype(np.uint8)  # 256 值, 顺序与 gguf 同
    return codes, scales, mins


def repack_q4_k(raw: np.ndarray, n_elements: int):
    """GGUF Q4_K 整张量 raw -> (u4codes[n], scales[n/32], zp[n/32]). 已展开为绝对 scale/min."""
    nblk = n_elements // QK_K
    raw = raw[: nblk * 144].reshape(nblk, 144)
    codes = np.empty(nblk * QK_K, dtype=np.uint8)
    scales = np.empty(nblk * 8, dtype=np.float32)
    mins = np.empty(nblk * 8, dtype=np.float32)
    for b in range(nblk):
        c, s, m = unpack_q4_k_block(raw[b])
        codes[b * QK_K:(b + 1) * QK_K] = c
        scales[b * 8:(b + 1) * 8] = s
        mins[b * 8:(b + 1) * 8] = m
    return codes, scales, mins


def repack_tensor(tensor) -> dict:
    """返回该 GGUF tensor 在 OV 中的表示. type 字段: u4/u8/f16/f32."""
    tt = tensor.tensor_type.name
    raw = np.frombuffer(tensor.data, dtype=np.uint8)
    n = int(tensor.n_elements)
    shape = list(tensor.shape)
    if tt == "Q4_K":
        codes, scales, mins = repack_q4_k(raw, n)
        return {"type": "u4", "gguf_type": "Q4_K", "codes": codes,
                "scales": scales, "zp": mins, "group": 32, "shape": shape}
    if tt == "Q8_0":
        r = repack_q8_0(raw, n, shape); r["gguf_type"] = "Q8_0"; return r
    if tt == "Q6_K":
        r = _repack_q6_k(raw, n, shape); r["gguf_type"] = "Q6_K"; return r
    if tt in ("F16",):
        return {"type": "f16", "gguf_type": "F16", "data": raw.view(np.float16),
                "shape": shape}
    if tt in ("F32",):
        return {"type": "f32", "gguf_type": "F32", "data": raw.view(np.float32),
                "shape": shape}
    raise NotImplementedError(f"未实现的 GGUF 类型: {tt}")


def _decode_q6_k_block(blk):
    """单 block(210B)解码 -> (codes[256]int8, scales[16]f32). 严格对照 gguf Q6_K.dequantize_blocks."""
    ql = blk[:128].reshape(2, 64)      # 128B -> (2,64), 与 gguf reshape(-1,1,64) 一致
    qh = blk[128:192].reshape(2, 32)    # 64B -> (2,32)
    sc8 = blk[192:208].astype(np.int8).astype(np.float32)   # int8, 无偏移
    d = _read_f16(blk, 208)
    # 低4位: (4,64) -> (4,1,64) >> [0,4] & 0x0F -> (4,2,64) -> reshape(8,32)
    ql4 = (ql.reshape(2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 2, 1)) & 0x0F
    ql4 = ql4.reshape(8, 32)
    # 高2位: (2,32) -> (2,1,32) >> [0,2,4,6] & 0x03 -> (2,4,32) -> reshape(8,32)
    qh2 = (qh.reshape(2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 4, 1)) & 0x03
    qh2 = qh2.reshape(8, 32)
    # q = (ql | (qh<<4)) - 32 -> (8,32), 再 reshape(16,16) 对齐 16 组 scale (与 gguf 同序)
    q = ((ql4 | (qh2 << 4)).astype(np.int8) - np.int8(32)).reshape(16, 16).reshape(-1)
    return q, (d * sc8).astype(np.float32)


def _repack_q6_k(raw, n, shape):
    """GGUF Q6_K (210B/block, 256 元素) -> OV i8 codes + per16 scale (对称)."""
    nblk = n // QK_K
    raw = raw[: nblk * 210].reshape(nblk, 210)
    codes = np.empty(nblk * QK_K, dtype=np.int8)
    scales = np.empty(nblk * 16, dtype=np.float32)
    for b in range(nblk):
        c, s = _decode_q6_k_block(raw[b])
        codes[b * QK_K:(b + 1) * QK_K] = c
        scales[b * 16:(b + 1) * 16] = s
    return {"type": "i8", "codes": codes, "scales": scales, "zp": None,
            "group": 16, "shape": shape}


def repack_q8_0(raw, n, shape):
    """GGUF Q8_0 (34B/block, 32 元素) -> OV i8 codes + per32 scale (对称)."""
    nblk = n // 32
    raw = raw[: nblk * 34].reshape(nblk, 34)
    d = raw[:, 0:2].view(np.float16).astype(np.float32)  # [nblk]
    qs = raw[:, 2:].astype(np.int8)                       # [nblk,32]
    codes = qs.reshape(-1)
    scales = np.repeat(d, 32)
    return {"type": "i8", "codes": codes, "scales": scales, "zp": None,
            "group": 32, "shape": shape}


def validate_against_gguf(gguf_path: str, max_tensors: int = 6, cap: int = 1_000_000):
    """逐个张量(仅前 cap 元素)对照 gguf.dequantize, 验证 repack 数值 bit-exact.
    cap 限制避免整张大张量(如 268M 的 token_embd)撑爆内存."""
    import gguf as _gguf
    r = _gguf.GGUFReader(gguf_path)
    from collections import Counter
    stats = Counter()
    print(f"=== 验证 {gguf_path} 的混合精度 repack (每张量前 {cap} 元素) ===")
    checked = 0
    for t in r.tensors:
        tt = t.tensor_type.name
        if tt not in ("Q4_K", "Q6_K", "Q8_0"):
            continue
        n = int(t.n_elements)
        rep = repack_tensor(t)
        # 用 gguf 原生解量化作 ground truth, 仅前若干 block (避免整张大张量撑爆内存)
        if tt == "Q4_K":
            blk = 256
            nb = min(n // blk, max(1, cap // blk))
            nkeep = nb * blk
            gt = _gguf.quants.Q4_K.dequantize_blocks(
                np.frombuffer(t.data, dtype=np.uint8).reshape(-1, 144)[:nb].copy()).reshape(-1)
            code = rep["codes"][:nkeep].astype(np.float32)
            sc = np.repeat(rep["scales"], 32)[:nkeep]
            zp = np.repeat(rep["zp"], 32)[:nkeep]
            mine = code * sc - zp
        elif tt == "Q8_0":
            blk = 32
            nb = min(n // blk, max(1, cap // blk))
            nkeep = nb * blk
            gt = _gguf.quants.Q8_0.dequantize_blocks(
                np.frombuffer(t.data, dtype=np.uint8).reshape(-1, 34)[:nb].copy()).reshape(-1)
            code = rep["codes"][:nkeep].astype(np.float32)
            sc = np.repeat(rep["scales"], 32)[:nkeep]
            mine = code * sc
        elif tt == "Q6_K":
            blk = 256
            nb = min(n // blk, max(1, cap // blk))
            nkeep = nb * blk
            gt = _gguf.quants.Q6_K.dequantize_blocks(
                np.frombuffer(t.data, dtype=np.uint8).reshape(-1, 210)[:nb].copy()).reshape(-1)
            code = rep["codes"][:nkeep].astype(np.float32)
            sc = np.repeat(rep["scales"], 16)[:nkeep]
            mine = code * sc
        gt = gt.reshape(-1).astype(np.float64)
        mine = mine.astype(np.float64)
        err = float(np.max(np.abs(gt - mine)))
        rel = err / (float(np.max(np.abs(gt))) + 1e-12)
        ok = err < 1e-2
        stats[tt] += 1
        print(f"  [{tt:5s}] {t.name:38s} elems={nkeep} max|err|={err:.2e} rel={rel:.2e} {'OK' if ok else 'FAIL'}")
        checked += 1
        if checked >= max_tensors:
            break
    print("=== 抽查张量数:", checked, dict(stats), "===")
    return stats


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "models/rwkv7-g1i-7.2b-Q4_K_M.gguf"
    validate_against_gguf(p)
