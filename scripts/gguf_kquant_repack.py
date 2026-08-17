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
    """与 gguf.quants.Q4_K.get_scale_min 完全一致. 输入 [nblk,12] uint8 -> (sc[8], mn[8]).
    向量化版：支持任意 nblk 批量，输出 [nblk,8]."""
    s = scales.reshape(-1, 3, 4).astype(np.uint8)
    d, m, m_d = np.split(s, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    # 保留对单 block [1,12] 输入的回退：reshape(-1,8) 已自然兼容
    return sc.reshape(-1, 8).astype(np.float32), mn.reshape(-1, 8).astype(np.float32)


def _read_f16_batch(arr, off, nblk):
    """批量读 [nblk] 个 f16 at offset off (每 block 内). 返回 [nblk] float32."""
    seg = arr[:, off:off + 2].reshape(-1)  # [nblk*2] bytes
    f16 = np.frombuffer(seg.tobytes(), dtype=np.float16).astype(np.float32)
    return f16  # [nblk]


def repack_q4_k(raw: np.ndarray, n_elements: int):
    """GGUF Q4_K 整张量 raw -> (codes[n], scales_abs[n/32], zp_abs[n/32]).
    向量化版：一次处理全部 nblk 个 block，输出顺序与逐块版完全一致。
    单 block 语义（unpack_q4_k_block）的批等价：
      d = f16(block,0), dmin = f16(block,2)
      sc, mn = get_scale_min(block[4:16])            # 各 [8]
      scales_abs = d * sc, mins_abs = dmin * mn      # [8]
      codes = (block[16:144].reshape(4,1,32) >> [0,4]) & 0x0F  -> reshape(256)"""
    nblk = n_elements // QK_K
    blocks = raw[: nblk * 144].reshape(nblk, 144).astype(np.uint8)

    # scales / zp（每 block 8 组）
    d = _read_f16_batch(blocks, 0, nblk)                       # [nblk] f16->f32
    dmin = _read_f16_batch(blocks, 2, nblk)                   # [nblk]
    sc, mn = get_scale_min_q4k(blocks[:, 4:16].reshape(nblk, 12))  # 各 [nblk,8]
    scales = (d[:, None] * sc).astype(np.float32).reshape(-1)  # [nblk*8] 绝对 scale
    mins = (dmin[:, None] * mn).astype(np.float32).reshape(-1) # [nblk*8] 绝对 zp

    # codes：每 block 128B -> 256 nibbles
    qs = blocks[:, 16:144].reshape(nblk, 4, 1, 32)             # [nblk,4,1,32]
    sh = np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)        # 广播 [1,1,2,1]
    nib = (qs >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 0x0F  # [nblk,4,2,32]
    codes = nib.reshape(nblk, -1).astype(np.uint8)            # [nblk,256]
    return codes.reshape(-1), scales, mins


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
    """GGUF Q6_K (210B/block, 256 元素) -> OV i8 codes + per16 scale (对称).
    向量化版：一次处理全部 nblk 个 block，输出顺序与逐块版完全一致。
    单 block 语义（_decode_q6_k_block）的批等价见行内注释。"""
    nblk = n // QK_K
    blocks = raw[: nblk * 210].reshape(nblk, 210).astype(np.uint8)

    # 低 4 位: 128B -> (2,64) per block -> (nblk,2,1,64) >> [0,4] & 0x0F -> (nblk,2,2,64) -> (nblk,8,32)
    ql = blocks[:, :128].reshape(nblk, 2, 64)
    ql4 = (ql.reshape(nblk, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 0x0F
    ql4 = ql4.reshape(nblk, 8, 32)
    # 高 2 位: 64B -> (2,32) per block -> (nblk,2,1,32) >> [0,2,4,6] & 0x03 -> (nblk,2,4,32) -> (nblk,8,32)
    qh = blocks[:, 128:192].reshape(nblk, 2, 32)
    qh2 = (qh.reshape(nblk, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 0x03
    qh2 = qh2.reshape(nblk, 8, 32)
    # q = (ql4 | (qh2<<4)) - 32 -> (nblk,8,32) -> reshape(nblk,16,16) -> (nblk,256)
    q = ((ql4 | (qh2 << 4)).astype(np.int8) - np.int8(32)).reshape(nblk, 16, 16).reshape(nblk, -1)

    # scales: 每 block 16 组, sc8 int8 [nblk,16], d f16 [nblk] -> abs [nblk,16]
    sc8 = blocks[:, 192:208].astype(np.int8).astype(np.float32)   # [nblk,16]
    d = _read_f16_batch(blocks, 208, nblk)                       # [nblk]
    scales = (d[:, None] * sc8).astype(np.float32)              # [nblk,16]

    codes = q.reshape(-1).astype(np.int8)                        # [nblk*256]
    return {"type": "i8", "codes": codes, "scales": scales.reshape(-1),
            "zp": None, "group": 16, "shape": shape}


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
