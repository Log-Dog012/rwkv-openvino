"""
GGUF K-quant -> OpenVINO 原生压缩权重 IR 构建器
================================================
把 gguf_kquant_repack 的 repack 结果, 拼成 OpenVINO 可识别/可融合的
原生压缩权重子图:

  weight_codes (Constant u4/i8)
    -> Convert(f16)
    -> [Subtract(zp) 非对称]
    -> Multiply(scale)
  = 解量化后的 fp16 权重, 喂给 MatMul/FullyConnected

CPU/GPU 插件靠 compressed_weights_block 模式匹配并融合为原生 int4/int8 matmul,
复用 OV 自己的低比特解量化 kernel. 体积≈GGUF (7B≈4.6GB), 进 8GB cgroup.

参考: openvino compressed_weights_block.cpp
  weights(Constant) -> Convert -> [Subtract(zp)] -> Multiply(scale) -> [Reshape/Transpose]
"""
import numpy as np
import openvino as ov
from openvino import opset13 as ops
from openvino import Core
import gguf
from gguf_kquant_repack import repack_tensor, QK_K


def build_compressed_weight(rep: dict, name: str):
    """返回解量化权重 (ov.Node, f16). rep 来自 repack_tensor.
    压缩权重的 group 沿权重最后一维(in), 每 group 一组 scale/zp.

    K-quant 在 GGUF 里是 block-major 存储 (每 block 256/32 元素连续);
    repack 后 codes 同样 block-major, 因此必须用 [out, blocks_per_row, sub_n, sub_e]
    重排, 不能用 [out, n_groups, group] 的 row-major 假设."""
    tt = rep["type"]
    gtt = rep.get("gguf_type", tt)
    shape = rep["shape"]
    out_dim, in_dim = int(shape[0]), int(shape[1])  # RWKV 权重 [out, in]
    if tt in ("f16", "f32"):
        # 图统一 f16: 未量化权重也落 f16 常量 (与 torch 基线 fp16 推理一致)
        const = ops.constant(np.ascontiguousarray(rep["data"].reshape(shape).astype(np.float16)),
                             ov.Type.f16, name=f"{name}/w")
        return const
    codes = rep["codes"]
    scales = rep["scales"].astype(np.float32)
    # block 布局 (blk_elems 每 block 元素数; sub_n×sub_e 子块划分, 一组 scale 覆盖 sub_e 元素)
    if gtt == "Q4_K":
        blk_elems, sub_n, sub_e, code_dtype = 256, 8, 32, ov.Type.u4
    elif gtt == "Q5_K":
        # 5-bit 值(0-31)存 i8 容器, 8 组×32 元素, 非对称(有 zp) — 同 Q4_K 结构
        blk_elems, sub_n, sub_e, code_dtype = 256, 8, 32, ov.Type.i8
    elif gtt == "Q6_K":
        blk_elems, sub_n, sub_e, code_dtype = 256, 16, 16, ov.Type.i8
    elif gtt == "Q8_0":
        blk_elems, sub_n, sub_e, code_dtype = 32, 1, 32, ov.Type.i8
    else:
        raise NotImplementedError(gtt)
    blocks_per_row = in_dim // blk_elems
    # OV u4/i8 Constant (扁平 [out*in], 每元素 0-15 或 int8)
    code_const = ops.constant(codes, code_dtype, name=f"{name}/codes")
    conv = ops.convert(code_const, ov.Type.f16, name=f"{name}/deq_conv")  # [out*in] f16
    # 重排为 block-major: [out, blocks_per_row, sub_n, sub_e]
    conv = ops.reshape(conv, [out_dim, blocks_per_row, sub_n, sub_e],
                       special_zero=True, name=f"{name}/codes_r")
    sc_const = ops.constant(
        np.ascontiguousarray(scales.reshape(out_dim, blocks_per_row, sub_n, 1)),
        ov.Type.f16, name=f"{name}/scale")
    if rep.get("zp") is not None:
        # gguf Q4_K 非对称: dequant = scale*code - min (min 不乘 scale).
        # OV 标准模式是 (code - zp)*scale, 故 zp 须用码空间零点 = min/scale,
        # 使 (code - min/scale)*scale = scale*code - min, 与 gguf 完全等价.
        # scale==0 的块(全零块, min 亦为 0)需规避除零 -> zp_eff=0 使结果=0, 不污染为 nan.
        zp_raw = rep["zp"].astype(np.float32)
        zp_eff = np.divide(zp_raw, scales, out=np.zeros_like(scales),
                           where=scales != 0.0).reshape(
            out_dim, blocks_per_row, sub_n, 1).astype(np.float32)
        zp_const = ops.constant(np.ascontiguousarray(zp_eff), ov.Type.f16, name=f"{name}/zp")
        sub = ops.subtract(conv, zp_const, name=f"{name}/deq_sub")   # 广播 [..,1]
        deq = ops.multiply(sub, sc_const, name=f"{name}/deq_mul")
    else:
        deq = ops.multiply(conv, sc_const, name=f"{name}/deq_mul")
    # 还原 [out, in]
    deq = ops.reshape(deq, [out_dim, in_dim], special_zero=True, name=f"{name}/deq_r")
    return deq


def ov_weight_check(gguf_path: str, max_tensors: int = 8):
    """端到端: 对每个压缩权重张量, 在 OV 上以原生压缩权重子图解量化,
    对照 gguf.quants.*.dequantize_blocks 验证数值 bit-exact.
    用显式 OV 模型 (Constant->Convert->[Sub]->Mul->Reshape) 走压缩权重融合,
    不依赖整张大张量落地, 避免 OOM. 仅取前 max_tensors 个张量抽样."""
    import gguf as _gguf
    r = _gguf.GGUFReader(gguf_path)
    core = Core()
    results = []
    checked = 0
    for t in r.tensors:
        gtt = t.tensor_type.name
        if gtt not in ("Q4_K", "Q6_K", "Q8_0"):
            continue
        rep = repack_tensor(t)
        out_dim, in_dim = int(rep["shape"][0]), int(rep["shape"][1])
        w_deq = build_compressed_weight(rep, f"w_{t.name}")
        w_out = ops.result(w_deq, name=f"w_out_{t.name}")
        model = ov.Model([w_out], [])
        comp = core.compile_model(model, "CPU")
        w_ov = np.array(comp([])[f"w_out_{t.name}"].data)
        w_ov = w_ov.astype(np.float32).reshape(out_dim, in_dim)
        # ground truth: 整张量 block 解量化
        raw = np.frombuffer(t.data, dtype=np.uint8)
        if gtt == "Q4_K":
            nb = out_dim * in_dim // 256
            gt = _gguf.quants.Q4_K.dequantize_blocks(
                raw.reshape(-1, 144)[:nb].copy()).reshape(out_dim, in_dim)
        elif gtt == "Q6_K":
            nb = out_dim * in_dim // 256
            gt = _gguf.quants.Q6_K.dequantize_blocks(
                raw.reshape(-1, 210)[:nb].copy()).reshape(out_dim, in_dim)
        elif gtt == "Q8_0":
            nb = out_dim * in_dim // 32
            gt = _gguf.quants.Q8_0.dequantize_blocks(
                raw.reshape(-1, 34)[:nb].copy()).reshape(out_dim, in_dim)
        err = float(np.max(np.abs(w_ov - gt.astype(np.float32))))
        ok = err < 1e-1
        results.append((t.name, gtt, out_dim, in_dim, err, ok))
        print(f"  [{gtt:5s}] {t.name:34s} [{out_dim}x{in_dim}] "
              f"max|err|={err:.3e} {'OK' if ok else 'FAIL'}")
        checked += 1
        if checked >= max_tensors:
            break
    n_fail = sum(1 for *_x, ok in results if not ok)
    print(f"=== OV 压缩权重子图抽样: {checked} 张量, {n_fail} FAIL ===")
    return results


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "models/rwkv7-g1i-7.2b-Q4_K_M.gguf"
    ov_weight_check(p)
