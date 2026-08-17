// RWKV K-quant matmul custom op (OpenVINO CPU extension)
//
// Goal: bypass the OV CPU plugin's failure to auto-fuse K-quant compressed
// weights on large graphs. This op takes the repacked int4 codes + scales +
// zp directly and runs a hand-written AVX2 dequant+matmul kernel (same idea
// as llama.cpp ggml, but registered through OV's extension mechanism).
//
// Inputs (aligned with scripts/gguf_kquant_repack.py repack output):
//   x      [M, K]  f16/f32   activations (M=1 for single-token)
//   codes  [N]     u8        one 4-bit value 0-15 per element (block-major flat, N=O*K)
//   scales [G]     f32       one absolute scale per 32 elements (G=N/32)
//   zp     [G]     f32       one absolute min/zp per 32 elements (Q4_K asym)
// Output:
//   y      [M, O]  f32       y = x @ dequant(codes, scales, zp)
//
// Dequant semantics (matches gguf Q4_K / repack_q4_k):
//   w[k] = scales[k/32] * codes[k] - zp[k/32]

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/op.hpp"

class RwkvKQuantMatMul : public ov::op::Op {
public:
    OPENVINO_OP("RwkvKQuantMatMul", "rwkv", ov::op::Op)

    RwkvKQuantMatMul() = default;
    explicit RwkvKQuantMatMul(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // x [M, K], codes [N], scales [G], zp [G] -> y [M, O], O = N / K
        const auto& x_ps = get_input_partial_shape(0);
        const auto& codes_ps = get_input_partial_shape(1);
        OPENVINO_ASSERT(get_input_element_type(1) == ov::element::u8, "codes must be u8");
        OPENVINO_ASSERT(get_input_element_type(2) == ov::element::f32, "scales must be f32");
        OPENVINO_ASSERT(get_input_element_type(3) == ov::element::f32, "zp must be f32");

        ov::PartialShape out_ps = x_ps;
        if (x_ps.rank().is_static() && codes_ps.rank().is_static()) {
            const int64_t M = x_ps[0].get_length();
            const int64_t K = x_ps[1].get_length();
            const int64_t N = codes_ps[0].get_length();
            OPENVINO_ASSERT(N % K == 0, "codes length must be multiple of K");
            out_ps = ov::PartialShape{M, N / K};
        }
        set_output_type(0, ov::element::f32, out_ps);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<RwkvKQuantMatMul>(new_args);
    }

    // CPU 插件 reference fallback 用 has_evaluate() 判断是否可走 reference 路径，
    // Node 默认返回 false，必须 override 为 true（否则报 "evaluate() is not implemented"）。
    bool has_evaluate() const override { return true; }

    // CPU 插件 reference fallback 走带 EvaluationContext 的 evaluate 重载，
    // 只实现无 context 版本会被当 "not implemented"，两个都实现。
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  const ov::EvaluationContext&) const override {
        return evaluate(outputs, inputs);
    }

    bool visit_attributes(ov::AttributeVisitor&) override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        const auto& x_t = inputs[0];
        const auto& codes_t = inputs[1];
        const auto& scales_t = inputs[2];
        const auto& zp_t = inputs[3];
        auto& y_t = outputs[0];

        const int64_t M = x_t.get_shape()[0];
        const int64_t K = x_t.get_shape()[1];
        const int64_t N = codes_t.get_shape()[0];
        const int64_t O = N / K;
        const int64_t G = scales_t.get_shape()[0];
        OPENVINO_ASSERT(G == N / 32, "scales count must be N/32");

        const uint8_t* codes = codes_t.data<const uint8_t>();
        const float* scales = scales_t.data<const float>();
        const float* zp = zp_t.data<const float>();
        float* y = y_t.data<float>();

        const bool x_is_f16 = x_t.get_element_type() == ov::element::f16;
        const void* x_ptr = x_t.data();

        const int64_t blocks_per_row = K / 256;  // Q4_K: 每行 256 元素一个 block

        for (int64_t o = 0; o < O; ++o) {
            for (int64_t m = 0; m < M; ++m) {
                __m256 acc = _mm256_setzero_ps();
                for (int64_t k = 0; k < K; k += 8) {
                    // scales/zp 是 block-major 展平 [nblk*8]：物理行 o、列 k 的元素
                    // 落在 block b = o*(K/256) + k/256，block 内组 = (k/32)%8（8|32，8 元素不跨组/不跨 block）。
                    const int64_t g = (o * blocks_per_row + k / 256) * 8 + (k / 32) % 8;
                    const float s = scales[g];
                    const float z = zp[g];

                    // raw 字节序 = 物理行优先 [O,K]：物理行 o 的列 k..k+7 连续
                    const uint8_t* co = codes + static_cast<size_t>(o) * K + k;
                    // 8 u8 codes -> 8 f32
                    __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(co));
                    __m256 c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
                    // deq = s*c - z
                    __m256 deq = _mm256_fmadd_ps(_mm256_set1_ps(s), c, _mm256_set1_ps(-z));
                    // x 8 elements
                    __m256 xv;
                    if (x_is_f16) {
                        const uint16_t* xh = static_cast<const uint16_t*>(x_ptr) + static_cast<size_t>(m) * K + k;
                        __m128i xh8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(xh));
                        xv = _mm256_cvtph_ps(xh8);
                    } else {
                        const float* xf = static_cast<const float*>(x_ptr) + static_cast<size_t>(m) * K + k;
                        xv = _mm256_loadu_ps(xf);
                    }
                    acc = _mm256_fmadd_ps(deq, xv, acc);
                }
                float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                y[static_cast<size_t>(m) * O + o] = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) +
                                                    (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]);
            }
        }
        return true;
    }
};

static std::vector<ov::Extension::Ptr> get_extensions() {
    return {std::make_shared<ov::OpExtension<RwkvKQuantMatMul>>()};
}

OPENVINO_CREATE_EXTENSIONS(get_extensions())
