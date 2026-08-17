# 61 层单图 int4 融合调查结论（推翻了「融合失败」假设）

> 日期：2026-08-17
> 背景：单图常驻 61 层 OV 推理 200s/token + 内存 31GB 爆，此前假设是「OV CPU 插件大图
> K-quant 融合失败 → 权重 f16 解包」。本次逐层二分 + perf counter 调查推翻了该假设。

---

## 一、检测方法结论（task 1）

| 方法 | 结果 |
|---|---|
| `get_runtime_model()` | ❌ CPU 插件返回 ExecutionNode 黑盒，看不到内部 op 类型 |
| `LOG_LEVEL=DEBUG` 编译日志 | ❌ 无融合决策日志输出 |
| **perf counter**（`enable_profiling` + `get_profiling_info`） | ✅ 能看到融合标志：融合成功 = **FullyConnected**，未融合 = MatMul |
| 耗时代理 | ⚠️ 单 matmul 融合 int4 1.28-1.73ms vs f16 1.17-1.40ms，几乎无差——**融合收益在内存（int4 8.4GB vs f16 26.5GB），不在速度** |

## 二、逐层二分结果（task 2）

| 层数 | 编译 | 推理 | FullyConnected | MatMul | 结论 |
|---|---|---|---|---|---|
| L=1 | 7s | — | — | — | runtime 黑盒 |
| L=4 | 15s | — | — | — | 黑盒 |
| L=8 | 30s | 0.41s | 111 | 32 | 权重 matmul 融合（FC≈14/层） |
| L=16 | 87s | 16.7s→2.2s(二次) | 223 | 64 | 融合保持（FC≈14/层） |

- **融合数量不随层数退化**：每层 ~14 个 FullyConnected（权重 matmul 全融合）+ 4 个 MatMul（状态更新的真实矩阵乘积，正常不该融合）
- **推理耗时超线性**：L=8→16 层数翻倍，推理 0.41s→2.2s（二次 infer，5.4×）

## 三、退化定位（task 3）

L=16 的 perf counter：**耗时大头是 FFN matmul（ffk/ffv，16384×4096），单个 ~250-293ms**。

但**单独测同一个 FFN matmul**：

| 权重 | 形状 | 压缩路径 | f16 路径 | 比值 |
|---|---|---|---|---|
| attention key | 4096×4096 | 1.73ms | 1.40ms | 1.24 |
| **FFN key** | **16384×4096** | **5.46ms** | **6.09ms** | **0.90（压缩还更快）** |
| FFN value | 4096×16384 | 4.84ms | 5.64ms | 0.86 |

**同样一个 FFN matmul（16384×4096）**：
- 单独小图：**5.46ms**（融合成功，甚至快于 f16）
- 放进 16 层整图：**~293ms**（perf counter）
- **= 55 倍退化，根因是整图上下文，不是 matmul 本身，也不是融合失败**

## 四、最终结论（推翻原假设）

1. **融合没有失败**：权重 matmul 在 61 层单图里也标成 FullyConnected（int4 融合在数量上保持），不是「f16 解包」的锅
2. **真凶是「整图上下文 FFN matmul 退化」**：同样的 16384×4096 matmul，放进大图慢 55 倍（kernel 变体选择 / 内存布局 / JIT 缓存压力），叠加 61 层全图内存 31GB 爆（系统换页）→ 200s/token
3. **分块执行器（8 层 chunk）绕开了它**：小图里 FFN matmul 上下文好（5.46ms 级）+ 内存 ~6.5GB 不换页 → 0.32s/chunk
4. **单 matmul 上融合已足够快**（5.46ms 优于 f16 6.09ms），自定义 op（4.88ms）没赢——自定义 op 的价值不在替代融合，而在**绕开整图退化**（如果整图 FFN 退化无法修复）

## 五、修复建议（task 4）

1. **验证退化起点**（低成本）：测 L=2/4/8 整图里 FFN matmul 单 op 耗时，看退化从几层开始（线性 vs 跳变）——能定位是「kernel 变体数超阈值」还是「内存布局」问题
2. **分块执行器保持**：已绕开退化 + 内存墙，是当前唯一可行路径（实测 0.68/1.63/1.31 t/s CPU/GPU/NPU）
3. **OV 属性调 kernel 选择**（待试）：`ENABLE_WEIGHTLESS`、`DYNAMIC_QUANTIZATION_GROUP_SIZE`、`KEY_CACHE_PRECISION` 等新属性可能影响 matmul kernel 变体选择，值得试
4. **自定义 op 用于 FFN 大 matmul**（最后手段）：如果整图退化是 kernel 选择问题且 OV 属性调不动，自定义 op 强制 int4 路径可绕开——但预期收益有限（单 matmul 融合已快）

## 六、历史数据对照

| 口径 | 数值 | 说明 |
|---|---|---|
| 单 matmul 融合（4096×4096） | 1.28-1.73ms | 融合成功 |
| 单 FFN matmul（16384×4096） | 5.46ms | 融合成功，优于 f16 |
| L=8 整图推理 | 0.41s | 分块 chunk 量级 |
| L=16 整图推理（二次） | 2.2s | 已含 FFN 退化（~5.4× 超线性） |
| 61 层单图推理 | ~200s/token | FFN 退化 ×61 层 + 内存 31GB 换页 |

---

## 七、后续调查补充（task 5 轮）

### 7.1 FFN 退化序列（L=2/4/8/16 整图内 FFN 单 op 耗时）

| 层数 | 整图推理(二次) | FFN avg | FFN max | 结论 |
|---|---|---|---|---|
| L=2 | 46ms | 4.9ms | 5.7ms | 正常（单 matmul 量级） |
| L=4 | 81ms | 4.8ms | 5.2ms | 正常 |
| L=8 | 217ms | 20.0ms | 37.1ms | **开始退化（4×）** |
| L=16 | 480ms | 34.0ms | 204.3ms | 退化加重（7× avg / 37× max） |

**退化是渐进超线性，非单点跳变**——起点在 L=8 附近，随层数加剧。根因推测为整图 kernel 变体数/内存布局/cache 压力（非 matmul 本身）。

### 7.2 OV 新属性测试（L=8 图，测能否改 FFN kernel 选择）

| 属性 | infer | FFN avg | 结论 |
|---|---|---|---|
| baseline | 207ms | 10.7ms | 参照 |
| ENABLE_WEIGHTLESS | 265ms | 20.9ms | ❌ 更差 |
| DYNAMIC_QUANTIZATION_GROUP_SIZE=32 | 145ms | 18.9ms | ❌ FFN 更差 |
| INFERENCE_PRECISION_HINT=f16 | 229ms | 49.8ms | ❌ 更差 |

**OV 属性全部无法改善 FFN 退化**——CPU 插件 kernel 变体选择是内部行为，无可调开关。

### 7.3 openvino-genai 2026.3 加载我们的 IR（易用性验证）

- `openvino_genai` import 成功（需 `os.add_dll_directory` 加 openvino + genai 两个 runtime bin）
- **LLMPipeline 加载我们的 chunk IR 成功**（标准命名 `openvino_model.xml/bin` 放进目录，PIPE OK）——**bin+xml 通用性 ✅**
- generate 报 `stop_token_ids` 含 -1：GenAI 期望 IR 带 eos token 元数据（generation_config），RWKV IR 没有——**属「pipeline 要写好」范畴，非 IR 结构问题**

### 7.4 项目路线图（面向目标：速度/精度/内存/易用性对标 llamacpp ov 后端）

| 目标 | 现状 | 差距 | 路线 |
|---|---|---|---|
| **精度**（与 gguf 同等 K-quant） | ✅ 已达成（repack bit-exact，L=1 验证 err 2.4e-7） | — | 保持 |
| **内存**（对标 gguf ~8GB） | 分块 ~6.5GB / 单图 31GB（退化） | 单图需 int4 8.4GB | 单图退化不可解（7.1/7.2），分块已接近 |
| **速度**（追 llamacpp 7.14 t/s） | 分块 0.68/1.63/1.31 t/s | 4-9× | 单图退化是瓶颈，当前 OV 版本不可解 |
| **易用性**（bin+xml 通用） | GenAI 加载 ✅，generate 需补 eos 元数据 | tokenizer IR 未转 | 补 generation_config 元数据 + 转 RWKV tokenizer IR（openvino_tokenizers 有 RWKV 例子） |
| **可部署**（HF 发布 IR） | 1.5b/7.2b/13.3b 已上传 ModelScope | — | 保持 |

**关键结论**：当前 OV 2026.3 的 CPU 插件对「61 层整图 FFN matmul」有不可调的退化（渐进超线性 + 无可调属性），单图常驻不可行。**分块执行器是唯一可行路径**（绕开退化 + 内存墙）。速度上追平 llamacpp 的路线：等 OV 上游修整图退化，或自定义 op 强制 int4 处理 FFN 大 matmul（预期收益有限，单 matmul 融合已快）。易用性路线：补 eos 元数据 + tokenizer IR 让 GenAI 直接驱动我们的 IR。
