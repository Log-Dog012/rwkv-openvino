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

---

## 八、Q5_K 支持 + 易用性补齐（task 6 轮）

### 8.1 Q5_K repack 支持（参考 llamacpp ov 后端，目标超越）

- **llamacpp ov 后端映射**（`ggml-openvino-extra.cpp` `get_extracted_layout`）：Q5_K → `is_u4=false`（**u8 容器**）、`weights_per_block=32`、`is_symmetric=false`（非对称 scale+zp）——与 Q4_K 同结构，容器 u8
- **我们的实现**（`gguf_kquant_repack.py` `repack_q5_k`）：向量化，block=256 元素/176B（2B d + 2B dmin + 12B scales + 32B qh + 128B qs），解码 `ql(4bit) | qh(1bit)<<4`（5-bit 值 0-31），8 组 × 32 元素非对称
- **接入**：`repack_tensor` 加 Q5_K 分支（type=i8），`build_compressed_weight` 加 Q5_K 分支（`256, 8, 32, ov.Type.i8`）
- **验证**：STEP1 还原 vs gguf 官方解码 **err=0.0**（bit-exact）✅；OV 压缩权重子图 matmul（合理 scale 数据）**err=3e-5** ✅
- **覆盖对比**：Q4_K/Q6_K/Q8_0/Q5_K 全覆盖 = llamacpp ov 后端支持的全部 K-quant 类型（Q3_K/Q2_K 两边都不支持）

### 8.2 易用性路径实测（GenAI 直接驱动我们的 IR）

| 步骤 | 做法 | 结果 |
|---|---|---|
| eos 元数据 | 模型目录放 `generation_config.json`（`{"eos_token_id": 0}`） | ✅ GenAI 读取成功，`stop_token_ids=-1` 报错消除 |
| tokenizer IR | `openvino_tokenizers.build_rwkv_tokenizer`（官方支持 RWKV vocab v20230424，正是我们的 vocab）转出 `openvino_tokenizer.xml/detokenizer.xml` | ✅ encode/decode 往返正确（'Hello world' → [[33155 40213]] → 'Hello world'） |
| 端到端加载 | 目录含 `openvino_model.xml/bin + generation_config.json + tokenizer/detokenizer IR` | ✅ **PIPE OK**（GenAI 完整加载我们的 IR） |
| 端到端 generate | `pipe.generate('Hello')` | ❌ 报 `attention_mask` 缺失 |

**易用性边界结论**：GenAI `StatefulLLMPipeline` 按 **transformer KV cache 约定**驱动（`input_ids/attention_mask/present.*`），RWKV 的 RNN 状态输入（`idx/s_att_x/s_kv/s_ffn`）不匹配——**bin+xml 能被 GenAI 加载（read_model 层通用），但不能被 LLMPipeline 直接驱动生成**。

**出路（二选一）**：
1. **自写 pipeline**（推荐）：用 `ov::InferRequest` 直接驱动我们的 stateful IR（本仓库分块执行器就是），GenAI 的 LLMPipeline 只借 tokenizer IR 层——「bin+xml 通用 + 自写驱动」
2. **改 IR 输入命名适配 GenAI**：把 `idx/s_att_x/s_kv/s_ffn` 重命名为 GenAI 期望的 `input_ids/attention_mask/present.*`（需要把 RNN 状态伪装成 KV cache，GenAI 的采样器才能工作）——可行性待验证，风险高

**项目定位**：我们的 IR 对标 llamacpp ov 后端产物（bin+xml 标准格式，可被 ov Core read_model / GenAI 加载），驱动层自写 pipeline（超越 llamacpp ov 后端的点：llamacpp ov 后端连加载 RWKV 都崩 CPY，我们至少 IR 全链路可用）。

### 8.3 自写 pipeline 落地（`scripts/genai_style_pipeline.py`）

**方案**（§8.2 出路 1 的实现）：GenAI 风格接口 + 分块执行器驱动 + tokenizer IR 编解码，不依赖 GenAI 的 StatefulLLMPipeline。

- **接口**：`RWKVPipeline(gguf, ir_dir, tokenizer_xml, detokenizer_xml).generate(prompt, max_new_tokens) -> str`（对标 openvino_genai.LLMPipeline）
- **tokenizer 层**：`openvino_tokenizers.build_rwkv_tokenizer` 转出的 IR，`ov::InferRequest` 驱动 encode/decode——验证与仓库 TRIE_TOKENIZER 编解码 **4 测试串全一致**（含中文）✅
- **模型层**：分块执行器语义（chunk IR 链式 `_full_pass`，states `[L,C]/[L,H,N,N]/[L,C]` 持久累积，绕开单图 FFN 退化）
- **端到端冒烟**：13.3b 短 prompt，pipeline ready（6.6s）→ encode → prompt sweep（201s，8 chunk 推理）→ 全链路无崩 ✅；生成阶段逻辑与已验证的分块执行器一致（13.3b 曾产出 ' Paris, France'）
- **踩坑**：①参数名 `gguf` 遮蔽 `import gguf` 模块（改名 `gguf_path`）②`_ir_dir` 未存实例 ③`_full_pass` 需链式 x_in/vf_in 传递（非 stream 数组）④tokenizer IR compile 需 `import openvino_tokenizers` 注册 TrieTokenizer 扩展

**易用性闭环达成**：我们的 IR（bin+xml）+ 官方 tokenizer IR + 自写 pipeline = 完整可用的「GGUF→OVIR→生成」链路，对标 llamacpp ov 后端（其连 RWKV 加载都崩 CPY）。

---

## 九、7.2B 单图（不分块）验证：退化复现 + GPU f16 数值问题首次暴露（task 7 轮，2026-08-29）

> 背景：在本机 32GB（Core Ultra 7 155H）把 7.2B（`rwkv7-g1i-7.2b-20260805-ctx16384.i1-Q4_K_M.gguf`，4.9GB，
> L=32 / C=4096 / FFN 16384，与 13.3B 同宽）做单图 K-quant IR，验证整图 FFN 退化是否复现，
> 并尝试单图常驻绕开分块架构。环境 conda AI + OV 2026.3.0。

### 9.1 分层退化曲线（perf counter，CPU 8 线程，`scripts/ffn_degradation_probe.py`）

| 层数 | 整图 infer 稳态 | ffk avg | ffv avg | 对照 13.3B（§7.1） |
|---|---|---|---|---|
| L=2 | 60.8ms | 6.5ms | 6.3ms | 46ms / 4.9ms（正常） |
| L=4 | 124.6ms | 7.1ms | 7.6ms | 81ms / 4.8ms（正常） |
| L=8 | 210.6ms | **7.6ms（未退化！）** | 7.8ms | 217ms / 20ms（已退化 4×） |
| L=16 | 552.9ms | **61.8ms（退化 8×）** | 60.2ms | 480ms / 34ms |

- 7.2B 退化起点比 13.3B **晚**（L=8 仍单 matmul 量级），但 L=16 退化更猛（61.8 vs 34ms）；
- L=8→16 整图推理 2.6× 超线性，与 13.3B 的「渐进超线性」模式一致；
- **结论：整图 FFN 退化与模型规模无关（同 C/FFN 宽度下按层数复现），起点具体位置有波动（8~16 之间），不可靠层间外推。**

### 9.2 全图（L=32）单图常驻 bench（`ov_single_graph_bench.py`）

| 设备 | 编译 | 稳态 tg | 数值 | 备注 |
|---|---|---|---|---|
| CPU（8 线程） | 738.5s | **0.04 t/s（~25s/token）** | ✅ 首 token 37138 与基线一致，生成 ", France. It" | 退化复现，与 13.3B 单图 200s/token 同级问题 |
| GPU（核显，f16 默认） | 185.3s | **3.23 t/s** | ❌ 生成 " inconvencombeally webs" 乱码 | 无性能退化！但数值错（见 9.3） |
| GPU（INFERENCE_PRECISION_HINT=f32） | 843.4s | 0.004 t/s（~250s/token） | ✅ 与 CPU 完全一致 | 全图抬 f32 修复数值但完全不可用 |

- IR：`out/rwkv7-7.2b-q4k_ov.xml`（bin 6.0GB，GGUF 4.9GB）。CPU 编译缓存 `out/ov_cache` 已达 **76GB**，注意磁盘。
- GPU 单图 3.23 t/s vs 13.3B **分块** GPU 1.63 t/s——若无数值问题，单图在 GPU 上反而比分块快一倍（分块的每 token 重载 chunk 代价在 GPU 上同样存在）。

### 9.3 GPU f16 数值问题定位（重要新发现）

| 图 | CPU vs GPU-f16 | CPU vs GPU-f32（HINT） |
|---|---|---|
| L=1 | maxabs **3.58**，top1 不一致 | maxabs 0.0078，top1 ✅ |
| L=2 | maxabs 11.07 | maxabs 0.0078 |
| L=8 | maxabs 14.54 | — |
| L=32 | maxabs 27.70，生成乱码 | 与 CPU 完全一致 |

- **GPU f16 从第 1 层就错**（不是跨层累积才出错），GPU f32 完全正确 → 根因是 **GPU 插件对 compressed-weights（int4/u8）FullyConnected 用 f16 累加**（CPU 插件 FC 累加 f32 所以 bit-exact），RWKV 乘性状态递归把 f16 累加误差放大成 logits 灾难。
- **历史验证盲区**：此前「L=1 验证 err 2.4e-7 PASS」只在 CPU 上做过，GPU 数值从未被单独验证过——GPU 上的分块推理速度（13.3B GPU 1.63 t/s 等）很可能同样产出乱码，**速度数字有效、文本不可信**。
- 修复尝试：`ALLOW_PRECISION_REDUCTION`（低版本 OV 的「仅 FC 保持 f32 累加」属性）在 2026.3 **已不存在**；`INFERENCE_PRECISION_HINT=f32` 是全图抬精度，代价不可接受。**待试出路**：① 报 OV 上游（GPU int4 FC 累加精度无开关）；② 图级 workaround——把解量化子图输出/FC 输出显式 f32 化（只抬 matmul 相关精度，激活大头保持 f16，代价待测）；③ GPU 只用于对精度不敏感的场合或配合输出校验。

### 9.4 最终结论（task 7）

1. **7.2B 单图 CPU 复现整图 FFN 退化**（0.04 t/s），退化与模型规模无关、按层数渐进超线性——分块执行器仍是 CPU 唯一可行路径（结论与 §7.4 一致）。
2. **GPU 单图无退化且最快（3.23 t/s，编译仅 185s）**，但 f16 累加数值错误使其不可用于生成；这是 OV 2026.3 GPU 插件对 int4 FC 的行为，属性层面无解。
3. 附带修复：`gguf_kquant_repack.py` 新增 **BF16** 分支（7.2B GGUF 的小 matmul a1/a2/g1/g2/v1/v2 是 BF16，bit-exact 转 f32）；`ov_single_graph_bench.py` 支持 GPU（不再传 INFERENCE_NUM_THREADS）+ `--fp32-hint` 开关。
4. 新工具：`scripts/ffn_degradation_probe.py`（任意 GGUF 任意层数的 FFN 退化探针，替代散失的 task 5 脚本）。

**定论（2026-08-30）**：CPU 侧单图常驻（7.2B/13.3B）因整图 FFN 退化正式判死，不再投入；唯一悬念是 GPU 单图（3.23 t/s 无退化）能否绕过 f16 累加数值错误（§9.3 出路 ②）。编译缓存 `out/ov_cache`（76GB，含 13.3B 单图 JIT blob）已随之清理。
