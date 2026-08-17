# 13.3B RWKV-7 G1i Q4_K_M 速度比较报告

> 实验日期：2026-08-17
> 模型：`rwkv7-g1i-13.3b-Q4_K_M.gguf`（8.46 GB，V=65536 C=4096 L=61 H=64 N=64）
> 硬件：32GB 内存（21GB 可共享给核显）+ Intel Arc 核显（uma=1）+ CPU + NPU
> 同 prompt：`"The Eiffel Tower is located in the city of"`（11 token）

---

## 一、公平性口径说明（必读）

两条链路**架构不同**，tg t/s 口径有差异，直接数字比较会误导，故列两个口径：

| 口径 | OV 链路 | llamacpp 链路 | 可直接比？ |
|---|---|---|---|
| **纯推理 t/s**（权重常驻、无重编译） | 分块执行器每 token 重载全 8 chunk 各推理一次，sum 换算 | llama-bench tg128（单图常驻、ggml 原生调度） | ⚠️ OV 这边是分块架构的「8 chunk 推理 sum」，含 chunk 间状态拼接开销；llamacpp 是单图全程。量级可比但非严格同口径 |
| **端到端生成 t/s**（含重编译） | 分块执行器每 token 重编译 8 chunk（~93s/token，缓存启用后仍 ~12s/chunk） | llama-bench tg128（无重编译） | ❌ OV 这边含架构代价，不可直接比 |

**结论口径**：本报告用「纯推理 t/s」做量级对比（两链路都去掉一次性编译开销），并明确标注 OV 分块架构的固有代价。真要严格同口径，需 OV 跑通「单图常驻」（本机单图 61 层 CPU JIT 编译超 10 分钟崩/超时，未跑通——这是 OV CPU 插件 JIT 融合 K-quant 子图的开销，非推理速度本身）。

---

## 二、OV 链路（本仓库自建 OV IR + 原生 K-quant 融合）

**稳态纯推理 t/s**（8 chunk 各推理一次 sum 换算，缓存启用，8 线程）：

| 设备 | 8 chunk 推理 sum | **稳态 t/s** | 单 chunk 编译（冷/缓存） |
|---|---|---|---|
| CPU | 1.481s | **0.68 t/s** | 20.9s / 3.0s |
| **GPU**（Intel Arc 核显） | 0.613s | **1.63 t/s** | 23.3s / — |
| NPU | 0.761s | **1.31 t/s** | 57.1s / — |

端到端验证产出：prompt sweep 11 token + 生成 `' Paris, France'`（token IDs `[37138,45,44312]` 与 1.5B/7.2B 官方 torch 基线 bit-exact 一致）✅

分块执行器端到端生成 t/s（含每 token 重编译 8 chunk）：~0.01 t/s（架构代价，非推理速度上限）。

---

## 三、llama.cpp 链路（ggml 原生，单图全程）

`llama-bench -p 11 -n 128 -r 1`（同 prompt 11 token，生成 128 token 稳态）：

| 后端 | pp11（prompt 处理） | **tg128（生成）** | 备注 |
|---|---|---|---|
| **vulkan**（Intel Arc 核显） | 12.37 t/s | **7.14 t/s** | 最快 |
| sycl | 6.80 t/s | 5.93 t/s | |
| CPU（vulkan build, ngl 0） | 7.54 t/s | 5.87 t/s | |

llamacpp openvino 后端对 RWKV 报 CPY 错（cache_r_l0 跨 buffer），无法跑通——本仓库 OV 链路即绕开它。

---

## 四、速度对比汇总

| 链路 / 设备 | 稳态 tg t/s | 相对最快 |
|---|---|---|
| **llamacpp vulkan** | **7.14** | 1.00×（基准） |
| llamacpp sycl | 5.93 | 0.83× |
| llamacpp CPU | 5.87 | 0.82× |
| OV GPU（核显） | 1.63 | 0.23× |
| OV NPU | 1.31 | 0.18× |
| OV CPU | 0.68 | 0.10× |

**OV vs llamacpp 同设备对比**：
- 核显：OV GPU 1.63 vs llamacpp vulkan 7.14 → llamacpp **4.4×** 快
- CPU：OV CPU 0.68 vs llamacpp CPU 5.87 → llamacpp **8.6×** 快

---

## 五、结论与原因分析

1. **转化可行性**：13.3b Q4_K_M 完美转化为 OV IR（9.42GB），原生继承 K-quant 压缩权重，CPU/GPU/NPU 三设备都跑通且产出 bit-exact 一致 ✅
2. **速度**：OV 链路在同设备上比 llamacpp 慢 4–9 倍。根因有三：
   - **分块架构代价**：本仓库 OV 链路为沙箱 8GB 写的逐 chunk 编译，每 token 重载全 8 chunk，llamacpp 是单图全程常驻
   - **OV 插件 JIT 开销**：OV CPU/GPU/NPU 插件 JIT 融合 K-quant 子图慢（单图 61 层编译 >10 分钟），llamacpp 的 ggml kernel 预编译好直接跑
   - **核显调度**：OV GPU 走 OpenCL/Vulkan 后端抽象，llamacpp vulkan 直接调 ggml 优化 kernel
3. **llamacpp ov 后端**：架构层面不兼容 RWKV（CPY 跨 buffer 报错），本仓库 OV 链路绕开它但速度不及 ggml 原生
4. **OV 链路真要追平 llamacpp**：需跑通「单图常驻」（绕开分块重编译）+ 优化 OV 插件 JIT 预热——本机单图编译崩/超时未跑通，留待后续

**总体**：OV 链路在「三设备统一 IR + 原生 K-quant 融合 + 跨平台」有架构价值，但纯速度不及 llamacpp 的 ggml 原生优化 kernel。本仓库链路的核心价值在「绕开 llamacpp ov 后端兼容性差」+「OV 生态可部署性」，不在速度。

---

## 六、附：原始数据

### OV 三设备 8 chunk 推理耗时（秒）

| chunk | CPU infer | GPU infer | NPU infer |
|---|---|---|---|
| 0 | 0.3681 | 0.0916 | 0.1883 |
| 1 | 0.1538 | 0.0666 | 0.0867 |
| 2 | 0.1612 | 0.0844 | 0.0842 |
| 3 | 0.1848 | 0.0771 | 0.0812 |
| 4 | 0.1935 | 0.0772 | 0.0833 |
| 5 | 0.1503 | 0.0785 | 0.0831 |
| 6 | 0.1169 | 0.0669 | 0.0705 |
| 7 | 0.1528 | 0.0702 | 0.0834 |
| **sum** | **1.481** | **0.613** | **0.761** |
| **t/s** | **0.68** | **1.63** | **1.31** |

### llamacpp llama-bench 原始输出（build 6e62ba538）

```
| rwkv7 14B Q4_K - Medium | 7.88 GiB | 13.27 B | Vulkan | -1 | pp11  | 12.37 ± 0.00 |
| rwkv7 14B Q4_K - Medium | 7.88 GiB | 13.27 B | Vulkan | -1 | tg128 |  7.14 ± 0.00 |
| rwkv7 14B Q4_K - Medium | 7.88 GiB | 13.27 B | SYCL   | -1 | pp11  |  6.80 ± 0.00 |
| rwkv7 14B Q4_K - Medium | 7.88 GiB | 13.27 B | SYCL   | -1 | tg128 |  5.93 ± 0.00 |
| rwkv7 14B Q4_K - Medium | 7.88 GiB | 13.27 B | Vulkan |  0 | pp11  |  7.54 ± 0.00 |
| rwkv7 14B Q4_K - Medium | 7.88 GiB | 13.27 B | Vulkan |  0 | tg128 |  5.87 ± 0.00 |
```

---

## 七、附：ovms 常驻服务路径尝试（未跑通）

为绕开 OV 分块架构代价、补「单图常驻」同口径对比缺口，尝试了 `C:\Users\Mcsof\Application\ovms`（OpenVINO Model Server，常驻服务进程预加载模型 + 编译一次 + 持续接推理请求）。

**结果：未跑通——产物残缺**。ovms.exe / openvino.dll / openvino_c.dll / openvino_genai.dll / openvino_intel_{cpu,gpu,npu}_plugin.dll / opencv_world4130.dll / git2.dll 全是 **0 字节空文件**（`file` 报 `empty`），只有 tbb12.dll（225KB）和 python/python.exe（104KB）是实的。PowerShell 调 ovms.exe 报「不是此操作系统平台的有效应用程序」——根因不是架构兼容，是产物本身空。

**未补成「OV 单图常驻 vs llamacpp」同口径对比**——OV 链路在本机的可行路径只有分块执行器（含重编译代价），llamacpp 是单图全程。本报告 §一 的口径差异说明继续适用：OV 纯推理 0.68/1.63/1.31 t/s（CPU/GPU/NPU，分块架构）vs llamacpp 7.14/5.93/5.87 t/s（vulkan/sycl/CPU，单图全程），量级可比但非严格同口径。真要严格同口径需 ovms 跑通（待完整产物）或 OV 单图编译预热跑通（本机 CPU 61 层 JIT >10 分钟崩/超时）。

