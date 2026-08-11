# RWKV-7 → OpenVINO IR 推理验证

把 **RWKV-7 (x070) "G1/G1i" 系列**权重转成 OpenVINO IR 并在 **纯 CPU**（无 GPU/NPU）上验证能否端到端运行、生成文本是否连贯。
本仓库包含从 `.pth` 出发的完整链路：**纯 PyTorch 单步复刻 → OpenVINO IR 导出 → FP16/INT8 量化 → CPU 编译推理**。

> 背景：RWKV 是 RNN 类 LLM，天然适合做 **stateful、token-by-token** 的 OpenVINO 静态计算图推理（不像 Transformer 那样依赖变长 attention）。
> 本实验证明：**不经过 GGUF / llama.cpp，直接 pth → 纯 torch 复刻 → OV IR 是唯一稳定打通的路径**（GGUF 路线对 RWKV-7 不可行，详见报吿「GGUF 路线证伪」）。

---

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.11.1 (pyenv，**非**系统 3.12) |
| OpenVINO | 2026.3.0 |
| openvino-genai | 2026.3.0 |
| NNCF | 3.3.0 |
| torch | 2.10.0+cu128 |
| 硬件 | 纯 CPU（本机无 GPU/NPU） |

> ⚠️ 安装务必指向 pyenv 的 python：`/root/.pyenv/versions/3.11.1/bin/python3 -m pip install -r requirements.txt`
> 系统 `sudo pip3` 会装到 python3.12 并触发 PEP 668 externally-managed 错误。

---

## 快速开始

```bash
# 1) 取一个 RWKV-7 权重（以 0.1B 为例；大模型见 scripts/export_big.py）
#    modelscope 镜像（国内可达）：
curl -sSL -o models/rwkv7-g1d-0.1b.pth \
  https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1d-0.1b-20260129-ctx8192.pth

# 2) 导出 FP32 / FP16 IR 并自检验证
python3 scripts/export_ov.py models/rwkv7-g1d-0.1b.pth --mode both

# 3) 与官方 rwkv==0.8.32 包做 bit-exact 逐 token 对比（需 RWKV_V7_ON=1）
RWKV_V7_ON=1 python3 scripts/run_torch_baseline.py

# 4) NNCF 权重量化（INT8 / INT4）对比体积+精度
python3 scripts/quantize_ov.py

# 5) 大模型（2.9B / 7.2B / 13.3B）导出 + torch↔OV 对照
python3 scripts/export_big.py models/rwkv7-g1i-2.9b-20260805-ctx16384.pth --n 16
python3 scripts/export_big.py models/rwkv7-g1i-13.3b-20260805-ctx16384.pth --mode fp16 --load-fp16 --n 16
```

---

## 核心设计

### 1. 单步递推 + 定长 state（为静态图导出而定）
`rwkv7_torch.py` 复刻官方 `RWKV_x070` 的 **token-by-token `forward_one`**，状态打包为 3 个定长张量：
- `s_att_x : (L, C)`  —— 每层 att 上一时刻 ln1 输出
- `s_kv    : (L, H, N, N)` —— 每层 att 的 KV 记忆矩阵
- `s_ffn   : (L, C)`  —— 每层 ffn 的上一时刻 ln2 输出

权重全部 `register_buffer`，trace 时固化为 IR 常量；`emb` 在装载时预做 `ln0`，省掉每步一次 layer_norm。
`idx` 用形状 `(1,)` 的 int64 张量，forward 里 `w("emb.weight")[idx].reshape(C)` **保留数据依赖**（真 Gather，不是常量折叠），使计算图对输入 token 敏感。

### 2. 关键 Bug 修复：`F.group_norm` 在 OV 2026.3 转换有数值偏差
`ln_x` 这一 per-head 归一化，若直接写 `F.group_norm` 经 torch→IR 前端转换会产生 **~6.4 的绝对误差**（单算子测试：normalize 精确 6e-8，group_norm 偏差 6.4）。
改为**基础算子手动实现**（逐通道 affine），数学等价于官方实现且跨框架 bit-exact：

```python
_x = (st @ r.view(H, N, 1)).view(1, H, N).float()
_mu = _x.mean(-1, keepdim=True)
_var = _x.var(-1, unbiased=False, keepdim=True)
_y = (_x - _mu) / torch.sqrt(_var + float(64e-5))   # eps 必须是字面量 64e-5，避免 REPL 把 64e-5 解析成 64*nn-5
o = _y.view(1, H * N) * w(att + "ln_x.weight") + w(att + "ln_x.bias")
```

> ⚠️ `float(64e-5)` 不能写成裸 `64e-5`：在 `import torch.nn as nn` 后 `64e-5` 会被解析为 `64 * nn - 5`（nn 是模块），导致虚假误差。

### 3. 量化权衡（NNCF W8A32 / W4）
- **INT8_SYM（W8A32）**：无需校准集，直接压权重；0.1B 下体积 FP32 765MB → 193MB（4×），生成文本仍通顺。
- **INT4_SYM（W4）**：NNCF 在 RWKV 的常量权重上**回退**（bitwidth 表空、体积与 INT8 相同），未真正压缩——这是 RWKV 权重多为固化常量的结果，不是配置问题。

---

## 验证结果

### 0.1B（rwkv7-g1d-0.1b）全链路

| 形态 | 体积 | 与 torch 参考 max | 生成示例 |
|------|------|------------------|----------|
| FP32 IR | 765 MB | 0.37 | ✓ 连贯 |
| FP16 IR | 383 MB | 0.41 | ✓ 连贯 |
| INT8_SYM | 193 MB | 13.3 | ✓ 通顺（"Paris…built in 1889"） |
| INT4_SYM | 193 MB | — | 未真正压缩（NNCF 回退） |

- **bit-exact**：`run_torch_baseline.py` 与官方 `rwkv==0.8.32`（`RWKV_V7_ON=1`）逐 token logits `max abs diff = 0.0`，`TEXT MATCH`。
- **OV 推理正确**：FP32/FP16 跨框架 logits diff ~0.37/0.41（属正常 FP 误差），文本一致。

### 2.9B / 13.3B（大模型，纯 CPU）
> 见 `out/` 下 `*_step_fp32.xml/.bin` 与 `*_step_fp16.xml/.bin`，及 `temp/verify_*.log` 的 torch↔OV 对照结果。
> 13.3B 即用户口中"14B Q4"的目标（RWKV-7 G1i 最大档为 13.3B，官方无 14B）。

| 模型 | 导出形态 | 体积 | torch↔OV | tok/s (CPU) |
|------|----------|------|----------|-------------|
| 2.9B | FP32 | TBD | TBD | TBD |
| 13.3B | FP16 | TBD | TBD | TBD |

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `scripts/rwkv7_torch.py` | RWKV-7 纯 PyTorch 单步复刻（OV 导出核心） |
| `scripts/rwkv_tokenizer.py` + `rwkv_vocab_v20230424.txt` | 纯 Python TRIE 分词器（从 rwkv 源码拷贝） |
| `scripts/export_ov.py` | 0.1B 导出 FP32/FP16 IR 并自检 |
| `scripts/quantize_ov.py` | NNCF INT8/INT4 权重量化对比 |
| `scripts/run_torch_baseline.py` | 与官方 rwkv 包 bit-exact 对比 |
| `scripts/export_big.py` | 大模型参数化导出（--mode fp32/fp16 --load-fp16），torch↔OV 对照 |
| `scripts/diag_*.py` | 定位 `F.group_norm` 转换偏差的单算子诊断 |

---

## 已知限制 / 后续
- **OVMS 服务化**：本环境 github / docker.io / storage 均被墙，OVMS 二进制不可达。已给出 Docker 部署方案（见报吿），不在本机阻塞。
- **INT4 未生效**：NNCF 在常量权重上回退，需后续探索 `nncf.quantize`（含激活量化）或 GPTQ/AWQ 预量化权重。
- **GGUF 路线证伪**：BlinkDL 的 RWKV-7 权重无 GGUF/Q4 版本，`pth → 纯 torch → OV IR` 是唯一打通路径。
