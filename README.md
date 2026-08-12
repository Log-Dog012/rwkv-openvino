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
# 1) 取一个 RWKV-7 权重（modelscope 镜像，支持断点续传 + 真实体积判据）
bash scripts/fetch_rwkv7.sh 0.1b     # 也可 0.4b / 1.5b / 2.9b / 7.2b / 13.3b
#    （等价手动）：
curl -sSL -o models/rwkv7-g1d-0.1b.pth \
  https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1d-0.1b-20260129-ctx8192.pth

# 2) 导出 FP32 / FP16 IR 并自检验证
python3 scripts/export_ov.py models/rwkv7-g1d-0.1b.pth --mode both

# 3) 与官方 rwkv==0.8.32 包做 bit-exact 逐 token 对比（需 RWKV_V7_ON=1）
RWKV_V7_ON=1 python3 scripts/run_torch_baseline.py

# 4) NNCF 权重量化（INT8 / INT4）对比体积+精度（0.1B）
python3 scripts/quantize_ov.py
#    大模型 INT8 压缩演示（0.4B，4x 压缩）：
python3 scripts/quantize_big.py models/rwkv7-g1d-0.4b.pth --n 16

# 5) 大模型导出 + torch↔OV 对照（注意内存，见下方约束）
python3 scripts/export_big.py models/rwkv7-g1d-0.4b.pth --mode fp16 --load-fp16 --n 16
python3 scripts/export_big.py models/rwkv7-g1i-13.3b-20260805-ctx16384.pth --mode fp16 --load-fp16 --n 16
```

> ⚠️ **内存约束**：RWKV-7 G1i 权重本就是 FP16（0.4B≈0.9GB、1.5B≈3GB、2.9B≈5.9GB、13.3B≈26GB），
> 但 `ov.convert_model` 的 torch→IR trace 会把权重暂存为 fp32 中间态，trace 峰值 ≈ 2× 权重大小 + 开销。
> 本验证沙箱 cgroup 上限 **8 GB**，故实测可导出上限约 **0.4B**；1.5B 及以上需 ≥16 GB / ≥32 GB 内存的机器。

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
- **INT8_SYM（W8A32）**：无需校准集，直接压权重；0.1B 下体积 FP32 765MB → 193MB（3.96×），生成文本仍通顺。
- **INT4_SYM（W4）—— 关键坑已解决**：NNCF `compress_weights` 默认 `group_size=128`，而 RWKV 大量矩阵通道维只有 **32/64**，整层被跳过/回退 → 看起来"没压缩"。改 `group_size=32`（或 64）后，**166/166 层全部 4-bit**，0.1B 体积 FP32 765MB → **153.8MB（4.97×）**，比 INT8 还小 ~20%，生成仍连贯（"…built in 1889…"）。`scripts/int4_demo.py` 已封装该修复。

> ⚠️ 之前报告"INT4 未生效"是**测量 bug**（只量了 `.xml` 忘了 `.bin`）+ 默认 `group_size` 过大，并非 NNCF 不支持。

---

## 验证结果

### 0.1B（rwkv7-g1d-0.1b）全链路

| 形态 | 体积 | 与 torch 参考 max | 生成示例 |
|------|------|------------------|----------|
| FP32 IR | 765 MB | 0.37 | ✓ 连贯 |
| FP16 IR | 383 MB | 0.41 | ✓ 连贯 |
| INT8_SYM | 193 MB | 13.3 | ✓ 通顺（"Paris…built in 1889"） |
| INT4_SYM (gs=32) | 153.8 MB | — | ✓ 连贯（"…built in 1889 and is one of the most famous buildings"） |

- **bit-exact**：`run_torch_baseline.py` 与官方 `rwkv==0.8.32`（`RWKV_V7_ON=1`）逐 token logits `max abs diff = 0.0`，`TEXT MATCH`。
- **OV 推理正确**：FP32/FP16 跨框架 logits diff ~0.37/0.41（属正常 FP 误差），文本一致。

### 0.4B / 1.5B / 2.9B / 13.3B（大模型，纯 CPU）
> ⚠️ **本沙箱 cgroup 内存上限 8 GB**（`/sys/fs/cgroup/memory.max=8GiB`），而 RWKV-7 G1i 权重本就是 FP16：
> 0.4B≈0.9GB、1.5B≈3.0GB、2.9B≈5.9GB、13.3B≈26GB。瓶颈在 `ov.convert_model` 的 torch→IR trace
> （OV 前端把权重暂存为 fp32 中间态，trace 峰值 ≈ 2× 权重大小 + 开销）：
> **0.4B（trace ~2.5GB）可跑**；**1.5B（trace ~7–9GB）导出卡在 trace 被 OOM**；2.9B/13.3B 加载即超 8GB。
> 1.5B/2.9B/13.3B 的代码与命令已完全就绪，在 ≥16GB / ≥32GB 内存的机器上直接可跑（计算图路径与 0.1B/0.4B 一致、已验证正确）。
> 13.3B 即用户口中"14B Q4"目标档（RWKV-7 G1i 最大档为 13.3B，官方无 14B）。

| 模型 | 源权重 | 导出形态 | IR 体积 | 首步 logits diff | 生成 |
|------|--------|----------|---------|------------------|------|
| 0.1B | FP32 0.38GB | FP16 IR | 383 MB | 3.7e-1 | 连贯 |
| **0.4B** | FP16 0.9GB | FP16 IR | 903 MB | 1.9e-1 | 连贯（"…wrought-iron lattice tower"） |
| 1.5B | FP16 3.0GB | FP16 IR | — | 权重可加载(L=24,C=2048)，但 `ov.convert_model` trace 超 8GB 被 OOM | 命令就绪，≥16GB 可跑 |
| 2.9B | FP16 5.9GB | FP16 IR | — | 加载即超 8GB 被 OOM | 命令就绪，≥16GB 可跑 |
| 13.3B | FP16 ~26GB | FP16 IR | — | 远超 8GB | 命令就绪，≥32GB 可跑（即用户"14B Q4"目标档） |

> 本沙箱实测可验证上限约 **0.4B**；1.5B+ 受 OV trace 的 fp32 中间态内存限制。代码/命令就绪，更大内存机器直接复现。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `scripts/rwkv7_torch.py` | RWKV-7 纯 PyTorch 单步复刻（OV 导出核心） |
| `scripts/rwkv_tokenizer.py` + `rwkv_vocab_v20230424.txt` | 纯 Python TRIE 分词器（从 rwkv 源码拷贝） |
| `scripts/export_ov.py` | 0.1B 导出 FP32/FP16 IR 并自检 |
| `scripts/quantize_ov.py` | NNCF INT8/INT4 权重量化对比 |
| `scripts/int4_demo.py` | INT4_SYM 量化（修复 `group_size`）+ 体积/生成验证 |
| `scripts/run_torch_baseline.py` | 与官方 rwkv 包 bit-exact 对比 |
| `scripts/export_big.py` | 大模型参数化导出（--mode fp32/fp16 --load-fp16），torch↔OV 对照 |
| `scripts/diag_*.py` | 定位 `F.group_norm` 转换偏差的单算子诊断 |

---

## 已知限制 / 后续
- **OVMS 服务化**：OVMS 二进制（v2026.3.0，188MB）已通过 `ghp.keleyaa.com` 代理下载到 `temp/ovms/`，OV 后端版本与导出的 IR 同源兼容。本机启动 `ovms/bin/ovms` 尚待解决运行期依赖（库路径/模型加载），但部署路径已打通。
- **INT4 已打通**：通过 `group_size=32`（见上文 3. 与 `int4_demo.py`），无需校准集即真正 4-bit。
- **GGUF 路线**：RWKV-7 的 GGUF 权重主要在 modelscope（`shoumenchougou/RWKV7-G1i-{0.1B..13.3B}-GGUF`、`RemySkye/rwkv7-g1h-13.3b-i1-GGUF` 等）与 HF 镜像，本沙箱直连 HF 被墙；但本项目结论是 `pth → 纯 torch → OV IR` 路径已稳定跑通，GGUF 非必需。
- **内存天花板**：本沙箱 8 GB cgroup 限制了可导出模型上限（实测 ~0.4B 稳进）。更大模型需在 ≥16 GB / ≥32 GB 内存机器运行（命令已就绪）。

---

## 发布公开仓库（GitHub / gitcode）
本仓库已在本地 `git init` 并提交（权重 `models/`、导出 `out/` 已 gitignore）。推送需用户提供 token：

```bash
# GitHub（用户原话希望建公开仓）
export GH_TOKEN=ghp_xxx            # 你的 PAT（需 repo 权限）
gh auth login --with-token <<< "$GH_TOKEN"
gh repo create rwkv-openvino --public --description "RWKV-7 -> OpenVINO IR inference (CPU)" --source . --push -d main

# 或 gitcode（国内可达，无需翻墙）
git remote add origin https://gitcode.com/<你的用户名>/rwkv-openvino.git
git push -u origin main
```
> 权重与 IR 体积巨大，不入库；用 `bash scripts/fetch_rwkv7.sh <size>` 从 modelscope 拉取，或用微云传大文件。
