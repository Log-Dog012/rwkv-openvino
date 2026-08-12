# AGENTS.md — 接手本仓库的开工指引

本仓库：**RWKV-7 GGUF → OpenVINO IR，原生继承 GGUF K-quant（Q4_K_M 等）压缩权重**（复用 OV 的 int4/int8 融合 matmul，不是 NNCF 重新量化）。

## 开工前必读

**`docs/WORKLOG.md`** —— 工作日志 + 崩溃重建指南。包含全部关键结论：
权重方向修复（reversed repack + `transpose_b=True`）、ln0 预归一、8GB cgroup 内存对策（madvise 释放 GGUF 页）、L=1 验证结果、进行中任务与待办。**先读它再动手，别重复踩坑。**

## 关键约定

- 核心脚本在 `scripts/`：
  - `rwkv7_ov.py` —— 主图构建（`cw()` 含方向修复）
  - `rwkv7_ov_layerwise.py` —— 逐层执行器（8GB 友好，madvise 管理 GGUF 页缓存）
  - `rwkv7_ref_np.py` —— numpy 参考实现（含显式 ln0）
  - `validate_rwkv7_ov.py` —— L=1 校验（阈值 0.1 内 PASS）
  - `_orient_test.py` / `_cw_vs_bs_test.py` —— 权重方向决定性证明
- 模型**不入库**（`.gitignore` 忽略 `models/`、`*.gguf` 相关大文件），用 `bash scripts/fetch_rwkv7.sh` 下载。
- 沙箱 **8GB cgroup 硬上限**，所有内存优化围绕它展开。
- **高频 commit + push**：本对话是工作与远端之间的唯一同步通道，中断即丢失未 push 的工作。

## 沙箱开发约定（本机特有）

- 工作目录：`/workspace/rwkv-openvino`（改代码在这）
- git 仓库：`/tmp/rwkv-ov-fg`（fastgit 代理远端，PAT 内嵌于 `.git/config`）
- 同步：`cp /workspace/rwkv-openvino/scripts/*.py /tmp/rwkv-ov-fg/scripts/`（保留 tmp 独有诊断脚本），随后 commit + push。

## 红线

- 勿把 PAT / 任何私密信息写入本仓库（含 commit message、文档）。
- 勿把 `.gguf` / 模型权重 commit 进仓库。
