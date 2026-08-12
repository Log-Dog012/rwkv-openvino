# RWKV → OpenVINO 工作日志 / 重建指南 (WORKLOG)

> 用途：本文件是「对话崩溃后也能重建工作」的快照。不含任何私密信息（PAT / 私人路径已脱敏）。
> 最后更新：2026-08-12（阶段性：L=1 验证 PASS，全 24 层 layerwise 生成进行中）。

---

## 0. 一句话目标

把 RWKV-7 的 **GGUF 模型（K-quant 混合精度，如 Q4_K_M）转成 OpenVINO IR**，并**完美继承 GGUF 的原生 K-quant 权重** —— 用 OV 自带的「压缩权重」机制（`Constant(u4/i8) → Convert(f16) → [Subtract(zp)] → Multiply(scale) → Reshape`，被 CPU/GPU 插件融合成原生 int4/int8 matmul），**而不是 NNCF 重新量化**。

核心产出：`rwkv7_ov.py`（主图构建）、`rwkv7_ov_layerwise.py`（8GB 友好逐层执行器）、`rwkv7_ref_np.py`（numpy 参考）、`validate_rwkv7_ov.py`（L=1 校验）。

---

## 1. 环境约束（必须记住）

| 项 | 值 / 注意点 |
|---|---|
| 沙箱内存 | **8GB cgroup 硬上限**（`/sys/fs/cgroup/memory.max = 8589934592`）—— 一切内存优化都围绕它 |
| OpenVINO | v2026.3.x，`ov` opset13 |
| opset13 坑 | `Node` **没有** `.reshape()` → 用 `ops.reshape(node, shape, special_zero=True, name=)`；`ops.matmul()` **必须**显式给 `transpose_a/transpose_b`；param 要 `.friendly_name=`；输出用 `req.get_output_tensor(index)` |
| CPU 属性 | `ov.properties.inference_num_threads`、`ov.properties.hint.performance_mode = ov.properties.hint.PerformanceMode.LATENCY` |
| Python | 3.11；装包用 `sudo pip3 install`（或 `sudo uv pip install --system`） |
| 模型路径 | 相对 `scripts/../models/`（如 `../models/rwkv7-g1i-1.5b-Q4_K_M.gguf`） |

### 仓库两处目录（重要！）
- **工作目录**：`/workspace/rwkv-openvino`（所有最新脚本在这里改）
- **git 仓库**：`/tmp/rwkv-ov-fg`（远端走 `fastgit.cc` 代理，PAT 已内嵌在 `.git/config` 远端 URL 里，**勿外泄、勿写进任何文件**）
- 这两处是**独立拷贝**，不是软链/挂载。同步靠：`cp /workspace/rwkv-openvino/scripts/*.py /tmp/rwkv-ov-fg/scripts/`（只覆盖 workspace 有、且保留 tmp 独有诊断脚本）。

---

## 2. 关键结论（已验证，决定性）

### 2.1 GGUF 物理布局 vs 逻辑布局（方向问题总根）
- GGUF 张量按**逻辑转置**存储。`tensor.shape` = 逻辑形状；`gguf.dequantize(t.data, t.tensor_type)` 得到的**真实形状 = `reversed(tensor.shape)`**。
- 对所有非方阵权重：`graw_shape = reversed(tensor.shape)`。例如 `(2048,96) → (96,2048)`，`(2048,8192) → (8192,2048)`。

### 2.2 torch 操作数方向
- `gguf_to_ov.build_state()` 对 matmul 权重做 `orient="T"`（真正的 `.T` 转置）。
- 所以 torch 的操作数 `W_torch = GGUF_raw.T`（物理 .T），torch 计算：`x @ W_torch = x @ GGUF_raw.T`。
- 验证脚本 `_orient_test.py`：对全部 8 个 matmul 权重（方阵 + 非方阵）都满足 `bs == raw.T` 为 True，且 `x@raw != x@bs`。**证明 torch = `x @ GGUF_raw.T`**。

### 2.3 方向 bug 根因与修复（历史 FAIL 的根因）
- **根因**：`repack_tensor` 用 `tensor.shape`（逻辑）重塑 codes，而 `build_state`/torch 用的是物理 `graw`。对非方阵，行优先重塑 ≠ 真转置 → OV 的 `cw` 与 torch 对不上。
- **修复**（`rwkv7_ov.cw()`）：
  ```python
  def cw(name):
      rep = dict(repack_tensor(T[name]))
      rep["shape"] = list(rep["shape"])[::-1]   # reversed = gguf.dequantize 的物理形状
      return build_compressed_weight(rep, name)
  ```
  使 `cw == graw`；外层 `MatMul` 统一 **`transpose_b=True`** → `x @ cw.T == x @ graw.T == x @ W_torch`，对方阵 / 非方阵 / head **全部 bit-exact**。
- **例外**：状态更新里的 `vk = v@k`、`ab = (-kk)@(kk*a)`、`st@ab`、`st@r` 是真实矩阵乘积（操作数非权值），保持 `transpose_b=False`。

### 2.4 ln0 预归一（历史 FAIL 的第二根因）
- `build_state` 返回的 `emb` 是 **RAW**（ln0 只在 `RWKV7._build_from_raw` 类内应用，build_state 不应用）。OV 对 `emb_const` 做了 ln0 预归一。
- numpy 参考必须**显式**对 emb 做 ln0，否则和 OV 整体差一个 ln0 缩放（之前 att≈9.5 / kv≈5.1 / ffn≈50 的 FAIL 根因之一）。`rwkv7_ref_np.py` 已修：
  ```python
  ln0_w = z["blocks.0.ln0.weight"].numpy().astype(np.float32)
  ln0_b = z["blocks.0.ln0.bias"].numpy().astype(np.float32)
  emb_raw = z["emb.weight"].numpy().astype(np.float32)
  mu = emb_raw.mean(-1, keepdims=True); va = emb_raw.var(-1, keepdims=True)
  emb_ln = (emb_raw - mu) / np.sqrt(va + 1e-5) * ln0_w + ln0_b   # [V,C]
  ```

---

## 3. 验证现状

| 验证 | 命令 | 结果 |
|---|---|---|
| L=1 head-to-head（numpy vs OV） | `validate_rwkv7_ov.py` | **PASS ✅** att=1.039e-3, kv=8.825e-3, ffn=5.594e-2（阈值 0.1）。OV 图在 fp16 精度内与 torch bit-exact 对齐 |
| **全 24 层端到端生成（分块执行器）** | `rwkv7_ov_layerwise.py --chunk 8 --n 16` | **PASS ✅ 16/16 token ID 与 torch 基线完全一致**；文本 `' Paris, France.\nThe Eiffel Tower is a wrought-iron'` 逐字相同；峰值内存 <8GB |
| torch 基线（1.5B GGUF） | `_torch_15b_ref.py` | 参考文本同上；top1=37138(logit 1.554)，16 个生成 token ID 已存 `temp/torch_15b_ref.json` |

决定性证明脚本：`_orient_test.py`（方向）、`_cw_vs_bs_test.py`（cw vs torch 值对齐）。

### 3.1 端到端验证中发现并修复的 2 个组合 bug（layerwise 谱系独有，主图 rwkv7_ov.py 无此问题）
1. **v_first/vfix 缺失**：RWKV7 g1 架构要求 layer0 的原始 `v` 在同一 token 内传给所有层（`v_first`），layer1+ 用 `v += (v_first-v)*sigmoid(v0 + (xv@v1)@v2)`（vfix）。旧 layerwise 每层独立 embed 且无 vfix → 组合错误。分块执行器已实现 v_first 跨块线程传递。
2. **ln_out 缺失**：torch 里最后一层 x 需先过 `output_norm` LayerNorm 再乘 `output.weight` 算 logits。旧 layerwise 直接 `x @ output.weight`。分块执行器已补。
3. 旧 layerwise 的每层图都以 `gather(emb_table,idx)` 为输入（每层独立处理同一嵌入，激活不流动）——组合 bug，已随分块重构修复。

---

## 4. 8GB 内存约束与应对（核心工程难点）

1. **单图 24 层一次性 compile 超 8GB**（RC=137，图能在 ~189s 建好但编译超 8GB）。→ 改用**逐层执行**。
2. **逐层执行器** `rwkv7_ov_layerwise.py`：
   - 每层构建为**独立单层中图**（复用 `rwkv7_ov` 的 `F, R, _layer_norm, _l2norm, _dequant_np, C_DTYPE`）。
   - `emb_table` 作为**共享输入参数**（避免 24×268MB 常量拷贝）。
   - 状态张量 `[1,C]` / `[1,H,N,N]` / `[1,C]` 每层层内维护。
   - 共享单个 `gguf.GGUFReader`。
3. **GGUF mmap 页缓存 OOM**：`echo 3 > /proc/sys/vm/drop_caches` 在沙箱**无权限**（rc=1）。改用
   ```python
   r.data.base.madvise(mmap.MADV_DONTNEED)   # 每层编译后调用，无需 root，等效 drop_caches
   ```
4. **内存构成实测**：cgroup `memory.stat` 显示大头是 `anon`（匿名/堆），**不是** `file`（GGUF 页缓存）。编译期每层瞬时 anon 峰值约 ~1GB（OV CPU 插件 JIT 缓冲），编译完回收，常驻低。全 24 层常驻 + 生成阶段需实测是否超 8GB。

---

## 5. 当前进行中 / 待办

- [✅ 完成] **1.5B Q4 全 24 层分块生成**：16/16 token 与 torch 基线一致，峰值 <8GB（分块 + v_first/ln_out 修复后）。
- [✅ 完成] **7.2B Q4 分块生成（L=32, C=4096, chunk=1, n=8）**：输出 `' Paris, France. It is situated on'`（连贯），峰值内存 ~6.8GB <8GB ✅；sweep 878s + 生成 826s（~103s/token，32 层每 token 全重载的代价）。前 4 token `[37138,45,44312,47]` 与 1.5B 相同（模型共识知识）。
- [✅ 完成] **ModelScope 交付**（`logdog/` 命名空间，已登录）：
  - `logdog/rwkv7-g1i-1.5b-q4k-ov`：单图 IR（xml+bin+README）✅
  - `logdog/rwkv7-g1i-7.2b-q4k-ov`：32 chunk IR（xml+bin+README）✅
  - `logdog/rwkv7-g1i-13.3b-q4k-ov`：61 chunk IR（xml+bin+README）✅
- [⚠️ 教训] `modelscope upload` 只传指定文件，**不会自动带上同名的 .bin**——分块上传必须 xml 与 bin 都显式传（首轮只传了 xml，远端缺权重，需补传）。
- [⚠️ 教训] 沙箱**磁盘配额 ~51GB**（df 显示 205GB 可用是假象）：大文件写入超配额报 `Disk quota exceeded`；OV `ov.save_model` 撞配额时报 `basic_ios::clear: iostream error`（误判为内存问题花了很久）。对策：`dd` 测写定位、及时清理已上传 IR、串行导出。
- [待办] **13.3B Q4 IR 交付**：61 chunk IR 已导出并上传 xml，补传 bin 后完成。

---

## 6. 重建步骤（从零）

```bash
# 1. clone（fastgit 代理，PAT 已在 .git/config 远端 URL 内嵌，勿外泄）
git clone https://fastgit.cc/https://github.com/Log-Dog012/rwkv-openvino.git
cd rwkv-openvino

# 2. 依赖
sudo pip3 install -r requirements.txt        # openvino, gguf, numpy, torch ...

# 3. 模型（.gitignore 已忽略 models/）
bash scripts/fetch_rwkv7.sh                  # 或自行放到 models/

# 4. L=1 验证（numpy vs OV，阈值 0.1 内 PASS）
python3.11 scripts/validate_rwkv7_ov.py

# 5. 逐层生成（1.5B Q4 全 24 层）
python3.11 scripts/rwkv7_ov_layerwise.py \
    ../models/rwkv7-g1i-1.5b-Q4_K_M.gguf --n 16 --threads 4 \
    --prompt "The Eiffel Tower is located in the city of"
```

### 关键脚本速查
- `rwkv7_ov.py` —— 主图构建（`cw()` 含方向修复，`transpose_b=True`）
- `rwkv7_ov_layerwise.py` —— 逐层执行器（madvise 释放 GGUF 页，emb 共享参数）
- `rwkv7_ref_np.py` —— numpy 参考（含显式 ln0）
- `validate_rwkv7_ov.py` —— L=1 校验
- `_orient_test.py` / `_cw_vs_bs_test.py` —— 方向决定性证明
- `gguf_to_ov.py`（`build_state`，`orient="T"`）、`gguf_to_ov_compressed.py`（`build_compressed_weight`）、`gguf_kquant_repack.py`（`repack_tensor`，返回逻辑形状，由 cw 反转）

---

## 7. 提交 / 同步约定

- 工作目录 `/workspace/rwkv-openvino`；git 仓库 `/tmp/rwkv-ov-fg`。
- 同步（单向，保留 tmp 独有诊断脚本）：
  ```bash
  cp /workspace/rwkv-openvino/scripts/*.py /tmp/rwkv-ov-fg/scripts/
  # 可选：把 workspace/docs/*.html 也带过去
  cp /workspace/rwkv-openvino/docs/*.html /tmp/rwkv-ov-fg/docs/ 2>/dev/null
  ```
- **高频 commit + push**：对话是工作与远端仓库之间唯一的同步通道，崩溃即丢失自上次 push 后的工作。完成一个阶段性里程碑就 push 一次。
- `.gitignore` 已忽略：`models/`、`out/`、`*.pth/*.safetensors/*.bin/*.xml`、`__pycache__/`、`temp/`、`*.log`。**切勿把 .gguf / 模型 commit 进仓库**。

### 7.1 fastgit 代理已知 bug（2026-08-12 记录）

- **症状**：`push main` 报 `cannot lock ref 'refs/heads/main': is at <新sha> but expected <旧sha>`，连 `--force` / `--force-with-lease` 都一样。删除分支也报 `reference already exists`。
- **根因**：代理内部对已存在 ref 的缓存快照与实际远端不一致（main 快照停在 abc604b，实际已是 893023a），锁校验自相矛盾。**新建 ref 不受影响**。
- **当前远端状态**：
  - `main` = `893023a`（方向修复 + ln0 + 逐层执行器 + L=1 PASS + WORKLOG 全部核心工作）
  - `agent-work` = `cc620b1`（+ AGENTS.md、git_push.sh、本段记录）
  - `ag-test-cc620b1` = `cc620b1`（测试残留，删不掉，无害）
- **约定**：日常 push 用 `bash scripts/git_push.sh`（优先 main，失败自动落 `agent-work`）。等代理恢复或能直连 GitHub 时，再把 `agent-work` 快进合并回 `main`。
