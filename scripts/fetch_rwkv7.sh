#!/bin/bash
# 从 modelscope 镜像下载 RWKV-7 G1/G1i 权重（支持断点续传 + 真实体积完成判据）。
# 用法: bash scripts/fetch_rwkv7.sh <size>     size ∈ 0.1b 0.4b 1.5b 2.9b 7.2b 13.3b
set -u
SIZE="${1:-0.1b}"
ROOT=/workspace/rwkv-openvino
URL="https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1i-${SIZE}-20260805-ctx16384.pth"
# 0.1b / 0.4b 是 g1d 旧命名
if [ "$SIZE" = "0.1b" ]; then URL="https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1d-0.1b-20260129-ctx8192.pth"; fi
if [ "$SIZE" = "0.4b" ]; then URL="https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1d-0.4b-20260210-ctx8192.pth"; fi
OUT="$ROOT/models/$(basename "$URL")"
mkdir -p "$ROOT/models"
TOTAL=$(curl -sL -r 0-0 -D - -o /dev/null --max-time 60 "$URL" 2>/dev/null | grep -i content-range | grep -oE '/[0-9]+' | tr -d '/')
echo "[fetch] $SIZE -> $OUT  (total=$((TOTAL/1000000)) MB)"
for i in $(seq 1 40); do
  CUR=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$CUR" -ge "$TOTAL" ]; then echo "[fetch] done $((CUR/1000000)) MB"; break; fi
  echo "[fetch] attempt $i: $((CUR/1000000))/$((TOTAL/1000000)) MB"; curl -sSL -C - -m 3000 -o "$OUT" "$URL"
done
echo "[fetch] file: $OUT"
