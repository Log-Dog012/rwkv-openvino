#!/bin/bash
cd "c:/Users/Mcsof/a/rwkv/rwkv-openvino"
source scripts/ov_env.sh
PY="C:/Users/Mcsof/miniconda3/envs/AI/python.exe"
GGUF="C:/Users/Mcsof/models/gguf/rwkv7-g1i-13.3b-Q4_K_M.gguf"
echo "START $(date +%H:%M:%S)"
"$PY" -u scripts/rwkv7_ov.py "$GGUF" --no-compile --out out/rwkv7-13.3b-q4k_ov.xml 2>&1
echo "EXIT $? END $(date +%H:%M:%S)"
touch logs/build_done.flag
