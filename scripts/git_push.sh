#!/usr/bin/env bash
# 高频推送助手（fastgit 代理专用）
# 代理已知 bug：无法更新/推进任何已存在 ref（main、agent-work 等），只允许【新建】分支。
# 策略：先试 main（若代理恢复则自动生效），失败就新建分支 agent-<epoch>，保证提交不丢。
# 用法: bash scripts/git_push.sh [commit-message]
set -u
cd "$(dirname "$0")/.." || exit 1
MSG="${1:-chore: push via helper}"

git add -A
if ! git diff --cached --quiet; then
  git commit -m "$MSG" || exit 1
fi

if git push origin main:refs/heads/main 2>/dev/null; then
  echo "OK  -> main"
  exit 0
fi

TS=$(date +%s)
echo "main 被代理锁 bug 卡住 -> 新建分支 agent-$TS（代理唯一可用的写入通道）"
git push origin main:refs/heads/agent-$TS || {
  echo "FATAL: agent-$TS 也失败，请检查网络/代理"; exit 1;
}
echo "OK  -> agent-$TS（最新提交所在分支，WORKLOG 已说明）"
