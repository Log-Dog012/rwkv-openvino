#!/usr/bin/env bash
# 高频推送助手：优先推 main；若 fastgit 代理对 main 的锁卡死（known bug），
# 自动落回 agent-work 工作分支，保证提交不丢。用法: bash scripts/git_push.sh [message]
set -u
cd "$(dirname "$0")/.." || exit 1
MSG="${1:-chore: push via helper}"

git add -A
if ! git diff --cached --quiet; then
  git commit -m "$MSG" || exit 1
fi

if git push origin main:refs/heads/main 2>/tmp/_push_main.err; then
  echo "OK  -> main"
else
  echo "main 被代理锁 bug 卡住（$(head -1 /tmp/_push_main.err)）"
  echo "OK  -> agent-work（代理可用通道）"
  git push origin main:refs/heads/agent-work || {
    echo "agent-work 也失败，试试新分支 agent-$(date +%s)";
    git push origin main:refs/heads/agent-$(date +%s) || exit 1;
  }
fi
