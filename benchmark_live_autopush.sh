#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

REMOTE="${REMOTE:-origin}"
REMOTE_BRANCH="${REMOTE_BRANCH:-codex/benchmark-live-20260310}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_common_baselines_three_stocks}"
STATUS_DIR="${STATUS_DIR:-autopush_logs}"
INTERVAL_SEC="${INTERVAL_SEC:-180}"
PID_FILE="${STATUS_DIR}/benchmark_live_autopush.pid"
RUN_LOG="${STATUS_DIR}/benchmark_live_autopush_runner.log"
STATUS_FILE="${STATUS_DIR}/benchmark_live_status.log"

mkdir -p "$STATUS_DIR"
echo $$ > "$PID_FILE"

snapshot_status() {
    {
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "Branch(local): $(git branch --show-current 2>/dev/null || true)"
        echo "HEAD(local): $(git rev-parse --short HEAD 2>/dev/null || true)"
        echo "Remote branch: ${REMOTE_BRANCH}"
        echo
        echo "Running benchmark processes:"
        ps -Ao pid,etime,%cpu,%mem,command | rg 'benchmark_paper_baselines_vs_old_new.py|benchmark_common_baselines_three_stocks' || true
        echo
        echo "Output files:"
        find "$OUTPUT_DIR" -maxdepth 3 -type f 2>/dev/null | sort || true
        echo
        echo "Recent git status for tracked benchmark artifacts:"
        git status --short -- "$OUTPUT_DIR" "$STATUS_FILE" 2>/dev/null || true
        echo "----"
    } > "$STATUS_FILE"
}

commit_and_push_if_changed() {
    git add -f -- "$OUTPUT_DIR" 2>/dev/null || true
    git add -f -- "$STATUS_FILE"

    if git diff --cached --quiet -- "$OUTPUT_DIR" "$STATUS_FILE"; then
        echo "[info] no new benchmark artifact changes" >> "$RUN_LOG"
        return 0
    fi

    local stamp
    stamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    git commit -m "Update live benchmark snapshot (${stamp})" -- "$OUTPUT_DIR" "$STATUS_FILE" >> "$RUN_LOG" 2>&1 || true
    git push "$REMOTE" "HEAD:refs/heads/${REMOTE_BRANCH}" >> "$RUN_LOG" 2>&1
}

echo "[info] starting live benchmark autopush loop at $(date '+%Y-%m-%d %H:%M:%S %z')" >> "$RUN_LOG"
echo "[info] remote=${REMOTE} branch=${REMOTE_BRANCH} output_dir=${OUTPUT_DIR}" >> "$RUN_LOG"

while true; do
    snapshot_status
    commit_and_push_if_changed
    sleep "$INTERVAL_SEC"
done
