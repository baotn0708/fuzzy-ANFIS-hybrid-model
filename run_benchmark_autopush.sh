#!/usr/bin/env bash
set -euo pipefail

# Run the three-stock benchmark in the background-friendly way and
# automatically commit/push only the generated result directory.
#
# Usage:
#   ./run_benchmark_autopush.sh
#   OUTPUT_DIR=my_dir BRANCH=codex/bench-20260310 ./run_benchmark_autopush.sh
#
# Optional env vars:
#   OUTPUT_DIR        default: benchmark_common_baselines_three_stocks
#   BRANCH            default: current branch
#   REMOTE            default: origin
#   STOCKS            default: AMZN,JPM,TSLA
#   MODELS            default: original,arima,svr,knn,random_forest,xgboost,mlp,lstm,gru,cnn_lstm,transformer
#   SEEDS             default: 42
#   EPOCHS            default: 12
#   ORIGINAL_EPOCHS   default: 150
#   SOURCE_MODE       default: raw

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-benchmark_common_baselines_three_stocks}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-$(git branch --show-current 2>/dev/null || true)}"
STOCKS="${STOCKS:-AMZN,JPM,TSLA}"
MODELS="${MODELS:-original,arima,svr,knn,random_forest,xgboost,mlp,lstm,gru,cnn_lstm,transformer}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-12}"
ORIGINAL_EPOCHS="${ORIGINAL_EPOCHS:-150}"
SOURCE_MODE="${SOURCE_MODE:-raw}"
LOG_DIR="${ROOT_DIR}/autopush_logs"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/benchmark_autopush_${STAMP}.log"

mkdir -p "$LOG_DIR"

echo "[info] root: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "[info] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[info] remote: $REMOTE" | tee -a "$LOG_FILE"
echo "[info] branch: ${BRANCH:-<empty>}" | tee -a "$LOG_FILE"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[error] not inside a git repository" | tee -a "$LOG_FILE"
    exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
    echo "[error] git remote '$REMOTE' is not configured" | tee -a "$LOG_FILE"
    echo "[hint] run: git remote add $REMOTE <your-repo-url>" | tee -a "$LOG_FILE"
    exit 1
fi

if [[ -z "${BRANCH}" ]]; then
    echo "[error] could not determine current branch" | tee -a "$LOG_FILE"
    exit 1
fi

if ! git config user.name >/dev/null; then
    echo "[error] git user.name is not configured" | tee -a "$LOG_FILE"
    exit 1
fi

if ! git config user.email >/dev/null; then
    echo "[error] git user.email is not configured" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[info] starting benchmark..." | tee -a "$LOG_FILE"

python3 benchmark_paper_baselines_vs_old_new.py \
    --stocks "$STOCKS" \
    --models "$MODELS" \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    --original-epochs "$ORIGINAL_EPOCHS" \
    --source-mode "$SOURCE_MODE" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "[info] benchmark finished, preparing git commit..." | tee -a "$LOG_FILE"

git add -- "$OUTPUT_DIR"

if git diff --cached --quiet -- "$OUTPUT_DIR"; then
    echo "[info] no changes detected in $OUTPUT_DIR, nothing to commit" | tee -a "$LOG_FILE"
    exit 0
fi

COMMIT_MSG="Add benchmark results for ${STOCKS} (${STAMP})"
git commit -m "$COMMIT_MSG" -- "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
git push "$REMOTE" "$BRANCH" 2>&1 | tee -a "$LOG_FILE"

echo "[info] push complete" | tee -a "$LOG_FILE"
