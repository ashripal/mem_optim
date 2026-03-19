#!/usr/bin/env bash
set -euo pipefail

# =========================
# MemArch benchmark sweep
# =========================
#
# Recommended usage:
#   chmod +x run_memarch_overnight.sh
#   ./run_memarch_overnight.sh
#
# Logs will be written under:
#   artifacts/benchmark_runs/memarch_overnight_logs/
#
# Notes:
# - Default MAX_INPUT_TOKENS=2048 for a stronger but still practical evaluation.
# - Increase to 4096 only if your machine has already handled 2048 comfortably.
# - Each run uses its own SQLite store path to avoid contamination across modes.
# - clear_disk_store_before_run is enabled for clean comparisons.

REPO_ROOT="/Users/ashripal/mem_optim/thesis_code"
cd "$REPO_ROOT"

TIER2_REPO="/Users/ashripal/mem_optim/thesis_code/baseline_old/tier2_disk/longbench_repo/data"

# -------------------------
# Global benchmark settings
# -------------------------
OUT_ROOT="artifacts/benchmark_runs/memarch"
LOG_ROOT="artifacts/benchmark_runs/memarch_overnight_logs"
mkdir -p "$LOG_ROOT"

MODEL_ID="microsoft/Phi-3-mini-128k-instruct"

# Stronger than your smoke test, but still realistic for overnight CPU runs.
MAX_INPUT_TOKENS=2048
MAX_NEW_TOKENS=64

# Increase this if your machine comfortably finishes overnight.
MAX_EXAMPLES=100

# Memory settings
RAM_CAPACITY_ITEMS=256

# Semantic settings
SEMANTIC_THRESHOLD_CONTEXT=0.85
SEMANTIC_THRESHOLD_BYPASS=0.95
MAX_SEMANTIC_CANDIDATES=5
EMBEDDING_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"

# Identity settings
USER_ID="user_a"
SESSION_ID="session_a"

# Optional task filter:
#   ""        = all tasks
#   "2wikimqa" = only 2wikimqa
TASK_GLOB=""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  --tier2_repo "$TIER2_REPO"
  --out_root "$OUT_ROOT"
  --task_glob "$TASK_GLOB"
  --max_examples "$MAX_EXAMPLES"
  --model_id "$MODEL_ID"
  --max_input_tokens "$MAX_INPUT_TOKENS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --cpu_fallback_on_long
  --ram_capacity_items "$RAM_CAPACITY_ITEMS"
  --user_id "$USER_ID"
  --session_id "$SESSION_ID"
  --write_summary_json
)

run_case () {
  local benchmark_name="$1"
  shift

  local log_path="${LOG_ROOT}/${benchmark_name}_${TIMESTAMP}.log"

  echo "============================================================"
  echo "Starting: ${benchmark_name}"
  echo "Log: ${log_path}"
  echo "Started at: $(date)"
  echo "============================================================"

  python scripts/run_memarch_benchmark.py \
    --benchmark_name "$benchmark_name" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$log_path"

  echo
  echo "Finished: ${benchmark_name} at $(date)"
  echo
}

# ============================================================
# 1) Exact-only runs
# ============================================================

run_case "memarch_exact_only_replay_once_${TIMESTAMP}" \
  --mode replay_once \
  --retrieval_mode exact_only \
  --disk_store_path "artifacts/benchmark_runs/memarch/memory/memarch_exact_only_replay_once_${TIMESTAMP}.sqlite" \
  --clear_disk_store_before_run

run_case "memarch_exact_only_cache_pressure_${TIMESTAMP}" \
  --mode cache_pressure \
  --retrieval_mode exact_only \
  --disk_store_path "artifacts/benchmark_runs/memarch/memory/memarch_exact_only_cache_pressure_${TIMESTAMP}.sqlite" \
  --clear_disk_store_before_run

# ============================================================
# 2) Semantic-context runs (semantic assist, no bypass)
# ============================================================

run_case "memarch_semantic_context_replay_once_${TIMESTAMP}" \
  --mode replay_once \
  --retrieval_mode semantic_context \
  --semantic_enabled \
  --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT" \
  --semantic_threshold_bypass 1.01 \
  --max_semantic_candidates "$MAX_SEMANTIC_CANDIDATES" \
  --embedding_model_id "$EMBEDDING_MODEL_ID" \
  --disk_store_path "artifacts/benchmark_runs/memarch/memory/memarch_semantic_context_replay_once_${TIMESTAMP}.sqlite" \
  --clear_disk_store_before_run

run_case "memarch_semantic_context_cache_pressure_${TIMESTAMP}" \
  --mode cache_pressure \
  --retrieval_mode semantic_context \
  --semantic_enabled \
  --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT" \
  --semantic_threshold_bypass 1.01 \
  --max_semantic_candidates "$MAX_SEMANTIC_CANDIDATES" \
  --embedding_model_id "$EMBEDDING_MODEL_ID" \
  --disk_store_path "artifacts/benchmark_runs/memarch/memory/memarch_semantic_context_cache_pressure_${TIMESTAMP}.sqlite" \
  --clear_disk_store_before_run

# ============================================================
# 3) Semantic-bypass runs
# ============================================================

run_case "memarch_semantic_bypass_replay_once_${TIMESTAMP}" \
  --mode replay_once \
  --retrieval_mode semantic_bypass \
  --semantic_enabled \
  --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT" \
  --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS" \
  --max_semantic_candidates "$MAX_SEMANTIC_CANDIDATES" \
  --embedding_model_id "$EMBEDDING_MODEL_ID" \
  --disk_store_path "artifacts/benchmark_runs/memarch/memory/memarch_semantic_bypass_replay_once_${TIMESTAMP}.sqlite" \
  --clear_disk_store_before_run

run_case "memarch_semantic_bypass_cache_pressure_${TIMESTAMP}" \
  --mode cache_pressure \
  --retrieval_mode semantic_bypass \
  --semantic_enabled \
  --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT" \
  --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS" \
  --max_semantic_candidates "$MAX_SEMANTIC_CANDIDATES" \
  --embedding_model_id "$EMBEDDING_MODEL_ID" \
  --disk_store_path "artifacts/benchmark_runs/memarch/memory/memarch_semantic_bypass_cache_pressure_${TIMESTAMP}.sqlite" \
  --clear_disk_store_before_run

echo "============================================================"
echo "All MemArch benchmark runs completed at $(date)"
echo "Logs: $LOG_ROOT"
echo "Artifacts root: $OUT_ROOT"
echo "============================================================"