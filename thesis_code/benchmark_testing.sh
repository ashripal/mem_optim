#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Fair benchmark sweep: Baseline vs MemArch (Qwen 0.5B only)
# ============================================================
#
# Design goals:
# - same model / device / dtype / token limits across both systems
# - same workload file for each paired comparison
# - explicit total_requests so runs are directly comparable
# - one consistent MemArch policy across workloads
# - no 1.5B model for now
# ============================================================

# ------------------------------------------------------------
# Path / environment configuration
# ------------------------------------------------------------

make_pair_names() {
  local group_label="$1"
  BASE_NAME="baseline__${group_label}"
  MEM_NAME="memarch__${group_label}"
}

REPO_ROOT="${REPO_ROOT:-/home/sravani/mem_optim/thesis_code}"
cd "$REPO_ROOT"

# Main data roots
SQUAD_ROOT="${SQUAD_ROOT:-$REPO_ROOT/baseline_old/tier2_disk/data/squad_clean}"
TIER2_REPO="${TIER2_REPO:-$SQUAD_ROOT}"

# Input workloads
WORKLOAD_EXACT="${WORKLOAD_EXACT:-$SQUAD_ROOT/workload_exact.jsonl}"
WORKLOAD_PARAPHRASE="${WORKLOAD_PARAPHRASE:-$SQUAD_ROOT/workload_paraphrase.jsonl}"
WORKLOAD_FAMILY_CLUSTERED="${WORKLOAD_FAMILY_CLUSTERED:-$SQUAD_ROOT/workload_family_clustered.jsonl}"

# Models
MODEL_05B="${MODEL_05B:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-$REPO_ROOT/models/all-MiniLM-L6-v2}"

# Scripts
BASELINE_SCRIPT="${BASELINE_SCRIPT:-scripts/run_baseline_benchmark.py}"
MEMARCH_SCRIPT="${MEMARCH_SCRIPT:-scripts/run_memarch_benchmark.py}"

# Output roots
OUT_ROOT_BASELINE="${OUT_ROOT_BASELINE:-artifacts/benchmark_runs/baseline}"
OUT_ROOT_MEMARCH="${OUT_ROOT_MEMARCH:-artifacts/benchmark_runs/memarch}"
LOG_ROOT="${LOG_ROOT:-artifacts/benchmark_runs/testing_logs}"
MEMORY_ROOT="${MEMORY_ROOT:-artifacts/benchmark_runs/memory}"

mkdir -p "$LOG_ROOT" "$MEMORY_ROOT"

# ------------------------------------------------------------
# Run toggles
# ------------------------------------------------------------
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_MEMARCH="${RUN_MEMARCH:-1}"

# ------------------------------------------------------------
# Auto-detect CUDA if DEVICE not explicitly set
# ------------------------------------------------------------
if [[ -z "${DEVICE:-}" ]]; then
  if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

DTYPE="${DTYPE:-float32}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-$DEVICE}"

# ------------------------------------------------------------
# Shared generation settings
# ------------------------------------------------------------
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# ------------------------------------------------------------
# Workload sizes
# These are explicit so baseline and MemArch process the same
# number of requests for each paired comparison.
# ------------------------------------------------------------
SEEDS_COLD="${SEEDS_COLD:-8}"
REQS_COLD="${REQS_COLD:-8}"

SEEDS_EXACT="${SEEDS_EXACT:-8}"
REQS_EXACT="${REQS_EXACT:-16}"

SEEDS_PARAPHRASE="${SEEDS_PARAPHRASE:-16}"
REQS_PARAPHRASE="${REQS_PARAPHRASE:-32}"

SEEDS_FAMILY="${SEEDS_FAMILY:-16}"
REQS_FAMILY="${REQS_FAMILY:-64}"

# ------------------------------------------------------------
# MemArch settings (single consistent policy across workloads)
# ------------------------------------------------------------
RAM_CAPACITY_ITEMS="${RAM_CAPACITY_ITEMS:-256}"
MAX_SEMANTIC_CANDIDATES="${MAX_SEMANTIC_CANDIDATES:-5}"

# Use one stable policy for all paired runs:
# - exact hits still work naturally
# - semantic bypass supports paraphrase reuse
# - same-document and evidence-support gates kept on
SEMANTIC_THRESHOLD_CONTEXT="${SEMANTIC_THRESHOLD_CONTEXT:-0.55}"
SEMANTIC_THRESHOLD_BYPASS="${SEMANTIC_THRESHOLD_BYPASS:-0.90}"
SEMANTIC_BYPASS_MIN_MARGIN="${SEMANTIC_BYPASS_MIN_MARGIN:-0.02}"
SEMANTIC_BYPASS_MAX_ANSWER_WORDS="${SEMANTIC_BYPASS_MAX_ANSWER_WORDS:-12}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_case() {
  local script_path="$1"
  local benchmark_name="$2"
  shift 2

  local log_path="${LOG_ROOT}/${benchmark_name}_${TIMESTAMP}.log"

  echo "============================================================"
  echo "Starting: ${benchmark_name}"
  echo "Script: ${script_path}"
  echo "Log: ${log_path}"
  echo "Started at: $(date)"
  echo "============================================================"

  python "$script_path" \
    --benchmark_name "$benchmark_name" \
    "$@" \
    2>&1 | tee "$log_path"

  echo
  echo "Finished: ${benchmark_name} at $(date)"
  echo
}

baseline_common_args() {
  local max_examples="$1"
  local total_requests="$2"

  BASELINE_ARGS=(
    --tier2_repo "$TIER2_REPO"
    --out_root "$OUT_ROOT_BASELINE"
    --model_id "$MODEL_05B"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --max_input_tokens "$MAX_INPUT_TOKENS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --max_examples "$max_examples"
    --total_requests "$total_requests"
    --write_summary_json
  )

  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    BASELINE_ARGS+=(--local_files_only)
  fi
}

memarch_common_args() {
  local max_examples="$1"
  local total_requests="$2"
  local disk_store_path="$3"

  MEMARCH_ARGS=(
    --tier2_repo "$TIER2_REPO"
    --out_root "$OUT_ROOT_MEMARCH"
    --model_id "$MODEL_05B"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --max_input_tokens "$MAX_INPUT_TOKENS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --max_examples "$max_examples"
    --total_requests "$total_requests"
    --ram_capacity_items "$RAM_CAPACITY_ITEMS"
    --disk_store_path "$disk_store_path"
    --clear_disk_store_before_run
    --embedding_model_id "$EMBEDDING_MODEL_ID"
    --embedding_device "$EMBEDDING_DEVICE"
    --max_semantic_candidates "$MAX_SEMANTIC_CANDIDATES"
    --retrieval_mode semantic_bypass
    --semantic_enabled
    --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT"
    --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS"
    --allow_semantic_bypass
    --require_same_document_for_semantic_bypass
    --require_evidence_support_for_semantic_bypass
    --semantic_bypass_min_margin "$SEMANTIC_BYPASS_MIN_MARGIN"
    --semantic_bypass_max_answer_words "$SEMANTIC_BYPASS_MAX_ANSWER_WORDS"
    --write_summary_json
  )

  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    MEMARCH_ARGS+=(--local_files_only --embedding_local_files_only)
  fi
}

# ------------------------------------------------------------
# Paired fair comparisons
# ------------------------------------------------------------

run_cold_pair() {
  local group_label="qwen05b__cold__exact__req${REQS_COLD}"
  make_pair_names "$group_label"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_COLD" "$REQS_COLD"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode cold
      --notes "Fair paired baseline cold run on exact workload"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_COLD" "$REQS_COLD" "${MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_EXACT"
      --mode cold
      --notes "Fair paired memarch cold run on exact workload"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_exact_pair() {
  local group_label="qwen05b__exact_reuse__req${REQS_EXACT}"
  make_pair_names "$group_label"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_EXACT" "$REQS_EXACT"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode mixed_reuse
      --notes "Fair paired baseline repeated exact-query workload"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_EXACT" "$REQS_EXACT" "${MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_EXACT"
      --mode exact_interleaved
      --notes "Fair paired memarch exact-interleaved workload"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_paraphrase_pair() {
  local group_label="qwen05b__paraphrase_reuse__req${REQS_PARAPHRASE}"
  make_pair_names "$group_label"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_PARAPHRASE" "$REQS_PARAPHRASE"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode mixed_reuse
      --notes "Fair paired baseline paraphrase workload"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_PARAPHRASE" "$REQS_PARAPHRASE" "${MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode approx_interleaved
      --notes "Fair paired memarch paraphrase workload"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_family_pair() {
  local group_label="qwen05b__family_reuse__req${REQS_FAMILY}"
  make_pair_names "$group_label"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_FAMILY" "$REQS_FAMILY"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_FAMILY_CLUSTERED"
      --mode mixed_reuse
      --notes "Fair paired baseline family-clustered workload"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_FAMILY" "$REQS_FAMILY" "${MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_FAMILY_CLUSTERED"
      --mode family_clustered
      --notes "Fair paired memarch family-clustered workload"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

# ------------------------------------------------------------
# Sanity print
# ------------------------------------------------------------
echo "============================================================"
echo "Fair benchmark testing configuration"
echo "REPO_ROOT:                 $REPO_ROOT"
echo "SQUAD_ROOT:                $SQUAD_ROOT"
echo "TIER2_REPO:                $TIER2_REPO"
echo "WORKLOAD_EXACT:            $WORKLOAD_EXACT"
echo "WORKLOAD_PARAPHRASE:       $WORKLOAD_PARAPHRASE"
echo "WORKLOAD_FAMILY_CLUSTERED: $WORKLOAD_FAMILY_CLUSTERED"
echo "MODEL_05B:                 $MODEL_05B"
echo "EMBEDDING_MODEL_ID:        $EMBEDDING_MODEL_ID"
echo "BASELINE_SCRIPT:           $BASELINE_SCRIPT"
echo "MEMARCH_SCRIPT:            $MEMARCH_SCRIPT"
echo "RUN_BASELINE:              $RUN_BASELINE"
echo "RUN_MEMARCH:               $RUN_MEMARCH"
echo "DEVICE / DTYPE:            $DEVICE / $DTYPE"
echo "LOCAL_FILES_ONLY:          $LOCAL_FILES_ONLY"
echo "MAX_INPUT_TOKENS:          $MAX_INPUT_TOKENS"
echo "MAX_NEW_TOKENS:            $MAX_NEW_TOKENS"
echo "REQS_COLD:                 $REQS_COLD"
echo "REQS_EXACT:                $REQS_EXACT"
echo "REQS_PARAPHRASE:           $REQS_PARAPHRASE"
echo "REQS_FAMILY:               $REQS_FAMILY"
echo "LOG_ROOT:                  $LOG_ROOT"
echo "MEMORY_ROOT:               $MEMORY_ROOT"
echo "============================================================"

# ------------------------------------------------------------
# Execute paired fair comparisons
# ------------------------------------------------------------
run_cold_pair
run_exact_pair
run_paraphrase_pair
run_family_pair

echo "============================================================"
echo "All requested fair benchmark runs completed at $(date)"
echo "Logs: $LOG_ROOT"
echo "Baseline artifacts: $OUT_ROOT_BASELINE"
echo "MemArch artifacts:  $OUT_ROOT_MEMARCH"
echo "============================================================"