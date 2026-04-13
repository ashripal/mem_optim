#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Fair benchmark sweep: Baseline vs MemArch
# Runs:
#   1) Qwen 0.5B
#   2) Qwen 1.5B
#
# Also generates grouped comparison plots per model:
#   artifacts/benchmark_runs/plots/qwen05
#   artifacts/benchmark_runs/plots/qwen15
# ============================================================

make_pair_names() {
  local model_group="$1"
  local workload_group="$2"
  BASE_NAME="baseline__${model_group}__${workload_group}"
  MEM_NAME="memarch__${model_group}__${workload_group}"
}

REPO_ROOT="${REPO_ROOT:-/home/sravani/mem_optim/thesis_code}"
cd "$REPO_ROOT"

# ------------------------------------------------------------
# Main data roots
# ------------------------------------------------------------
SQUAD_ROOT="${SQUAD_ROOT:-$REPO_ROOT/baseline_old/tier2_disk/data/squad_clean}"
TIER2_REPO="${TIER2_REPO:-$SQUAD_ROOT}"

# Input workloads
WORKLOAD_EXACT="${WORKLOAD_EXACT:-$SQUAD_ROOT/workload_exact.jsonl}"
WORKLOAD_PARAPHRASE="${WORKLOAD_PARAPHRASE:-$SQUAD_ROOT/workload_paraphrase.jsonl}"
WORKLOAD_FAMILY_CLUSTERED="${WORKLOAD_FAMILY_CLUSTERED:-$SQUAD_ROOT/workload_family_clustered.jsonl}"

# Models
MODEL_05B="${MODEL_05B:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}"
MODEL_15B="${MODEL_15B:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-$REPO_ROOT/models/all-MiniLM-L6-v2}"

# Scripts
BASELINE_SCRIPT="${BASELINE_SCRIPT:-scripts/run_baseline_benchmark.py}"
MEMARCH_SCRIPT="${MEMARCH_SCRIPT:-scripts/run_memarch_benchmark.py}"
GROUPED_PLOT_SCRIPT="${GROUPED_PLOT_SCRIPT:-scripts/plot_grouped_benchmark_results.py}"

# Output roots
OUT_ROOT_BASELINE="${OUT_ROOT_BASELINE:-artifacts/benchmark_runs/baseline}"
OUT_ROOT_MEMARCH="${OUT_ROOT_MEMARCH:-artifacts/benchmark_runs/memarch}"
LOG_ROOT="${LOG_ROOT:-artifacts/benchmark_runs/testing_logs}"
MEMORY_ROOT="${MEMORY_ROOT:-artifacts/benchmark_runs/memory}"
PLOT_ROOT="${PLOT_ROOT:-artifacts/benchmark_runs/plots}"

mkdir -p "$LOG_ROOT" "$MEMORY_ROOT" "$OUT_ROOT_BASELINE" "$OUT_ROOT_MEMARCH" "$PLOT_ROOT"

# ------------------------------------------------------------
# Run toggles
# ------------------------------------------------------------
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_MEMARCH="${RUN_MEMARCH:-1}"
RUN_MODEL_05B="${RUN_MODEL_05B:-1}"
RUN_MODEL_15B="${RUN_MODEL_15B:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
PREFER_LATEST_PLOTS="${PREFER_LATEST_PLOTS:-1}"

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
# MemArch settings
# ------------------------------------------------------------
RAM_CAPACITY_ITEMS="${RAM_CAPACITY_ITEMS:-256}"
MAX_SEMANTIC_CANDIDATES="${MAX_SEMANTIC_CANDIDATES:-5}"

SEMANTIC_THRESHOLD_CONTEXT="${SEMANTIC_THRESHOLD_CONTEXT:-0.55}"
SEMANTIC_THRESHOLD_BYPASS="${SEMANTIC_THRESHOLD_BYPASS:-0.90}"
SEMANTIC_BYPASS_MIN_MARGIN="${SEMANTIC_BYPASS_MIN_MARGIN:-0.02}"
SEMANTIC_BYPASS_MAX_ANSWER_WORDS="${SEMANTIC_BYPASS_MAX_ANSWER_WORDS:-12}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ------------------------------------------------------------
# Per-model state
# ------------------------------------------------------------
CURRENT_MODEL_ALIAS=""
CURRENT_MODEL_GROUP=""
CURRENT_MODEL_PATH=""
CURRENT_OUT_ROOT_BASELINE=""
CURRENT_OUT_ROOT_MEMARCH=""
CURRENT_LOG_ROOT=""
CURRENT_MEMORY_ROOT=""
CURRENT_PLOT_ROOT=""

set_current_model() {
  local model_alias="$1"   # qwen05 or qwen15
  local model_group="$2"   # qwen05b or qwen15b
  local model_path="$3"

  CURRENT_MODEL_ALIAS="$model_alias"
  CURRENT_MODEL_GROUP="$model_group"
  CURRENT_MODEL_PATH="$model_path"

  CURRENT_OUT_ROOT_BASELINE="${OUT_ROOT_BASELINE}/${model_alias}"
  CURRENT_OUT_ROOT_MEMARCH="${OUT_ROOT_MEMARCH}/${model_alias}"
  CURRENT_LOG_ROOT="${LOG_ROOT}/${model_alias}"
  CURRENT_MEMORY_ROOT="${MEMORY_ROOT}/${model_alias}"
  CURRENT_PLOT_ROOT="${PLOT_ROOT}/${model_alias}"

  mkdir -p \
    "$CURRENT_OUT_ROOT_BASELINE" \
    "$CURRENT_OUT_ROOT_MEMARCH" \
    "$CURRENT_LOG_ROOT" \
    "$CURRENT_MEMORY_ROOT" \
    "$CURRENT_PLOT_ROOT"
}

run_case() {
  local script_path="$1"
  local benchmark_name="$2"
  shift 2

  local log_path="${CURRENT_LOG_ROOT}/${benchmark_name}_${TIMESTAMP}.log"

  echo "============================================================"
  echo "Starting: ${benchmark_name}"
  echo "Model alias: ${CURRENT_MODEL_ALIAS}"
  echo "Model group: ${CURRENT_MODEL_GROUP}"
  echo "Model path:  ${CURRENT_MODEL_PATH}"
  echo "Script:      ${script_path}"
  echo "Log:         ${log_path}"
  echo "Started at:  $(date)"
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
    --out_root "$CURRENT_OUT_ROOT_BASELINE"
    --model_id "$CURRENT_MODEL_PATH"
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
    --out_root "$CURRENT_OUT_ROOT_MEMARCH"
    --model_id "$CURRENT_MODEL_PATH"
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
# Plotting
# ------------------------------------------------------------
plot_current_model() {
  if [[ "$RUN_PLOTS" != "1" ]]; then
    return
  fi

  echo "============================================================"
  echo "Generating grouped comparison plots for ${CURRENT_MODEL_ALIAS}"
  echo "Baseline root: ${CURRENT_OUT_ROOT_BASELINE}"
  echo "MemArch root:  ${CURRENT_OUT_ROOT_MEMARCH}"
  echo "Out dir:       ${CURRENT_PLOT_ROOT}"
  echo "============================================================"

  local -a plot_args=(
    --baseline_root "$CURRENT_OUT_ROOT_BASELINE"
    --memarch_root "$CURRENT_OUT_ROOT_MEMARCH"
    --out_dir "$CURRENT_PLOT_ROOT"
  )

  if [[ "$PREFER_LATEST_PLOTS" == "1" ]]; then
    plot_args+=(--prefer_latest)
  fi

  python "$GROUPED_PLOT_SCRIPT" "${plot_args[@]}"
}

# ------------------------------------------------------------
# Paired fair comparisons
# ------------------------------------------------------------
run_cold_pair() {
  local workload_group="cold__exact__req${REQS_COLD}"
  make_pair_names "$CURRENT_MODEL_GROUP" "$workload_group"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_COLD" "$REQS_COLD"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode cold
      --notes "Fair paired baseline cold run on exact workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_COLD" "$REQS_COLD" "${CURRENT_MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_EXACT"
      --mode cold
      --notes "Fair paired memarch cold run on exact workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_exact_pair() {
  local workload_group="exact_reuse__req${REQS_EXACT}"
  make_pair_names "$CURRENT_MODEL_GROUP" "$workload_group"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_EXACT" "$REQS_EXACT"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode mixed_reuse
      --notes "Fair paired baseline repeated exact-query workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_EXACT" "$REQS_EXACT" "${CURRENT_MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_EXACT"
      --mode exact_interleaved
      --notes "Fair paired memarch exact-interleaved workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_paraphrase_pair() {
  local workload_group="paraphrase_reuse__req${REQS_PARAPHRASE}"
  make_pair_names "$CURRENT_MODEL_GROUP" "$workload_group"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_PARAPHRASE" "$REQS_PARAPHRASE"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode mixed_reuse
      --notes "Fair paired baseline paraphrase workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_PARAPHRASE" "$REQS_PARAPHRASE" "${CURRENT_MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode approx_interleaved
      --notes "Fair paired memarch paraphrase workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_family_pair() {
  local workload_group="family_reuse__req${REQS_FAMILY}"
  make_pair_names "$CURRENT_MODEL_GROUP" "$workload_group"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    baseline_common_args "$SEEDS_FAMILY" "$REQS_FAMILY"
    local -a args=("${BASELINE_ARGS[@]}")
    args+=(
      --input_path "$WORKLOAD_FAMILY_CLUSTERED"
      --mode mixed_reuse
      --notes "Fair paired baseline family-clustered workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$BASELINE_SCRIPT" "$BASE_NAME" "${args[@]}"
  fi

  if [[ "$RUN_MEMARCH" == "1" ]]; then
    memarch_common_args "$SEEDS_FAMILY" "$REQS_FAMILY" "${CURRENT_MEMORY_ROOT}/${MEM_NAME}.sqlite"
    local -a margs=("${MEMARCH_ARGS[@]}")
    margs+=(
      --input_path "$WORKLOAD_FAMILY_CLUSTERED"
      --mode family_clustered
      --notes "Fair paired memarch family-clustered workload (${CURRENT_MODEL_GROUP})"
    )
    run_case "$MEMARCH_SCRIPT" "$MEM_NAME" "${margs[@]}"
  fi
}

run_full_suite_for_model() {
  local model_alias="$1"
  local model_group="$2"
  local model_path="$3"

  set_current_model "$model_alias" "$model_group" "$model_path"

  echo "============================================================"
  echo "Running full suite for model alias: ${CURRENT_MODEL_ALIAS}"
  echo "Model group:                  ${CURRENT_MODEL_GROUP}"
  echo "Model path:                   ${CURRENT_MODEL_PATH}"
  echo "Baseline out root:            ${CURRENT_OUT_ROOT_BASELINE}"
  echo "MemArch out root:             ${CURRENT_OUT_ROOT_MEMARCH}"
  echo "Logs:                         ${CURRENT_LOG_ROOT}"
  echo "Memory store root:            ${CURRENT_MEMORY_ROOT}"
  echo "Plot root:                    ${CURRENT_PLOT_ROOT}"
  echo "============================================================"

  run_cold_pair
  run_exact_pair
  run_paraphrase_pair
  run_family_pair
  plot_current_model

  echo "============================================================"
  echo "Completed full suite for model alias: ${CURRENT_MODEL_ALIAS}"
  echo "============================================================"
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
echo "MODEL_15B:                 $MODEL_15B"
echo "EMBEDDING_MODEL_ID:        $EMBEDDING_MODEL_ID"
echo "BASELINE_SCRIPT:           $BASELINE_SCRIPT"
echo "MEMARCH_SCRIPT:            $MEMARCH_SCRIPT"
echo "GROUPED_PLOT_SCRIPT:       $GROUPED_PLOT_SCRIPT"
echo "RUN_BASELINE:              $RUN_BASELINE"
echo "RUN_MEMARCH:               $RUN_MEMARCH"
echo "RUN_MODEL_05B:             $RUN_MODEL_05B"
echo "RUN_MODEL_15B:             $RUN_MODEL_15B"
echo "RUN_PLOTS:                 $RUN_PLOTS"
echo "PREFER_LATEST_PLOTS:       $PREFER_LATEST_PLOTS"
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
echo "OUT_ROOT_BASELINE:         $OUT_ROOT_BASELINE"
echo "OUT_ROOT_MEMARCH:          $OUT_ROOT_MEMARCH"
echo "PLOT_ROOT:                 $PLOT_ROOT"
echo "============================================================"

# ------------------------------------------------------------
# Execute suites in order: qwen05 first, then qwen15
# ------------------------------------------------------------
if [[ "$RUN_MODEL_05B" == "1" ]]; then
  run_full_suite_for_model "qwen05" "qwen05b" "$MODEL_05B"
fi

if [[ "$RUN_MODEL_15B" == "1" ]]; then
  run_full_suite_for_model "qwen15" "qwen15b" "$MODEL_15B"
fi

echo "============================================================"
echo "All requested fair benchmark runs completed at $(date)"
echo "Logs root:          $LOG_ROOT"
echo "Baseline root:      $OUT_ROOT_BASELINE"
echo "MemArch root:       $OUT_ROOT_MEMARCH"
echo "Plots root:         $PLOT_ROOT"
echo "============================================================"