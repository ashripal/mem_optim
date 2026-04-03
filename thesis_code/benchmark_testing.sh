#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Benchmark testing sweep: Baseline + MemArch
# ============================================================
#
# Usage examples:
#
#   chmod +x benchmark_testing.sh
#   ./benchmark_testing.sh
#
#   REPO_ROOT=/path/to/thesis_code \
#   MODEL_05B=/path/to/Qwen2.5-0.5B-Instruct \
#   MODEL_15B=/path/to/Qwen2.5-1.5B-Instruct \
#   EMBEDDING_MODEL_ID=/path/to/all-MiniLM-L6-v2 \
#   ./benchmark_testing.sh
#
# Optional knobs:
#   RUN_BASELINE=1
#   RUN_MEMARCH=1
#   RUN_05B=1
#   RUN_15B=1
#   MAX_EXACT=8
#   MAX_APPROX=16
#   DEVICE=cpu
#   DTYPE=float32
#   LOCAL_FILES_ONLY=1
#
# This script:
# - uses environment variables for all important paths
# - runs baseline and memarch comparisons
# - includes exact / paraphrase / family-clustered tests
# - keeps each run on its own store path
# - writes logs under a dedicated log root
# ============================================================

# ------------------------------------------------------------
# Path / environment configuration
# ------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/Users/ashripal/mem_optim/thesis_code}"
cd "$REPO_ROOT"

# Main data roots
SQUAD_ROOT="${SQUAD_ROOT:-$REPO_ROOT/baseline_old/tier2_disk/data/squad_clean}"
TIER2_REPO="${TIER2_REPO:-$SQUAD_ROOT}"

# Input workloads
WORKLOAD_EXACT="${WORKLOAD_EXACT:-$SQUAD_ROOT/workload_exact.jsonl}"
WORKLOAD_PARAPHRASE="${WORKLOAD_PARAPHRASE:-$SQUAD_ROOT/workload_paraphrase.jsonl}"
WORKLOAD_PARAPHRASED="${WORKLOAD_PARAPHRASED:-$SQUAD_ROOT/workload_paraphrased.jsonl}"
WORKLOAD_FAMILY_CLUSTERED="${WORKLOAD_FAMILY_CLUSTERED:-$SQUAD_ROOT/workload_family_clustered.jsonl}"

# Models
MODEL_05B="${MODEL_05B:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}"
MODEL_15B="${MODEL_15B:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
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
RUN_05B="${RUN_05B:-1}"
RUN_15B="${RUN_15B:-0}"

# ------------------------------------------------------------
# Compute / identity settings
# ------------------------------------------------------------
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"

USER_ID="${USER_ID:-user_a}"
SESSION_ID="${SESSION_ID:-session_a}"
COHORT_ID="${COHORT_ID:-}"

# ------------------------------------------------------------
# Benchmark sizes / generation settings
# ------------------------------------------------------------
MAX_COLD="${MAX_COLD:-8}"
MAX_EXACT="${MAX_EXACT:-8}"
MAX_APPROX="${MAX_APPROX:-16}"
MAX_FAMILY="${MAX_FAMILY:-16}"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# ------------------------------------------------------------
# Memory / retrieval settings
# ------------------------------------------------------------
RAM_CAPACITY_ITEMS="${RAM_CAPACITY_ITEMS:-256}"

LEXICAL_CONTEXT_THRESHOLD="${LEXICAL_CONTEXT_THRESHOLD:-0.55}"
LEXICAL_DIRECT_THRESHOLD="${LEXICAL_DIRECT_THRESHOLD:-0.75}"

SEMANTIC_THRESHOLD_CONTEXT_STRICT="${SEMANTIC_THRESHOLD_CONTEXT_STRICT:-0.85}"
SEMANTIC_THRESHOLD_BYPASS_STRICT="${SEMANTIC_THRESHOLD_BYPASS_STRICT:-0.95}"

SEMANTIC_THRESHOLD_CONTEXT_LOOSE="${SEMANTIC_THRESHOLD_CONTEXT_LOOSE:-0.55}"
SEMANTIC_THRESHOLD_BYPASS_LOOSE="${SEMANTIC_THRESHOLD_BYPASS_LOOSE:-0.90}"

MAX_SEMANTIC_CANDIDATES="${MAX_SEMANTIC_CANDIDATES:-5}"
SEMANTIC_BYPASS_MIN_MARGIN="${SEMANTIC_BYPASS_MIN_MARGIN:-0.02}"
SEMANTIC_BYPASS_MAX_ANSWER_WORDS="${SEMANTIC_BYPASS_MAX_ANSWER_WORDS:-12}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
bool_flag() {
  # Usage: bool_flag 1 --local_files_only
  local enabled="$1"
  local flag="$2"
  if [[ "$enabled" == "1" ]]; then
    printf '%s\n' "$flag"
  fi
}

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
  local model_id="$1"
  local max_examples="$2"
  cat <<EOF
--tier2_repo
$TIER2_REPO
--out_root
$OUT_ROOT_BASELINE
--max_examples
$max_examples
--model_id
$model_id
--device
$DEVICE
--dtype
$DTYPE
--max_input_tokens
$MAX_INPUT_TOKENS
--max_new_tokens
$MAX_NEW_TOKENS
--ram_capacity_items
$RAM_CAPACITY_ITEMS
--user_id
$USER_ID
--session_id
$SESSION_ID
--write_summary_json
EOF
}

memarch_common_args() {
  local model_id="$1"
  local max_examples="$2"
  cat <<EOF
--tier2_repo
$TIER2_REPO
--out_root
$OUT_ROOT_MEMARCH
--max_examples
$max_examples
--model_id
$model_id
--device
$DEVICE
--dtype
$DTYPE
--ram_capacity_items
$RAM_CAPACITY_ITEMS
--user_id
$USER_ID
--session_id
$SESSION_ID
--max_input_tokens
$MAX_INPUT_TOKENS
--max_new_tokens
$MAX_NEW_TOKENS
--embedding_model_id
$EMBEDDING_MODEL_ID
--embedding_device
$EMBEDDING_DEVICE
--max_semantic_candidates
$MAX_SEMANTIC_CANDIDATES
--write_summary_json
EOF
}

append_local_files_only() {
  local -n _arr=$1
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    _arr+=(--local_files_only)
    _arr+=(--embedding_local_files_only)
  fi
}

run_model_suite() {
  local model_label="$1"
  local model_id="$2"

  # -------------------------
  # Baseline tests
  # -------------------------
  if [[ "$RUN_BASELINE" == "1" ]]; then
    # Cold smoke
    local baseline_cold_name="baseline_${model_label}_cpu_smoke_squad_8_${TIMESTAMP}"
    local -a baseline_cold_args=()
    mapfile -t baseline_cold_args < <(baseline_common_args "$model_id" "$MAX_COLD")
    append_local_files_only baseline_cold_args
    baseline_cold_args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode cold
      --notes "Baseline CPU smoke test on laptop (${model_label})"
    )
    run_case "$BASELINE_SCRIPT" "$baseline_cold_name" "${baseline_cold_args[@]}"

    # Exact interleaved
    local baseline_exact_name="baseline_${model_label}_exact_interleaved_8_${TIMESTAMP}"
    local -a baseline_exact_args=()
    mapfile -t baseline_exact_args < <(baseline_common_args "$model_id" "$MAX_EXACT")
    append_local_files_only baseline_exact_args
    baseline_exact_args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode exact_interleaved
      --notes "Baseline exact interleaved test (${model_label})"
    )
    run_case "$BASELINE_SCRIPT" "$baseline_exact_name" "${baseline_exact_args[@]}"

    # Approx interleaved paraphrase
    local baseline_para_name="baseline_${model_label}_paraphrase_approx_16_${TIMESTAMP}"
    local -a baseline_para_args=()
    mapfile -t baseline_para_args < <(baseline_common_args "$model_id" "$MAX_APPROX")
    append_local_files_only baseline_para_args
    baseline_para_args+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode approx_interleaved
      --notes "Baseline paraphrase approx test (${model_label})"
    )
    run_case "$BASELINE_SCRIPT" "$baseline_para_name" "${baseline_para_args[@]}"

    # Family-clustered
    local baseline_family_name="baseline_${model_label}_family_clustered_16_${TIMESTAMP}"
    local -a baseline_family_args=()
    mapfile -t baseline_family_args < <(baseline_common_args "$model_id" "$MAX_FAMILY")
    append_local_files_only baseline_family_args
    baseline_family_args+=(
      --input_path "$WORKLOAD_FAMILY_CLUSTERED"
      --mode family_clustered
      --notes "Baseline family clustered test (${model_label})"
    )
    run_case "$BASELINE_SCRIPT" "$baseline_family_name" "${baseline_family_args[@]}"
  fi

  # -------------------------
  # MemArch tests
  # -------------------------
  if [[ "$RUN_MEMARCH" == "1" ]]; then
    # Cold smoke with lexical + semantic
    local memarch_cold_name="memarch_${model_label}_cpu_smoke_squad_8_${TIMESTAMP}"
    local -a memarch_cold_args=()
    mapfile -t memarch_cold_args < <(memarch_common_args "$model_id" "$MAX_COLD")
    append_local_files_only memarch_cold_args
    memarch_cold_args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode cold
      --retrieval_mode lexical_gated_direct_semantic_context
      --lexical_enabled
      --semantic_enabled
      --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT_STRICT"
      --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS_STRICT"
      --clear_disk_store_before_run
      --disk_store_path "${MEMORY_ROOT}/${memarch_cold_name}.sqlite"
      --notes "MemArch CPU smoke test on laptop (${model_label})"
    )
    run_case "$MEMARCH_SCRIPT" "$memarch_cold_name" "${memarch_cold_args[@]}"

    # Exact interleaved
    local memarch_exact_name="memarch_${model_label}_exact_interleaved_8_${TIMESTAMP}"
    local -a memarch_exact_args=()
    mapfile -t memarch_exact_args < <(memarch_common_args "$model_id" "$MAX_EXACT")
    append_local_files_only memarch_exact_args
    memarch_exact_args+=(
      --input_path "$WORKLOAD_EXACT"
      --mode exact_interleaved
      --retrieval_mode lexical_gated_direct_semantic_context
      --lexical_enabled
      --semantic_enabled
      --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT_STRICT"
      --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS_STRICT"
      --clear_disk_store_before_run
      --disk_store_path "${MEMORY_ROOT}/${memarch_exact_name}.sqlite"
      --notes "MemArch CPU exact interleaved reuse test (${model_label})"
    )
    run_case "$MEMARCH_SCRIPT" "$memarch_exact_name" "${memarch_exact_args[@]}"

    # Approx paraphrase with lexical + semantic
    local memarch_para_mix_name="memarch_${model_label}_paraphrase_mix_16_${TIMESTAMP}"
    local -a memarch_para_mix_args=()
    mapfile -t memarch_para_mix_args < <(memarch_common_args "$model_id" "$MAX_APPROX")
    append_local_files_only memarch_para_mix_args
    memarch_para_mix_args+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode approx_interleaved
      --retrieval_mode lexical_gated_direct_semantic_context
      --lexical_enabled
      --semantic_enabled
      --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT_STRICT"
      --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS_STRICT"
      --clear_disk_store_before_run
      --disk_store_path "${MEMORY_ROOT}/${memarch_para_mix_name}.sqlite"
      --notes "MemArch paraphrase approx test lexical+semantic (${model_label})"
    )
    run_case "$MEMARCH_SCRIPT" "$memarch_para_mix_name" "${memarch_para_mix_args[@]}"

    # Semantic bypass approx interleaved
    local memarch_para_sem_name="memarch_${model_label}_para_semantic_bypass_090_${TIMESTAMP}"
    local -a memarch_para_sem_args=()
    mapfile -t memarch_para_sem_args < <(memarch_common_args "$model_id" "$MAX_APPROX")
    append_local_files_only memarch_para_sem_args
    memarch_para_sem_args+=(
      --input_path "$WORKLOAD_PARAPHRASE"
      --mode approx_interleaved
      --retrieval_mode semantic_bypass
      --semantic_enabled
      --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT_LOOSE"
      --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS_LOOSE"
      --allow_semantic_bypass
      --require_same_document_for_semantic_bypass
      --require_evidence_support_for_semantic_bypass
      --semantic_bypass_min_margin "$SEMANTIC_BYPASS_MIN_MARGIN"
      --semantic_bypass_max_answer_words "$SEMANTIC_BYPASS_MAX_ANSWER_WORDS"
      --clear_disk_store_before_run
      --disk_store_path "${MEMORY_ROOT}/${memarch_para_sem_name}.sqlite"
      --notes "MemArch semantic bypass approx test (${model_label})"
    )
    run_case "$MEMARCH_SCRIPT" "$memarch_para_sem_name" "${memarch_para_sem_args[@]}"

    # Family-clustered semantic bypass
    local memarch_family_name="memarch_${model_label}_family_clustered_semantic_bypass_090_${TIMESTAMP}"
    local -a memarch_family_args=()
    mapfile -t memarch_family_args < <(memarch_common_args "$model_id" "$MAX_FAMILY")
    append_local_files_only memarch_family_args
    memarch_family_args+=(
      --input_path "$WORKLOAD_FAMILY_CLUSTERED"
      --mode family_clustered
      --retrieval_mode semantic_bypass
      --semantic_enabled
      --semantic_threshold_context "$SEMANTIC_THRESHOLD_CONTEXT_LOOSE"
      --semantic_threshold_bypass "$SEMANTIC_THRESHOLD_BYPASS_LOOSE"
      --allow_semantic_bypass
      --require_same_document_for_semantic_bypass
      --require_evidence_support_for_semantic_bypass
      --semantic_bypass_min_margin "$SEMANTIC_BYPASS_MIN_MARGIN"
      --semantic_bypass_max_answer_words "$SEMANTIC_BYPASS_MAX_ANSWER_WORDS"
      --clear_disk_store_before_run
      --disk_store_path "${MEMORY_ROOT}/${memarch_family_name}.sqlite"
      --notes "MemArch family clustered paraphrase test before canonicalization (${model_label})"
    )
    run_case "$MEMARCH_SCRIPT" "$memarch_family_name" "${memarch_family_args[@]}"
  fi
}

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
echo "============================================================"
echo "Benchmark testing configuration"
echo "REPO_ROOT:               $REPO_ROOT"
echo "SQUAD_ROOT:              $SQUAD_ROOT"
echo "TIER2_REPO:              $TIER2_REPO"
echo "WORKLOAD_EXACT:          $WORKLOAD_EXACT"
echo "WORKLOAD_PARAPHRASE:     $WORKLOAD_PARAPHRASE"
echo "WORKLOAD_PARAPHRASED:    $WORKLOAD_PARAPHRASED"
echo "WORKLOAD_FAMILY_CLUSTERED:$WORKLOAD_FAMILY_CLUSTERED"
echo "MODEL_05B:               $MODEL_05B"
echo "MODEL_15B:               $MODEL_15B"
echo "EMBEDDING_MODEL_ID:      $EMBEDDING_MODEL_ID"
echo "BASELINE_SCRIPT:         $BASELINE_SCRIPT"
echo "MEMARCH_SCRIPT:          $MEMARCH_SCRIPT"
echo "RUN_BASELINE:            $RUN_BASELINE"
echo "RUN_MEMARCH:             $RUN_MEMARCH"
echo "RUN_05B:                 $RUN_05B"
echo "RUN_15B:                 $RUN_15B"
echo "DEVICE / DTYPE:          $DEVICE / $DTYPE"
echo "LOCAL_FILES_ONLY:        $LOCAL_FILES_ONLY"
echo "LOG_ROOT:                $LOG_ROOT"
echo "============================================================"

# ------------------------------------------------------------
# Execute suites
# ------------------------------------------------------------
if [[ "$RUN_05B" == "1" ]]; then
  run_model_suite "qwen05b" "$MODEL_05B"
fi

if [[ "$RUN_15B" == "1" ]]; then
  run_model_suite "qwen15b" "$MODEL_15B"
fi

echo "============================================================"
echo "All requested benchmark runs completed at $(date)"
echo "Logs: $LOG_ROOT"
echo "Baseline artifacts: $OUT_ROOT_BASELINE"
echo "MemArch artifacts:  $OUT_ROOT_MEMARCH"
echo "============================================================"