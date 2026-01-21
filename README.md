# Baseline Truncation Runner for LongBench

This directory contains a baseline tiered memory system implementation for running LongBench evaluations with token truncation strategies. The system implements a 3-tier architecture without advanced memory optimizations to establish performance baselines.

## Architecture Overview

- **Tier 2 (Disk)**: LongBench JSONL dataset files + output logs
- **Tier 1 (RAM)**: In-process dataset cache (LRU-style)  
- **Tier 0 (Compute)**: MPS if available, else CPU fallback

## Prerequisites

1. Python 3.8+ with virtual environment
2. Required packages (install via pip):
   ```bash
   pip install torch transformers datasets psutil pandas matplotlib
   ```
3. Access to HuggingFace models (default: microsoft/Phi-3-mini-128k-instruct)

## Setup Instructions

### 1. Prepare the LongBench Dataset

First, you need to obtain and extract the LongBench dataset:

```bash
# Extract the LongBench data.zip file (if you have it)
python prepare_longbench_data.py --repo_dir tier2_disk/longbench_repo

# Or specify custom paths
python prepare_longbench_data.py --repo_dir /path/to/longbench --out_dir /path/to/extraction
```

This will extract JSONL files from the LongBench dataset zip archive.

### 2. Run the Baseline Evaluation

Run the main evaluation script with the extracted dataset:

```bash
# Basic run with default settings
python run_longbench_baseline.py --tier2_repo tier2_disk/longbench_repo/data

# Run with custom parameters
python run_longbench_baseline.py \
    --tier2_repo tier2_disk/longbench_repo/data \
    --out_dir results \
    --model_id microsoft/Phi-3-mini-128k-instruct \
    --max_examples 50 \
    --max_input_tokens 4096 \
    --cpu_fallback_on_long
```

### 3. Analyze Results

After running the evaluation, analyze the results:

```bash
# Generate summary statistics
python summarize_run.py --run_jsonl tier2_disk/runs/baseline_longbench_[timestamp].jsonl

# Create visualization plots
python plot_run.py --run_jsonl tier2_disk/runs/baseline_longbench_[timestamp].jsonl --out_png results_plot.png
```

## Command Line Arguments

### run_longbench_baseline.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tier2_repo` | str | **Required** | Path to LongBench dataset repo (Tier 2) |
| `--out_dir` | str | `tier2_disk/runs` | Output folder for results |
| `--model_id` | str | `microsoft/Phi-3-mini-128k-instruct` | HuggingFace model ID |
| `--task_glob` | str | `""` | Substring filter for task files (e.g., "trec") |
| `--max_examples` | int | `25` | Maximum number of examples to process |
| `--max_new_tokens` | int | `64` | Maximum tokens to generate |
| `--max_cache_items` | int | `64` | RAM cache size (Tier 1) |
| `--max_input_tokens` | int | `8192` | Maximum input tokens before truncation |
| `--cpu_fallback_on_long` | flag | `False` | Use CPU when MPS encounters long sequences |

### summarize_run.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run_jsonl` | str | **Required** | Path to baseline_longbench_*.jsonl file |
| `--out_csv` | str | `""` | Optional output CSV path |

### plot_run.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run_jsonl` | str | **Required** | Path to baseline_longbench_*.jsonl file |
| `--out_png` | str | `latency_vs_input_tokens.png` | Output PNG path for plot |

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows

# 2. Install dependencies
pip install torch transformers datasets psutil pandas matplotlib

# 3. Extract LongBench dataset
python prepare_longbench_data.py --repo_dir tier2_disk/longbench_repo

# 4. Run evaluation with CPU fallback enabled
python run_longbench_baseline.py \
    --tier2_repo tier2_disk/longbench_repo/data \
    --max_examples 100 \
    --max_input_tokens 4096 \
    --cpu_fallback_on_long

# 5. Find the generated results file
ls tier2_disk/runs/baseline_longbench_*.jsonl

# 6. Analyze results (replace [timestamp] with actual timestamp)
python summarize_run.py --run_jsonl tier2_disk/runs/baseline_longbench_[timestamp].jsonl

# 7. Generate plots
python plot_run.py --run_jsonl tier2_disk/runs/baseline_longbench_[timestamp].jsonl
```

## Output Files

The system generates several output files:

- **Results JSONL**: `tier2_disk/runs/baseline_longbench_[timestamp].jsonl`
  - Contains evaluation results for each processed example
  - Includes metrics like latency, tokens/sec, memory usage, truncation flags
  
- **Summary Statistics**: Console output from `summarize_run.py`
  - Descriptive statistics on latency, throughput, token counts
  - Device usage breakdown, truncation rates

- **Visualization**: `latency_vs_input_tokens.png` (or custom name)
  - Scatter plot showing latency vs input token count
  - Separate series for CPU vs MPS device usage

## Key Features

- **Token Truncation**: Automatically truncates inputs that exceed model limits
- **Device Flexibility**: Supports MPS (Apple Silicon) and CPU execution  
- **Memory Monitoring**: Tracks RSS memory usage before/after each example
- **Error Handling**: Gracefully handles runtime errors and logs them
- **Caching**: LRU-style RAM cache for dataset files to reduce disk I/O
- **Comprehensive Logging**: Detailed metrics for performance analysis

## Troubleshooting

### Common Issues

1. **Model loading fails**: Ensure you have sufficient disk space and internet connectivity for HuggingFace downloads

2. **MPS errors**: Use `--cpu_fallback_on_long` flag to automatically switch to CPU for problematic inputs

3. **Memory issues**: Reduce `--max_examples`, `--max_input_tokens`, or `--max_cache_items`

4. **No JSONL files found**: Verify the dataset path and ensure `prepare_longbench_data.py` completed successfully

### Performance Tips

- Use `--task_glob` to filter to specific tasks during development
- Start with small `--max_examples` values to test setup
- Monitor memory usage and adjust cache sizes accordingly
- Enable CPU fallback for stability with diverse input lengths

## Purpose

This baseline implementation establishes:
- Tier plumbing infrastructure  
- Reproducible performance metrics
- Failure mode documentation
- Foundation for advanced memory optimization comparisons

It intentionally does NOT implement:
- Retrieval indexing
- Hierarchical promotion/eviction
- Content summarization/compression
- Advanced memory management strategies

These limitations are by design to create a clean baseline for measuring the impact of memory optimizations in subsequent implementations.
