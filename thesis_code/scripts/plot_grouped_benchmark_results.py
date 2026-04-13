from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


VALID_GROUP_PATTERNS = {
    "qwen05b__cold__exact__req8": "Cold",
    "qwen05b__exact_reuse__req16": "Exact Reuse",
    "qwen05b__paraphrase_reuse__req32": "Paraphrase Reuse",
    "qwen05b__family_reuse__req64": "Family Reuse",

    "qwen15b__cold__exact__req8": "Cold",
    "qwen15b__exact_reuse__req16": "Exact Reuse",
    "qwen15b__paraphrase_reuse__req32": "Paraphrase Reuse",
    "qwen15b__family_reuse__req64": "Family Reuse",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot grouped benchmark results for baseline vs memarch."
    )
    ap.add_argument(
        "--baseline_root",
        type=str,
        default="artifacts/benchmark_runs/baseline",
        help="Root directory containing baseline run JSONL files.",
    )
    ap.add_argument(
        "--memarch_root",
        type=str,
        default="artifacts/benchmark_runs/memarch",
        help="Root directory containing memarch run JSONL files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/benchmark_runs/plots",
        help="Directory where plots and summary JSON will be written.",
    )
    ap.add_argument(
        "--prefer_latest",
        action="store_true",
        help="If multiple runs exist for the same group/system, keep the latest one.",
    )
    return ap.parse_args()


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e
    return records


def extract_run_info_from_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    try:
        records = load_jsonl_records(path)
    except Exception:
        return None

    header = next((r for r in records if r.get("type") == "run_header"), None)
    footer = next((r for r in reversed(records) if r.get("type") == "run_footer"), None)
    if footer is None:
        return None

    benchmark_name = None
    if header is not None:
        benchmark_name = header.get("benchmark_name") or header.get("run_id")
    if benchmark_name is None:
        benchmark_name = footer.get("benchmark_name") or footer.get("run_id") or path.stem

    latencies = [
        float(r["latency_s"])
        for r in records
        if r.get("type") == "example_result"
        and r.get("ok") is True
        and isinstance(r.get("latency_s"), (int, float))
    ]

    mean_latency = None
    agg = footer.get("aggregate_metrics", {})
    if isinstance(agg, dict) and isinstance(agg.get("mean_latency_s"), (int, float)):
        mean_latency = float(agg["mean_latency_s"])
    elif latencies:
        mean_latency = sum(latencies) / len(latencies)

    return {
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "benchmark_name": str(benchmark_name),
        "latencies": latencies,
        "mean_latency_s": mean_latency,
        "footer": footer,
    }


def normalize_system_and_group(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Only accept the strict paired naming scheme:
      baseline__qwen05b__cold__exact__req8
      memarch__qwen05b__cold__exact__req8
    """
    if name.startswith("baseline__"):
        system = "baseline"
        group = name[len("baseline__"):]
    elif name.startswith("memarch__"):
        system = "memarch"
        group = name[len("memarch__"):]
    else:
        return None, None

    if group not in VALID_GROUP_PATTERNS:
        return None, None

    return system, group


def discover_runs(root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not root.exists():
        return runs

    for path in root.rglob("*.jsonl"):
        info = extract_run_info_from_jsonl(path)
        if info is None:
            continue

        system, group_label = normalize_system_and_group(info["benchmark_name"])
        if system is None or group_label is None:
            continue

        info["system"] = system
        info["group_label"] = group_label
        runs.append(info)

    return runs


def select_runs(runs: List[Dict[str, Any]], prefer_latest: bool) -> Dict[Tuple[str, str], Dict[str, Any]]:
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for run in runs:
        key = (run["group_label"], run["system"])
        if key not in selected:
            selected[key] = run
            continue

        if prefer_latest and run["mtime"] > selected[key]["mtime"]:
            selected[key] = run

    return selected


def group_runs(
    baseline_runs: List[Dict[str, Any]],
    memarch_runs: List[Dict[str, Any]],
    prefer_latest: bool,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    selected = {}
    selected.update(select_runs(baseline_runs, prefer_latest))
    selected.update(select_runs(memarch_runs, prefer_latest))

    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (group_label, system), run in selected.items():
        grouped.setdefault(group_label, {})
        grouped[group_label][system] = run

    return grouped


def ordered_groups(grouped: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    order = list(VALID_GROUP_PATTERNS.keys())
    return [g for g in order if g in grouped and "baseline" in grouped[g] and "memarch" in grouped[g]]


def write_bar_chart(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Path:
    groups = ordered_groups(grouped)

    labels: List[str] = []
    baseline_means: List[float] = []
    memarch_means: List[float] = []

    for group_label in groups:
        pair = grouped[group_label]
        b = pair["baseline"].get("mean_latency_s")
        m = pair["memarch"].get("mean_latency_s")
        if b is None or m is None:
            continue

        labels.append(VALID_GROUP_PATTERNS[group_label])
        baseline_means.append(float(b))
        memarch_means.append(float(m))

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], baseline_means, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], memarch_means, width=width, label="MemArch")
    plt.xticks(x, labels)
    plt.ylabel("Average per-query latency (s)")
    plt.xlabel("Workload")
    plt.title("Average Per-Query Latency by Workload")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "avg_latency_by_workload.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def write_normalized_bar_chart(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Path:
    groups = ordered_groups(grouped)

    labels: List[str] = []
    normalized_vals: List[float] = []

    for group_label in groups:
        pair = grouped[group_label]
        b = pair["baseline"].get("mean_latency_s")
        m = pair["memarch"].get("mean_latency_s")
        if b is None or m is None or float(b) == 0.0:
            continue

        labels.append(VALID_GROUP_PATTERNS[group_label])
        normalized_vals.append(float(m) / float(b))

    x = list(range(len(labels)))

    plt.figure(figsize=(10, 6))
    plt.bar(x, normalized_vals)
    plt.axhline(y=1.0, linestyle="--")
    plt.xticks(x, labels)
    plt.ylabel("Normalized latency (MemArch / Baseline)")
    plt.xlabel("Workload")
    plt.title("Normalized Average Latency by Workload (Lower is Better)")

    for i, v in enumerate(normalized_vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()

    out_path = out_dir / "normalized_latency.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def write_per_query_plots(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> List[Path]:
    groups = ordered_groups(grouped)
    written: List[Path] = []

    for group_label in groups:
        pair = grouped[group_label]
        base_lat = pair["baseline"].get("latencies", [])
        mem_lat = pair["memarch"].get("latencies", [])

        if not base_lat and not mem_lat:
            continue

        plt.figure(figsize=(10, 5))

        if base_lat:
            plt.plot(range(1, len(base_lat) + 1), base_lat, marker="o", label="Baseline")
        if mem_lat:
            plt.plot(range(1, len(mem_lat) + 1), mem_lat, marker="o", label="MemArch")

        plt.xlabel("Query index")
        plt.ylabel("Latency (s)")
        plt.title(f"Per-Query Latency: {VALID_GROUP_PATTERNS[group_label]}")
        plt.legend()
        plt.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_label)
        out_path = out_dir / f"latency_curve__{safe_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(out_path)

    return written


def write_summary_json(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Path:
    groups = ordered_groups(grouped)
    summary: Dict[str, Any] = {}

    for group_label in groups:
        pair = grouped[group_label]
        friendly = VALID_GROUP_PATTERNS[group_label]
        summary[friendly] = {}

        for system in ("baseline", "memarch"):
            run = pair[system]
            summary[friendly][system] = {
                "benchmark_name": run.get("benchmark_name"),
                "path": run.get("path"),
                "mean_latency_s": run.get("mean_latency_s"),
                "num_queries": len(run.get("latencies", [])),
            }

    out_path = out_dir / "grouped_latency_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()

    baseline_root = Path(args.baseline_root).expanduser().resolve()
    memarch_root = Path(args.memarch_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_runs = discover_runs(baseline_root)
    memarch_runs = discover_runs(memarch_root)

    grouped = group_runs(
        baseline_runs=baseline_runs,
        memarch_runs=memarch_runs,
        prefer_latest=bool(args.prefer_latest),
    )

    valid_groups = ordered_groups(grouped)
    if not valid_groups:
        raise SystemExit(
            "No valid paired runs found. Expected benchmark names like "
            "'baseline__qwen05b__exact_reuse__req16' and "
            "'memarch__qwen05b__exact_reuse__req16'."
        )

    bar_path = write_bar_chart(grouped, out_dir)
    norm_path = write_normalized_bar_chart(grouped, out_dir)
    curve_paths = write_per_query_plots(grouped, out_dir)
    summary_path = write_summary_json(grouped, out_dir)

    print("Used workload groups:")
    for g in valid_groups:
        print(f"- {g} -> {VALID_GROUP_PATTERNS[g]}")

    print("\nWrote:")
    print(f"- {bar_path}")
    print(f"- {norm_path}")
    for p in curve_paths:
        print(f"- {p}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()