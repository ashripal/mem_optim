from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Group baseline and memarch benchmark outputs and plot latency comparisons."
    )
    ap.add_argument(
        "--baseline_root",
        type=str,
        default="artifacts/benchmark_runs/baseline",
        help="Root directory containing baseline run artifacts.",
    )
    ap.add_argument(
        "--memarch_root",
        type=str,
        default="artifacts/benchmark_runs/memarch",
        help="Root directory containing memarch run artifacts.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/benchmark_runs/plots",
        help="Directory where plots will be written.",
    )
    ap.add_argument(
        "--prefer_latest",
        action="store_true",
        help="If multiple runs exist for the same system/group, keep only the most recently modified one.",
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
        if r.get("type") == "example_result" and r.get("ok") is True and isinstance(r.get("latency_s"), (int, float))
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


def normalize_system_and_group(name: str) -> Tuple[Optional[str], str]:
    """
    Expected naming convention:
      baseline__qwen05b__exact_reuse__req16
      memarch__qwen05b__exact_reuse__req16
    """
    if name.startswith("baseline__"):
        return "baseline", name[len("baseline__"):]
    if name.startswith("memarch__"):
        return "memarch", name[len("memarch__"):]

    # Fallback for older naming patterns
    if name.startswith("baseline_"):
        stripped = re.sub(r"^baseline_", "", name)
        stripped = re.sub(r"_\d{8}_\d{6}.*$", "", stripped)
        return "baseline", stripped

    if name.startswith("memarch_"):
        stripped = re.sub(r"^memarch_", "", name)
        stripped = re.sub(r"_\d{8}_\d{6}.*$", "", stripped)
        return "memarch", stripped

    return None, name


def discover_runs(root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not root.exists():
        return runs

    for path in root.rglob("*.jsonl"):
        info = extract_run_info_from_jsonl(path)
        if info is None:
            continue
        system, group_label = normalize_system_and_group(info["benchmark_name"])
        if system is None:
            continue
        info["system"] = system
        info["group_label"] = group_label
        runs.append(info)
    return runs


def select_runs(runs: List[Dict[str, Any]], prefer_latest: bool) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Keyed by (group_label, system)
    """
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


def friendly_label(group_label: str) -> str:
    return group_label.replace("__", "\n")


def write_bar_chart(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Path:
    labels: List[str] = []
    baseline_means: List[float] = []
    memarch_means: List[float] = []

    for group_label in sorted(grouped.keys()):
        pair = grouped[group_label]
        if "baseline" not in pair or "memarch" not in pair:
            continue

        b = pair["baseline"].get("mean_latency_s")
        m = pair["memarch"].get("mean_latency_s")
        if b is None or m is None:
            continue

        labels.append(friendly_label(group_label))
        baseline_means.append(float(b))
        memarch_means.append(float(m))

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 2.4), 6))
    plt.bar([i - width / 2 for i in x], baseline_means, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], memarch_means, width=width, label="MemArch")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Average per-query latency (s)")
    plt.xlabel("Workload group")
    plt.title("Average Per-Query Latency by Workload")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "avg_latency_by_workload.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def write_per_query_plots(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> List[Path]:
    written: List[Path] = []

    for group_label in sorted(grouped.keys()):
        pair = grouped[group_label]
        if "baseline" not in pair or "memarch" not in pair:
            continue

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
        plt.title(f"Per-Query Latency: {group_label}")
        plt.legend()
        plt.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_label)
        out_path = out_dir / f"latency_curve__{safe_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(out_path)

    return written

def write_normalized_bar_chart(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Path:
    labels: List[str] = []
    normalized_vals: List[float] = []

    for group_label in sorted(grouped.keys()):
        pair = grouped[group_label]

        if "baseline" not in pair or "memarch" not in pair:
            continue

        b = pair["baseline"].get("mean_latency_s")
        m = pair["memarch"].get("mean_latency_s")

        if b is None or m is None or b == 0:
            continue

        norm = float(m) / float(b)

        labels.append(group_label.replace("__", "\n"))
        normalized_vals.append(norm)

    x = list(range(len(labels)))

    plt.figure(figsize=(max(10, len(labels) * 2.4), 6))
    plt.bar(x, normalized_vals)

    # horizontal reference line at 1.0
    plt.axhline(y=1.0, linestyle="--")

    plt.xticks(x, labels)
    plt.ylabel("Normalized Latency (MemArch / Baseline)")
    plt.xlabel("Workload group")
    plt.title("Normalized Latency Comparison (Lower is Better)")

    # annotate values
    for i, v in enumerate(normalized_vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()

    out_path = out_dir / "normalized_latency.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def write_summary_json(grouped: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Path:
    summary: Dict[str, Any] = {}
    for group_label, pair in grouped.items():
        summary[group_label] = {}
        for system, run in pair.items():
            summary[group_label][system] = {
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

    if not grouped:
        raise SystemExit("No grouped benchmark runs were found.")

    bar_path = write_bar_chart(grouped, out_dir)
    norm_path = write_normalized_bar_chart(grouped, out_dir)
    curve_paths = write_per_query_plots(grouped, out_dir)
    summary_path = write_summary_json(grouped, out_dir)

    print("Wrote:")
    print(f"- {bar_path}")
    for p in curve_paths:
        print(f"- {p}")
    print(f"- {summary_path}")
    print(f"- {norm_path}")


if __name__ == "__main__":
    main()