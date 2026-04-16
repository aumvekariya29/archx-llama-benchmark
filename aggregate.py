#!/usr/bin/env python3
"""
ArchX Results Aggregator

Compiles all per-config JSON files from results/raw/ into results/summary.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(raw_dir: Path) -> list[dict]:
    """Load all individual benchmark JSON files (skip *_all.json summaries)."""
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.stem.endswith("_all"):
            continue
        with open(path) as f:
            data = json.load(f)
        trial = data.get("trial_result", {})
        records.append({
            "platform": data["platform"],
            "precision": data["precision"],
            "test_type": data["test_type"],
            "parameter_name": data["parameter_name"],
            "parameter_value": data["parameter_value"],
            "median": trial.get("median"),
            "p95": trial.get("p95"),
            "p99": trial.get("p99"),
            "std": trial.get("std"),
            "unit": trial.get("unit"),
            "n_trials": len(trial.get("values", [])),
            "timestamp": data.get("timestamp", ""),
        })
    return records


def aggregate(raw_dir: Path, output_path: Path) -> None:
    """Build summary CSV from raw JSON results."""
    records = load_results(raw_dir)
    if not records:
        print(f"No result files found in {raw_dir}")
        return

    df = pd.DataFrame(records)

    # Sort for readability
    df = df.sort_values(
        ["platform", "precision", "test_type", "parameter_name", "parameter_value"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Aggregated {len(df)} records -> {output_path}")
    print(f"\nPlatforms: {sorted(df['platform'].unique())}")
    print(f"Precisions: {sorted(df['precision'].unique())}")
    print(f"Test types: {sorted(df['test_type'].unique())}")
    print(f"\nPreview:")
    print(df.to_string(max_rows=20))


def main() -> None:
    parser = argparse.ArgumentParser(description="ArchX — Aggregate benchmark results")
    parser.add_argument(
        "--raw-dir",
        default="results/raw",
        help="Directory containing raw JSON files (default: results/raw)",
    )
    parser.add_argument(
        "--output",
        default="results/summary.csv",
        help="Output CSV path (default: results/summary.csv)",
    )
    args = parser.parse_args()
    aggregate(Path(args.raw_dir), Path(args.output))


if __name__ == "__main__":
    main()
