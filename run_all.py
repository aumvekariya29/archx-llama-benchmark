#!/usr/bin/env python3
"""
ArchX — Single Entry Point CLI

Orchestrates all benchmark phases:
  Phase 1 (benchmark): Measure TTFT, PTL, E2E latency, and throughput
  Phase 2 (decompose): Per-component latency decomposition via forward hooks
  Phase 3 (visualize): Generate publication-quality plots

Usage:
  python run_all.py                                    # Run everything, auto-detect platform
  python run_all.py --platform M2 --precision q4       # Single precision on M2
  python run_all.py --phase visualize                  # Only regenerate plots
  python run_all.py --phase benchmark --trials 5       # Quick benchmark with fewer trials
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_default_config_path() -> Path:
    """Return path to default config relative to this script."""
    return Path(__file__).parent / "configs" / "default_config.yaml"


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    """Auto-detect Apple Silicon chip from system info."""
    import platform as plat
    machine = plat.machine().lower()
    if "arm" not in machine and "aarch64" not in machine:
        return "M1"  # fallback
    try:
        result = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip().lower()
        if "m4" in result:
            return "M4"
        if "m2" in result:
            return "M2"
        if "m1" in result:
            return "M1"
    except Exception:
        pass
    return "M1"


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

PRECISION_MAP = {
    "f16": ("F16", "Llama-3.2-1B-Instruct-f16.gguf"),
    "q8": ("Q8_0", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
    "q4": ("Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}


def run_benchmark(
    platform: str,
    precisions: list[str],
    models_dir: Path,
    output_dir: Path,
    trials: int,
    warmup: int,
) -> None:
    """Run Phase 1 benchmarks for specified precisions."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Benchmark — {platform}")
    print(f"  Precisions: {', '.join(precisions)}")
    print(f"  Trials: {trials}, Warmup: {warmup}")
    print(f"{'='*60}\n")

    raw_dir = output_dir / "raw"
    for prec_key in precisions:
        prec_label, filename = PRECISION_MAP[prec_key]
        model_path = models_dir / filename

        if not model_path.exists():
            print(f"  [SKIP] Model not found: {model_path}")
            print(f"         Download with: python run_all.py --download-models")
            continue

        cmd = [
            sys.executable, "benchmark.py",
            "-m", str(model_path),
            "-p", prec_label,
            "--platform", platform,
            "-o", str(raw_dir),
            "--trials", str(trials),
            "--warmup", str(warmup),
        ]
        print(f"  Running: {prec_label}...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [ERROR] Benchmark failed for {prec_label}")

    # Aggregate results
    print("\n  Aggregating results...")
    subprocess.run([sys.executable, "aggregate.py",
                    "--raw-dir", str(raw_dir),
                    "--output", str(output_dir / "summary.csv")])


def run_decompose(
    platform: str,
    output_dir: Path,
) -> None:
    """Run Phase 2 decomposition."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: Latency Decomposition — {platform}")
    print(f"{'='*60}\n")

    decomp_dir = output_dir / "decomposition"
    raw_dir = output_dir / "raw"

    cmd = [
        sys.executable, "decompose.py",
        "--platform", platform,
        "-o", str(decomp_dir),
        "--simulate-q4",
        "--raw-dir", str(raw_dir),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  [ERROR] Decomposition failed")


def run_visualize(output_dir: Path) -> None:
    """Run Phase 3 visualization."""
    print(f"\n{'='*60}")
    print(f"  Phase 3: Generating Plots")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "visualize.py",
        "--summary", str(output_dir / "summary.csv"),
        "--decomp-dir", str(output_dir / "decomposition"),
        "--plots-dir", "plots",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  [ERROR] Visualization failed")


def download_models(models_dir: Path) -> None:
    """Download all GGUF model files from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"  Downloading Models to {models_dir}/")
    print(f"{'='*60}\n")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [ERROR] huggingface-hub not installed. Run: pip install huggingface-hub")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    repo = "bartowski/Llama-3.2-1B-Instruct-GGUF"

    for prec_key, (_, filename) in PRECISION_MAP.items():
        target = models_dir / filename
        if target.exists():
            print(f"  [SKIP] Already exists: {filename}")
            continue
        print(f"  Downloading {filename}...")
        try:
            hf_hub_download(repo, filename, local_dir=str(models_dir))
            print(f"  Done: {filename}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArchX — LLaMA 3.2-1B Benchmark Suite for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                              # Full suite, auto-detect platform
  python run_all.py --platform M2 --precision q4 # Q4 only on M2
  python run_all.py --phase visualize            # Regenerate plots only
  python run_all.py --phase benchmark --trials 3 # Quick test run
  python run_all.py --download-models            # Download GGUF files
        """,
    )
    parser.add_argument(
        "--platform",
        choices=["M1", "M2", "M4"],
        default=None,
        help="Target platform (auto-detected if omitted)",
    )
    parser.add_argument(
        "--precision",
        choices=["f16", "q8", "q4", "all"],
        default="all",
        help="Quantization precision to benchmark (default: all)",
    )
    parser.add_argument(
        "--phase",
        choices=["benchmark", "decompose", "visualize", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory (default: results/)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing GGUF model files (default: models/)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of timed trials per config (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up runs (default: 3)",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download GGUF model files from HuggingFace and exit",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: configs/default_config.yaml)",
    )

    args = parser.parse_args()

    # Handle model download
    if args.download_models:
        download_models(Path(args.models_dir))
        return

    # Detect platform
    platform = args.platform or detect_platform()
    print(f"\nArchX Benchmark Suite")
    print(f"Platform: {platform} ({'auto-detected' if not args.platform else 'specified'})")

    # Resolve precisions
    if args.precision == "all":
        precisions = ["f16", "q8", "q4"]
    else:
        precisions = [args.precision]

    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    phase = args.phase

    start_time = time.time()

    # Run phases
    if phase in ("benchmark", "all"):
        run_benchmark(platform, precisions, models_dir, output_dir, args.trials, args.warmup)

    if phase in ("decompose", "all"):
        run_decompose(platform, output_dir)

    if phase in ("visualize", "all"):
        run_visualize(output_dir)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nCompleted in {minutes}m {seconds}s")
    print(f"Results: {output_dir}/")
    print(f"Plots:   plots/")


if __name__ == "__main__":
    main()
