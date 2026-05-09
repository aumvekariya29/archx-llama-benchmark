#!/usr/bin/env python3
"""
ArchX Benchmark Runner — LLaMA 3.2-1B Token Generation Latency

Measures TTFT, per-token latency, end-to-end latency, and throughput
across prompt/context/output lengths and quantization levels.
"""

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from llama_cpp import Llama


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

PROMPT_LENGTHS = [64, 128, 256, 512, 1024]
CONTEXT_LENGTHS = [64, 128, 256, 512, 1024]
OUTPUT_LENGTHS = [32, 64, 128]
WARMUP_RUNS = 3
TIMED_TRIALS = 10
IQR_MULTIPLIER = 1.5

PRECISION_FILES = {
    "F16": "Llama-3.2-1B-Instruct-f16.gguf",
    "Q8_0": "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "Q4_K_M": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}

SUPPORTED_PLATFORMS = {"M1", "M2", "M4"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    values: list[float]
    median: float
    p95: float
    p99: float
    std: float
    unit: str


@dataclass
class BenchmarkResult:
    platform: str
    precision: str
    test_type: str
    parameter_name: str
    parameter_value: int
    trial_result: dict
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    """Auto-detect platform from system info."""
    machine = platform.machine().lower()
    uname = platform.uname()

    if "arm" in machine or "aarch64" in machine:
        # Apple Silicon — try to distinguish chip
        try:
            import subprocess
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip().lower()
            if "m4" in chip:
                return "M4"
            if "m2" in chip:
                return "M2"
            if "m1" in chip:
                return "M1"
        except Exception:
            pass
        return "M1"  # fallback for Apple Silicon

    return "M1"  # fallback


def filter_outliers_iqr(data: list[float], multiplier: float = IQR_MULTIPLIER) -> list[float]:
    """Remove outliers using IQR method."""
    # IQR requires at least 4 points to compute meaningful quartiles
    if len(data) < 4:
        return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return [x for x in data if lower <= x <= upper]


def compute_stats(data: list[float], unit: str = "ms") -> dict:
    """Compute median, p95, p99, std from a list of measurements."""
    filtered = filter_outliers_iqr(data)
    if not filtered:
        filtered = data  # fallback if everything got filtered
    return {
        "values": [round(v, 4) for v in filtered],
        "median": round(float(np.median(filtered)), 4),
        "p95": round(float(np.percentile(filtered, 95)), 4),
        "p99": round(float(np.percentile(filtered, 99)), 4),
        "std": round(float(np.std(filtered, ddof=1)) if len(filtered) > 1 else 0.0, 4),
        "unit": unit,
    }


def build_prompt_tokens(llm: Llama, target_len: int) -> list[int]:
    """Build a token sequence of approximately `target_len` tokens."""
    # Repeat a simple sentence and tokenize, then truncate to exact length
    seed_text = "The quick brown fox jumps over the lazy dog. " * (target_len // 5 + 10)
    tokens = llm.tokenize(seed_text.encode("utf-8"), add_bos=True)
    return tokens[:target_len]


def load_model(model_path: str, platform_name: str, n_ctx: int = 2048) -> Llama:
    """Load model with appropriate backend settings."""
    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "verbose": False,
    }

    if platform_name in ("M1", "M2", "M4"):
        # Metal backend — offload all layers to GPU
        kwargs["n_gpu_layers"] = -1

    return Llama(**kwargs)


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def measure_ttft(llm: Llama, prompt_tokens: list[int]) -> float:
    """Measure time to first token (ms). Uses the streaming API."""
    llm.reset()

    # top_k=1, temp=0.0: greedy decoding for deterministic, reproducible timings
    start = time.perf_counter()
    generator = llm.generate(prompt_tokens, top_k=1, top_p=1.0, temp=0.0)
    _ = next(generator)  # first generated token
    elapsed = time.perf_counter() - start

    # Drain the rest to avoid state issues (generate only 1 more)
    try:
        next(generator)
    except StopIteration:
        pass

    return elapsed * 1000.0  # ms


def measure_per_token_latency(llm: Llama, prompt_tokens: list[int], gen_count: int = 32) -> float:
    """Measure average per-token latency (ms) after the first token."""
    llm.reset()

    generator = llm.generate(prompt_tokens, top_k=1, top_p=1.0, temp=0.0)

    # Skip first token (that's TTFT)
    _ = next(generator)

    times = []
    for _ in range(gen_count - 1):
        t0 = time.perf_counter()
        try:
            _ = next(generator)
        except StopIteration:
            break
        times.append((time.perf_counter() - t0) * 1000.0)

    return statistics.mean(times) if times else 0.0


def measure_e2e_latency(llm: Llama, prompt_tokens: list[int], output_len: int) -> float:
    """Measure end-to-end generation latency (ms) for `output_len` tokens."""
    llm.reset()

    start = time.perf_counter()
    generator = llm.generate(prompt_tokens, top_k=1, top_p=1.0, temp=0.0)
    count = 0
    for _ in generator:
        count += 1
        if count >= output_len:
            break
    elapsed = time.perf_counter() - start

    return elapsed * 1000.0


def measure_throughput(llm: Llama, prompt_tokens: list[int], output_len: int = 64) -> float:
    """Measure throughput in tokens/second."""
    llm.reset()

    start = time.perf_counter()
    generator = llm.generate(prompt_tokens, top_k=1, top_p=1.0, temp=0.0)
    count = 0
    for _ in generator:
        count += 1
        if count >= output_len:
            break
    elapsed = time.perf_counter() - start

    return count / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_trials(fn, trials: int | None = None, warmup: int | None = None, **kwargs) -> list[float]:
    """Run warmup + timed trials, returning the timed measurements."""
    trials = trials if trials is not None else TIMED_TRIALS
    warmup = warmup if warmup is not None else WARMUP_RUNS
    for _ in range(warmup):
        fn(**kwargs)
    results = []
    for _ in range(trials):
        results.append(fn(**kwargs))
    return results


def run_benchmark(
    model_path: str,
    platform_name: str,
    precision: str,
    output_dir: Path,
    prompt_lengths: Optional[list[int]] = None,
    context_lengths: Optional[list[int]] = None,
    output_lengths: Optional[list[int]] = None,
    n_ctx: int = 2048,
    trials: int | None = None,
    warmup: int | None = None,
):
    prompt_lengths = prompt_lengths or PROMPT_LENGTHS
    context_lengths = context_lengths or CONTEXT_LENGTHS
    output_lengths = output_lengths or OUTPUT_LENGTHS

    print(f"\n{'='*60}")
    print(f"  ArchX Benchmark — {platform_name} / {precision}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    llm = load_model(model_path, platform_name, n_ctx=n_ctx)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    all_results: list[dict] = []

    # --- TTFT across prompt lengths ---
    print("[1/4] Measuring TTFT...")
    for plen in prompt_lengths:
        print(f"  prompt_length={plen}", end=" ", flush=True)
        tokens = build_prompt_tokens(llm, plen)
        raw = run_trials(measure_ttft, trials=trials, warmup=warmup, llm=llm, prompt_tokens=tokens)
        stats = compute_stats(raw, unit="ms")
        all_results.append({
            "platform": platform_name,
            "precision": precision,
            "test_type": "ttft",
            "parameter_name": "prompt_length",
            "parameter_value": plen,
            "trial_result": stats,
            "timestamp": timestamp,
        })
        print(f"  median={stats['median']:.2f} ms")

    # --- Per-token latency across context lengths ---
    print("\n[2/4] Measuring per-token latency...")
    for clen in context_lengths:
        print(f"  context_length={clen}", end=" ", flush=True)
        tokens = build_prompt_tokens(llm, clen)
        raw = run_trials(measure_per_token_latency, trials=trials, warmup=warmup, llm=llm, prompt_tokens=tokens)
        stats = compute_stats(raw, unit="ms")
        all_results.append({
            "platform": platform_name,
            "precision": precision,
            "test_type": "per_token_latency",
            "parameter_name": "context_length",
            "parameter_value": clen,
            "trial_result": stats,
            "timestamp": timestamp,
        })
        print(f"  median={stats['median']:.2f} ms")

    # --- End-to-end latency across output lengths ---
    print("\n[3/4] Measuring end-to-end latency...")
    base_prompt = build_prompt_tokens(llm, 128)  # fixed prompt for e2e
    for olen in output_lengths:
        print(f"  output_length={olen}", end=" ", flush=True)
        raw = run_trials(measure_e2e_latency, trials=trials, warmup=warmup, llm=llm, prompt_tokens=base_prompt, output_len=olen)
        stats = compute_stats(raw, unit="ms")
        all_results.append({
            "platform": platform_name,
            "precision": precision,
            "test_type": "e2e_latency",
            "parameter_name": "output_length",
            "parameter_value": olen,
            "trial_result": stats,
            "timestamp": timestamp,
        })
        print(f"  median={stats['median']:.2f} ms")

    # --- Throughput ---
    print("\n[4/4] Measuring throughput...")
    for olen in output_lengths:
        print(f"  output_length={olen}", end=" ", flush=True)
        raw = run_trials(measure_throughput, trials=trials, warmup=warmup, llm=llm, prompt_tokens=base_prompt, output_len=olen)
        stats = compute_stats(raw, unit="tokens/s")
        all_results.append({
            "platform": platform_name,
            "precision": precision,
            "test_type": "throughput",
            "parameter_name": "output_length",
            "parameter_value": olen,
            "trial_result": stats,
            "timestamp": timestamp,
        })
        print(f"  median={stats['median']:.2f} tok/s")

    # --- Save results ---
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in all_results:
        pval = result["parameter_value"]
        ttype = result["test_type"]
        fname = f"{platform_name}_{precision}_{ttype}_{pval}.json"
        out_path = output_dir / fname
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    summary_path = output_dir / f"{platform_name}_{precision}_all.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"Summary: {summary_path}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArchX — LLaMA 3.2-1B Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--precision", "-p",
        choices=list(PRECISION_FILES.keys()),
        required=True,
        help="Quantization precision label (F16, Q8_0, Q4_K_M)",
    )
    parser.add_argument(
        "--platform",
        choices=list(SUPPORTED_PLATFORMS),
        default=None,
        help="Platform name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/raw",
        help="Directory for JSON result files (default: results/raw)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context window size (default: 2048)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TIMED_TRIALS,
        help=f"Number of timed trials per config (default: {TIMED_TRIALS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help=f"Number of warm-up runs (default: {WARMUP_RUNS})",
    )

    args = parser.parse_args()

    plat = args.platform or detect_platform()

    run_benchmark(
        model_path=args.model,
        platform_name=plat,
        precision=args.precision,
        output_dir=Path(args.output_dir),
        n_ctx=args.n_ctx,
        trials=args.trials,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
