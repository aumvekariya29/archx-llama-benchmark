# ArchX: Quantization-Aware LLaMA Latency Benchmarking Across Apple Silicon

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![llama-cpp-python](https://img.shields.io/badge/llama--cpp--python-0.2.90%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M4-000000?logo=apple&logoColor=white)

A comprehensive benchmarking harness for measuring LLaMA 3.2-1B token generation latency across Apple Silicon chips (M1, M2, M4) at multiple quantization levels (FP16, Q8, Q4). Built for reproducible research with per-component latency decomposition and publication-quality visualizations.

**Team:** Aum Vekariya, Janam Rangani, Aarav Patel — CSULB CECS 551

---

## Key Findings

| Metric (median, Q4_K_M) | M1 | M2 | M4 |
|---|---|---|---|
| Per-Token Latency @ ctx=128 | — | 12.06 ms | — |
| Throughput @ out=128 | — | 76.7 tok/s | — |
| Quantization Speedup (FP16/Q4) | — | 2.30x | — |
| BW Utilization (FP16) | — | 89.4% | — |

*Table auto-updates as results from additional platforms are added.*

## Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **TTFT** | Time to first token across prompt lengths 64–1024 | ms |
| **Per-Token Latency** | Average decode latency per generated token | ms |
| **End-to-End Latency** | Total generation time for 32/64/128 output tokens | ms |
| **Throughput** | Token generation rate | tokens/s |

Each configuration runs **3 warm-up + 10 timed trials** with **IQR-based outlier filtering** (1.5x multiplier). Reports median, p95, p99, and standard deviation.

## Setup

### Prerequisites

- macOS on Apple Silicon (M1, M2, or M4)
- Python 3.10+
- Xcode Command Line Tools (`xcode-select --install`)

### 1. Clone and Install

```bash
git clone https://github.com/your-repo/archx-llama-benchmark.git
cd archx-llama-benchmark

python3 -m venv .venv && source .venv/bin/activate

# Install llama-cpp-python with Metal (GPU) support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Install all other dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Login to HuggingFace (required for gated models)
huggingface-cli login

# Download all 3 quantizations automatically
python run_all.py --download-models
```

Or manually:

```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-f16.gguf" \
  --include "Llama-3.2-1B-Instruct-Q8_0.gguf" \
  --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

## Running Benchmarks

### Full Suite (recommended)

```bash
# Auto-detects platform, runs all 3 phases for all precisions
python run_all.py
```

### Selective Runs

```bash
# Single precision
python run_all.py --precision q4

# Specific phase only
python run_all.py --phase benchmark
python run_all.py --phase decompose
python run_all.py --phase visualize

# Quick test (fewer trials)
python run_all.py --phase benchmark --trials 3 --warmup 1

# Explicit platform
python run_all.py --platform M4
```

### Individual Scripts

```bash
# Phase 1: Benchmark
python benchmark.py -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p Q4_K_M

# Phase 1: Aggregate
python aggregate.py

# Phase 2: Decomposition
python decompose.py --simulate-q4

# Phase 3: Visualization
python visualize.py
```

## Reproducing Paper Results

### On Apple M1

```bash
python run_all.py --platform M1
```

### On Apple M2

```bash
python run_all.py --platform M2
```

### On Apple M4

```bash
python run_all.py --platform M4
```

### Merging Cross-Platform Results

After running on each machine, copy `results/raw/*.json` and `results/decomposition/*.json` to one machine, then:

```bash
python aggregate.py
python visualize.py
```

## Project Structure

```
archx-llama-benchmark/
├── run_all.py               # Single entry point CLI
├── benchmark.py              # Phase 1: TTFT, PTL, E2E, throughput measurement
├── aggregate.py              # Compile raw JSONs into summary CSV
├── decompose.py              # Phase 2: Per-component latency decomposition
├── visualize.py              # Phase 3: Generate 8 publication-quality plots
│
├── configs/
│   └── default_config.yaml   # All benchmark parameters (no hardcoded values)
│
├── models/                   # GGUF model files (not tracked in git)
│   ├── Llama-3.2-1B-Instruct-f16.gguf
│   ├── Llama-3.2-1B-Instruct-Q8_0.gguf
│   └── Llama-3.2-1B-Instruct-Q4_K_M.gguf
│
├── results/
│   ├── raw/                  # Per-config JSON results
│   ├── decomposition/        # Component breakdown JSONs
│   ├── summary.csv           # Aggregated results table
│   └── README.md             # Data dictionary
│
├── plots/                    # Generated figures (300 DPI PNG)
│   ├── ttft_vs_prompt_length.png
│   ├── ptl_vs_context_length.png
│   ├── throughput_comparison.png
│   ├── quantization_speedup.png
│   ├── component_breakdown_f16.png
│   ├── component_breakdown_q4.png
│   ├── roofline_plot.png
│   └── bandwidth_utilization.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Generated Plots

| Plot | Description |
|------|-------------|
| `ttft_vs_prompt_length.png` | TTFT vs prompt length (64–1024), one subplot per precision |
| `ptl_vs_context_length.png` | Per-token latency vs context length at Q4, with p25-p75 band |
| `throughput_comparison.png` | Grouped bar chart: throughput across platforms and precisions |
| `quantization_speedup.png` | Q4/FP16 speedup ratio with theoretical reference lines |
| `component_breakdown_f16.png` | Stacked bar: % latency per component at FP16 |
| `component_breakdown_q4.png` | Stacked bar: % latency per component at Q4 |
| `roofline_plot.png` | Roofline model with component dots per platform |
| `bandwidth_utilization.png` | Effective memory bandwidth utilization (%) |

## Configuration

All parameters are defined in `configs/default_config.yaml`:

- Benchmark sweep ranges (prompt/context/output lengths)
- Trial counts and warmup iterations
- Model file paths and HuggingFace repo IDs
- Visualization settings (DPI, colors, peak hardware specs)
- Model dimensions for arithmetic intensity calculations

## License

This project is for academic research purposes (CSULB CECS 551).
