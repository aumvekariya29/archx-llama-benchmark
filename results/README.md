# Results Directory — Data Dictionary

## Raw JSON Files (`raw/`)

Each benchmark configuration produces one JSON file named:
`{platform}_{precision}_{test_type}_{parameter_value}.json`

Example: `M2_Q4_K_M_ttft_256.json`

### JSON Fields

| Field | Type | Description |
|-------|------|-------------|
| `platform` | string | Hardware platform: `M1`, `M2`, or `M4` |
| `precision` | string | Quantization level: `F16`, `Q8_0`, or `Q4_K_M` |
| `test_type` | string | One of: `ttft`, `per_token_latency`, `e2e_latency`, `throughput` |
| `parameter_name` | string | Variable being swept: `prompt_length`, `context_length`, or `output_length` |
| `parameter_value` | int | Value of the swept parameter (e.g., 64, 128, 256, 512, 1024) |
| `trial_result` | object | Statistical summary of timed trials (see below) |
| `timestamp` | string | ISO 8601 timestamp of when the benchmark was run |

### Trial Result Object

| Field | Type | Description |
|-------|------|-------------|
| `values` | float[] | Individual trial measurements after IQR outlier filtering |
| `median` | float | Median of filtered values |
| `p95` | float | 95th percentile |
| `p99` | float | 99th percentile |
| `std` | float | Standard deviation (ddof=1) |
| `unit` | string | `ms` for latency metrics, `tokens/s` for throughput |

### Test Types Explained

- **`ttft`** (Time to First Token): Time from submitting the prompt to receiving the first generated token. Measures prefill/prompt-processing speed. Swept across prompt lengths 64–1024 tokens.
- **`per_token_latency`**: Average time to generate each token after the first. Measures decode speed. Swept across context lengths 64–1024 tokens.
- **`e2e_latency`**: Total wall-clock time to generate a fixed number of output tokens from a 128-token prompt. Swept across output lengths 32, 64, 128.
- **`throughput`**: Tokens generated per second. Computed as `output_tokens / wall_time`. Swept across output lengths 32, 64, 128.

### Outlier Filtering

All measurements use IQR-based filtering with a 1.5x multiplier:
- Q1 = 25th percentile, Q3 = 75th percentile
- IQR = Q3 - Q1
- Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are removed
- `n_trials` in summary.csv shows how many of the original 10 trials survived filtering

## Summary CSV (`summary.csv`)

Aggregated view of all raw JSON files with one row per benchmark configuration.

### Columns

| Column | Description |
|--------|-------------|
| `platform` | M1, M2, or M4 |
| `precision` | F16, Q8_0, or Q4_K_M |
| `test_type` | ttft, per_token_latency, e2e_latency, or throughput |
| `parameter_name` | The variable being swept |
| `parameter_value` | The specific value tested |
| `median` | Median measurement (ms or tokens/s) |
| `p95` | 95th percentile |
| `p99` | 99th percentile |
| `std` | Standard deviation |
| `unit` | Measurement unit |
| `n_trials` | Number of trials after outlier filtering (out of 10) |
| `timestamp` | When the benchmark was run |

## Decomposition Files (`decomposition/`)

Named `{platform}_{precision}_decomp.json`. Contains per-component latency breakdown from PyTorch forward hooks.

### Component Keys

| Key | Component | Description |
|-----|-----------|-------------|
| `embedding` | Token Embedding | Lookup table, no matmul |
| `qkv_proj` | Q/K/V Projections | Three linear layers per attention block |
| `attn_core` | Attention Core | Softmax(QK^T)V computation (derived) |
| `o_proj` | Output Projection | Linear layer after attention |
| `mlp` | MLP Block | Gate + Up + Down projections with SiLU |
| `layernorm` | RMSNorm | All RMSNorm layers (2 per block + 1 final) |
| `lm_head` | Language Model Head | Final linear layer to vocabulary logits |
| `framework_overhead` | Framework Overhead | Wall time minus sum of instrumented components |
| `total` | Total | Full wall-clock time for all decode steps |

Each component includes: `total_ms`, `mean_per_step_ms`, `pct_of_total`, `arithmetic_intensity_flop_per_byte`, `flops_per_step`, `bytes_per_step`.

## Platform Detection Logic

The benchmarking code auto-detects which Apple Silicon chip is present:

```
sysctl -n machdep.cpu.brand_string
```

This returns a string like `"Apple M2"`. The code searches for `"m1"`, `"m2"`, or `"m4"` (case-insensitive) in this string to determine the platform. On non-ARM systems, it defaults to `M1`. The platform can always be overridden via `--platform`.
