#!/usr/bin/env python3
"""
ArchX Phase 3 — Research Paper Visualization

Generates 8 publication-quality plots from Phase 1 (summary.csv)
and Phase 2 (decomposition JSONs) results.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------

PLATFORM_COLORS = {
    "M1": "#2E5090",
    "M2": "#C0392B",
    "M4": "#27AE60",
}
PLATFORM_ORDER = ["M1", "M2", "M4"]
PRECISION_ORDER = ["F16", "Q8_0", "Q4_K_M"]
PRECISION_LABELS = {"F16": "FP16", "Q8_0": "Q8", "Q4_K_M": "Q4"}

COMPONENT_ORDER = ["MLP", "QKV Proj", "Attn Core", "LayerNorm", "LM Head", "O Proj", "Embedding", "Overhead"]
COMPONENT_COLORS = {
    "MLP": "#3498DB",
    "QKV Proj": "#E74C3C",
    "Attn Core": "#F39C12",
    "LayerNorm": "#2ECC71",
    "LM Head": "#9B59B6",
    "O Proj": "#1ABC9C",
    "Embedding": "#95A5A6",
    "Overhead": "#BDC3C7",
}
COMP_KEY_TO_LABEL = {
    "mlp": "MLP",
    "qkv_proj": "QKV Proj",
    "attn_core": "Attn Core",
    "layernorm": "LayerNorm",
    "lm_head": "LM Head",
    "o_proj": "O Proj",
    "embedding": "Embedding",
    "framework_overhead": "Overhead",
}

# Peak memory bandwidth (GB/s)
PLATFORM_PEAK_BW = {
    "M1": 68.25,
    "M2": 100.0,
    "M4": 120.0,
}
# Peak compute (GFLOP/s, FP16)
PLATFORM_PEAK_FLOPS = {
    "M1": 2600.0,
    "M2": 3600.0,
    "M4": 4600.0,
}

FIG_SIZE = (8, 5)
DPI = 300


def setup_style() -> None:
    """Configure academic-style matplotlib defaults."""
    plt.rcParams.update({
        "figure.figsize": FIG_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.6,
        "legend.handletextpad": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
    })
    try:
        matplotlib.font_manager.findfont("Times New Roman", fallback_to_default=False)
        plt.rcParams["font.serif"] = ["Times New Roman"]
    except Exception:
        pass


def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_decomposition(decomp_dir: Path) -> dict:
    data = {}
    for p in decomp_dir.glob("*_decomp.json"):
        with open(p) as f:
            d = json.load(f)
        key = (d["platform"], d["precision"])
        data[key] = d
    return data


def available_platforms(df: pd.DataFrame) -> list[str]:
    present = set(df["platform"].unique())
    return [p for p in PLATFORM_ORDER if p in present]


def color_for(platform: str) -> str:
    return PLATFORM_COLORS.get(platform, "#555555")


# ---------------------------------------------------------------------------
# Plot 1: TTFT vs Prompt Length
# ---------------------------------------------------------------------------

def plot_ttft(df: pd.DataFrame, plots_dir: Path) -> None:
    ttft = df[df["test_type"] == "ttft"].copy()
    platforms = available_platforms(ttft)
    precisions = [p for p in PRECISION_ORDER if p in ttft["precision"].unique()]

    ncols = len(precisions) if len(precisions) > 1 else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True, squeeze=False)

    for i, prec in enumerate(precisions):
        ax = axes[0, i]
        sub = ttft[ttft["precision"] == prec]
        for plat in platforms:
            d = sub[sub["platform"] == plat].sort_values("parameter_value")
            if d.empty:
                continue
            ax.plot(d["parameter_value"], d["median"],
                    marker="o", markersize=6, linewidth=2.0,
                    color=color_for(plat), label=plat)
        ax.set_xlabel("Prompt Length (tokens)")
        if i == 0:
            ax.set_ylabel("TTFT (ms)")
        ax.set_title(PRECISION_LABELS.get(prec, prec), fontsize=12, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.set_xticklabels(["64", "128", "256", "512", "1024"])
        ax.tick_params(axis="x", rotation=0)
        ax.legend(loc="upper left")

    fig.suptitle("Time to First Token vs Prompt Length", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(plots_dir / "ttft_vs_prompt_length.png")
    plt.close(fig)
    print("  [1/8] ttft_vs_prompt_length.png")


# ---------------------------------------------------------------------------
# Plot 2: PTL vs Context Length
# ---------------------------------------------------------------------------

def plot_ptl(df: pd.DataFrame, plots_dir: Path) -> None:
    ptl = df[(df["test_type"] == "per_token_latency") & (df["precision"] == "Q4_K_M")].copy()
    platforms = available_platforms(ptl)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for plat in platforms:
        d = ptl[ptl["platform"] == plat].sort_values("parameter_value")
        if d.empty:
            continue
        x = d["parameter_value"].values
        y = d["median"].values
        std = d["std"].values
        lo = y - 0.675 * std
        hi = y + 0.675 * std

        ax.plot(x, y, marker="s", markersize=6, linewidth=2.0,
                color=color_for(plat), label=plat, zorder=3)
        ax.fill_between(x, lo, hi, alpha=0.18, color=color_for(plat), zorder=2)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Per-Token Latency (ms)")
    ax.set_title("Per-Token Latency vs Context Length (Q4_K_M)", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024])
    ax.set_xticklabels(["64", "128", "256", "512", "1024"])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "ptl_vs_context_length.png")
    plt.close(fig)
    print("  [2/8] ptl_vs_context_length.png")


# ---------------------------------------------------------------------------
# Plot 3: Throughput Comparison
# ---------------------------------------------------------------------------

def plot_throughput(df: pd.DataFrame, plots_dir: Path) -> None:
    tp = df[(df["test_type"] == "throughput") & (df["parameter_value"] == 128)].copy()
    platforms = available_platforms(tp)
    precisions = [p for p in PRECISION_ORDER if p in tp["precision"].unique()]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    n_plat = len(platforms)
    n_prec = len(precisions)

    # Adaptive bar width: tighter grouping when few platforms
    group_width = min(0.8, 0.25 * n_prec)
    bar_width = group_width / n_prec
    x = np.arange(n_plat)

    palette = ["#66C2A5", "#FC8D62", "#8DA0CB"]

    for j, prec in enumerate(precisions):
        vals = []
        for plat in platforms:
            row = tp[(tp["platform"] == plat) & (tp["precision"] == prec)]
            vals.append(row["median"].values[0] if len(row) else 0)
        offset = (j - n_prec / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.88,
                      label=PRECISION_LABELS.get(prec, prec),
                      color=palette[j % len(palette)],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Platform")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput Comparison (output_length = 128)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(title="Precision", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "throughput_comparison.png")
    plt.close(fig)
    print("  [3/8] throughput_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4: Quantization Speedup
# ---------------------------------------------------------------------------

def plot_quantization_speedup(df: pd.DataFrame, plots_dir: Path) -> None:
    ptl = df[(df["test_type"] == "per_token_latency") & (df["parameter_value"] == 128)].copy()
    platforms = available_platforms(ptl)

    speedups = []
    for plat in platforms:
        f16 = ptl[(ptl["platform"] == plat) & (ptl["precision"] == "F16")]
        q4 = ptl[(ptl["platform"] == plat) & (ptl["precision"] == "Q4_K_M")]
        if len(f16) and len(q4):
            ratio = f16["median"].values[0] / q4["median"].values[0]
            speedups.append((plat, ratio))

    if not speedups:
        print("  [4/8] quantization_speedup.png — SKIPPED (need F16 + Q4 data)")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plats = [s[0] for s in speedups]
    vals = [s[1] for s in speedups]
    x = np.arange(len(plats))

    bar_width = min(0.5, 0.35 * max(len(plats), 1))
    bars = ax.bar(x, vals, bar_width,
                  color=[color_for(p) for p in plats],
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{v:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Reference lines — always show both, adjust y-axis accordingly
    max_val = max(max(vals), 4.5)
    ax.set_ylim(0, max_val * 1.2)

    ax.axhline(y=2.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(x[-1] + bar_width / 2 + 0.05, 2.0, " 2.0x theoretical (FP16/Q8)",
            fontsize=8, color="#666666", va="center", ha="left")

    ax.axhline(y=4.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(x[-1] + bar_width / 2 + 0.05, 4.0, " 4.0x theoretical (FP16/Q4)",
            fontsize=8, color="#666666", va="center", ha="left")

    ax.set_xlabel("Platform")
    ax.set_ylabel("Speedup (FP16 / Q4_K_M)")
    ax.set_title("Quantization Speedup: Q4_K_M vs FP16", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plats)

    # Add right margin for reference labels
    ax.margins(x=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "quantization_speedup.png")
    plt.close(fig)
    print("  [4/8] quantization_speedup.png")


# ---------------------------------------------------------------------------
# Plot 5 & 6: Component Breakdown (stacked horizontal bar)
# ---------------------------------------------------------------------------

def _plot_component_breakdown(decomp_data: dict, precision_filter: str,
                              title: str, filename: str, plots_dir: Path) -> None:
    entries = {}
    for (plat, prec), data in decomp_data.items():
        if prec == precision_filter:
            entries[plat] = data

    if not entries:
        print(f"  {filename} — SKIPPED (no {precision_filter} decomposition data)")
        return

    platforms = [p for p in PLATFORM_ORDER if p in entries]
    n_plat = len(platforms)

    # Adaptive figure height
    fig_h = max(4, n_plat * 1.8 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_height = 0.55 if n_plat > 1 else 0.4
    y_pos = np.arange(n_plat)
    lefts = np.zeros(n_plat)

    handles = []
    labels_list = []

    for comp_label in COMPONENT_ORDER:
        json_key = None
        for k, v in COMP_KEY_TO_LABEL.items():
            if v == comp_label:
                json_key = k
                break
        if json_key is None:
            continue

        pcts = []
        for plat in platforms:
            comp_data = entries[plat].get("components", {}).get(json_key, {})
            pcts.append(comp_data.get("pct_of_total", 0))

        pcts = np.array(pcts)
        bar = ax.barh(y_pos, pcts, left=lefts, height=bar_height,
                      color=COMPONENT_COLORS.get(comp_label, "#999"),
                      edgecolor="white", linewidth=0.8)
        handles.append(bar[0])
        labels_list.append(comp_label)

        # Annotate segments wider than 8%
        for i, (pct, left) in enumerate(zip(pcts, lefts)):
            if pct > 8:
                ax.text(left + pct / 2, y_pos[i], f"{pct:.1f}%",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
            elif pct > 4:
                ax.text(left + pct / 2, y_pos[i], f"{pct:.0f}%",
                        ha="center", va="center", fontsize=7, color="white")
        lefts += pcts

    ax.set_yticks(y_pos)
    ax.set_yticklabels(platforms, fontsize=12, fontweight="bold")
    ax.set_xlabel("% of Total Latency")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlim(0, 105)

    # Place legend below the chart with generous spacing
    ax.legend(handles, labels_list,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.28),
              ncol=4,
              fontsize=9,
              columnspacing=1.5,
              handletextpad=0.8,
              handlelength=1.5,
              borderpad=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(plots_dir / filename)
    plt.close(fig)


def plot_component_f16(decomp_data: dict, plots_dir: Path) -> None:
    _plot_component_breakdown(
        decomp_data, "float16",
        "Component Latency Breakdown — FP16",
        "component_breakdown_f16.png", plots_dir,
    )
    print("  [5/8] component_breakdown_f16.png")


def plot_component_q4(decomp_data: dict, plots_dir: Path) -> None:
    _plot_component_breakdown(
        decomp_data, "Q4_simulated",
        "Component Latency Breakdown — Q4 (Simulated)",
        "component_breakdown_q4.png", plots_dir,
    )
    print("  [6/8] component_breakdown_q4.png")


# ---------------------------------------------------------------------------
# Plot 7: Roofline Model
# ---------------------------------------------------------------------------

def plot_roofline(decomp_data: dict, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ai_range = np.logspace(-1, 2, 500)

    # Draw roofline for each platform
    platforms_plotted = []
    for plat in PLATFORM_ORDER:
        peak_bw = PLATFORM_PEAK_BW.get(plat)
        peak_flops = PLATFORM_PEAK_FLOPS.get(plat)
        if peak_bw is None or peak_flops is None:
            continue
        roofline = np.minimum(peak_flops, ai_range * peak_bw)
        ax.plot(ai_range, roofline, linewidth=2.0, color=color_for(plat),
                label=f"{plat} ({peak_flops:.0f} GFLOP/s, {peak_bw:.0f} GB/s)",
                alpha=0.7)
        platforms_plotted.append(plat)

    # Plot component dots
    markers = {"MLP": "o", "QKV Proj": "s", "Attn Core": "D",
               "LayerNorm": "^", "LM Head": "v", "O Proj": "P"}

    # Collect all points for smart label placement
    points = []

    for (plat, prec), data in decomp_data.items():
        if prec != "float16":
            continue
        if plat not in platforms_plotted:
            continue

        comps = data.get("components", {})
        decode_steps = data.get("decode_steps", 30)

        for json_key, display_label in COMP_KEY_TO_LABEL.items():
            if json_key in ("framework_overhead", "embedding"):
                continue
            comp = comps.get(json_key, {})
            ai = comp.get("arithmetic_intensity_flop_per_byte")
            flops = comp.get("flops_per_step")
            total_ms = comp.get("total_ms", 0)

            if ai is None or ai == 0 or flops is None or total_ms == 0:
                continue

            achieved_gflops = (flops * decode_steps) / (total_ms * 1e-3) / 1e9
            points.append((ai, achieved_gflops, display_label, plat))

            ax.scatter(ai, achieved_gflops, marker=markers.get(display_label, "o"),
                       s=80, color=color_for(plat), zorder=5, edgecolors="black",
                       linewidths=0.6)

    # Smart label placement: offset labels to avoid overlap
    # Sort by y-value and stagger offsets for points at same AI
    label_offsets = {
        "MLP": (10, 8),
        "LM Head": (10, -14),
        "O Proj": (-45, -18),
        "QKV Proj": (10, -10),
        "Attn Core": (10, 8),
        "LayerNorm": (10, 8),
    }

    for ai, gflops, label, plat in points:
        ox, oy = label_offsets.get(label, (8, 5))
        ax.annotate(
            label,
            (ai, gflops),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8,
            fontweight="bold",
            color=color_for(plat),
            arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.5),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline Model — LLaMA 3.2-1B Decode (FP16)", fontweight="bold")
    ax.set_xlim(0.1, 100)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "roofline_plot.png")
    plt.close(fig)
    print("  [7/8] roofline_plot.png")


# ---------------------------------------------------------------------------
# Plot 8: Bandwidth Utilization
# ---------------------------------------------------------------------------

def plot_bandwidth_utilization(df: pd.DataFrame, decomp_data: dict, plots_dir: Path) -> None:
    PARAM_COUNT = 1.24e9
    BYTES_PER_PARAM = {"F16": 2.0, "Q8_0": 1.0, "Q4_K_M": 0.5}

    ptl = df[(df["test_type"] == "per_token_latency") & (df["parameter_value"] == 128)].copy()
    platforms = available_platforms(ptl)
    precisions = [p for p in PRECISION_ORDER if p in ptl["precision"].unique()]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    n_plat = len(platforms)
    n_prec = len(precisions)

    group_width = min(0.8, 0.25 * n_prec)
    bar_width = group_width / n_prec
    x = np.arange(n_plat)

    palette = ["#66C2A5", "#FC8D62", "#8DA0CB"]

    for j, prec in enumerate(precisions):
        utils = []
        for plat in platforms:
            row = ptl[(ptl["platform"] == plat) & (ptl["precision"] == prec)]
            peak_bw = PLATFORM_PEAK_BW.get(plat, 100)
            if len(row):
                ptl_s = row["median"].values[0] / 1000.0
                model_bytes = PARAM_COUNT * BYTES_PER_PARAM.get(prec, 2.0)
                effective_bw_gbs = (model_bytes / ptl_s) / 1e9
                util_pct = (effective_bw_gbs / peak_bw) * 100.0
                utils.append(util_pct)
            else:
                utils.append(0)

        offset = (j - n_prec / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, utils, bar_width * 0.88,
                      label=PRECISION_LABELS.get(prec, prec),
                      color=palette[j % len(palette)],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, utils):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=100, color="#C0392B", linestyle="-", linewidth=1.2, alpha=0.8, zorder=1)
    ax.text(x[-1] + group_width / 2 + 0.05, 100, " 100% ceiling",
            fontsize=9, color="#C0392B", va="center", ha="left", fontweight="bold")

    ax.set_xlabel("Platform")
    ax.set_ylabel("Bandwidth Utilization (%)")
    ax.set_title("Effective Memory Bandwidth Utilization", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.set_ylim(0, 120)
    ax.margins(x=0.3)
    ax.legend(title="Precision", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "bandwidth_utilization.png")
    plt.close(fig)
    print("  [8/8] bandwidth_utilization.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ArchX Phase 3 — Generate research plots")
    parser.add_argument("--summary", default="results/summary.csv", help="Path to summary CSV")
    parser.add_argument("--decomp-dir", default="results/decomposition", help="Decomposition JSON dir")
    parser.add_argument("--plots-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"\nArchX Phase 3 — Generating plots to {plots_dir}/\n")

    df = load_summary(Path(args.summary))
    decomp = load_decomposition(Path(args.decomp_dir))

    print(f"  Platforms in data: {available_platforms(df)}")
    print(f"  Precisions in data: {sorted(df['precision'].unique())}")
    print(f"  Decomposition files: {list(decomp.keys())}\n")

    plot_ttft(df, plots_dir)
    plot_ptl(df, plots_dir)
    plot_throughput(df, plots_dir)
    plot_quantization_speedup(df, plots_dir)
    plot_component_f16(decomp, plots_dir)
    plot_component_q4(decomp, plots_dir)
    plot_roofline(decomp, plots_dir)
    plot_bandwidth_utilization(df, decomp, plots_dir)

    print(f"\nDone. {len(list(plots_dir.glob('*.png')))} plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
