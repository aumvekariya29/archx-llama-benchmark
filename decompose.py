#!/usr/bin/env python3
"""
ArchX Phase 2 — Per-Component Latency Decomposition

Instruments LLaMA 3.2-1B via HuggingFace Transformers forward hooks to
measure per-component latency breakdown during autoregressive decoding.

Components tracked:
  embedding, qkv_proj, attn_core, o_proj, mlp, layernorm, lm_head

Also computes arithmetic intensity (FLOP/byte) per component.
"""

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# LLaMA 3.2-1B model dimensions
# ---------------------------------------------------------------------------

MODEL_DIMS = {
    "layers": 16,
    "hidden": 2048,
    "heads": 32,
    "kv_heads": 8,
    "head_dim": 64,
    "ffn_intermediate": 8192,
    "vocab": 128256,
}

DECODE_STEPS = 30
HF_MODEL_ID = "meta-llama/Llama-3.2-1B"


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip().lower()
            if "m4" in chip:
                return "M4"
            if "m2" in chip:
                return "M2"
            if "m1" in chip:
                return "M1"
        except Exception:
            pass
        return "M1"
    return "M1"  # fallback


def get_device(platform_name: str) -> torch.device:
    if platform_name in ("M1", "M2", "M4"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize GPU to get accurate timings."""
    if device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Hook-based profiler
# ---------------------------------------------------------------------------

class ComponentProfiler:
    """Attaches forward hooks to LLaMA components and records per-call timings."""

    # Mapping from module name patterns to component labels
    COMPONENT_RULES = [
        ("model.embed_tokens", "embedding"),
        ("lm_head", "lm_head"),
        # Per-layer components (matched by suffix)
        ("self_attn.q_proj", "qkv_proj"),
        ("self_attn.k_proj", "qkv_proj"),
        ("self_attn.v_proj", "qkv_proj"),
        ("self_attn.o_proj", "o_proj"),
        ("mlp.gate_proj", "mlp"),
        ("mlp.up_proj", "mlp"),
        ("mlp.down_proj", "mlp"),
        ("input_layernorm", "layernorm"),
        ("post_attention_layernorm", "layernorm"),
        ("model.norm", "layernorm"),
    ]

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.device = device
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._hooks = []
        self._pending_start: dict[str, float] = {}
        self._attach_hooks(model)

    def _classify(self, module_name: str) -> str | None:
        for pattern, label in self.COMPONENT_RULES:
            if module_name.endswith(pattern) or module_name == pattern:
                return label
        return None

    def _attach_hooks(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            label = self._classify(name)
            if label is None:
                continue

            # Capture label by value via default arg
            def make_pre_hook(lbl):
                def pre_hook(mod, inp):
                    sync_device(self.device)
                    # id(mod) makes the key unique per module instance so hooks
                    # across 16 layers with the same label don't overwrite each other
                    self._pending_start[lbl + str(id(mod))] = time.perf_counter()
                return pre_hook

            def make_post_hook(lbl):
                def post_hook(mod, inp, out):
                    sync_device(self.device)
                    key = lbl + str(id(mod))
                    start = self._pending_start.pop(key, None)
                    if start is not None:
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        self.timings[lbl].append(elapsed_ms)
                return post_hook

            h1 = module.register_forward_pre_hook(make_pre_hook(label))
            h2 = module.register_forward_hook(make_post_hook(label))
            self._hooks.extend([h1, h2])

    def reset(self):
        self.timings.clear()
        self._pending_start.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_summary(self) -> dict[str, dict]:
        """Average timings per component across all recorded calls."""
        summary = {}
        for label, times in self.timings.items():
            arr = np.array(times)
            summary[label] = {
                "total_ms": float(arr.sum()),
                "mean_ms": float(arr.mean()),
                "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "calls": len(arr),
            }
        return summary


# ---------------------------------------------------------------------------
# Attention core timing (derived)
# ---------------------------------------------------------------------------

def derive_attn_core(profiler: ComponentProfiler, model: torch.nn.Module, device: torch.device,
                     input_ids: torch.Tensor, steps: int) -> float:
    """
    Measure attention core (softmax(QK^T)V) by timing the full self_attn module
    and subtracting qkv_proj + o_proj.

    Returns total attn_core time in ms across all steps.
    """
    def make_pre(store):
        def hook(mod, inp):
            sync_device(device)
            store["start"] = time.perf_counter()
        return hook

    def make_post(store, times_list):
        def hook(mod, inp, out):
            sync_device(device)
            times_list.append((time.perf_counter() - store["start"]) * 1000.0)
        return hook

    # Find all self_attn modules
    attn_modules = []
    for name, mod in model.named_modules():
        if name.endswith("self_attn"):
            attn_modules.append(mod)

    hooks = []
    stores = []
    per_module_times: list[list[float]] = []
    for mod in attn_modules:
        store = {}
        times_list: list[float] = []
        stores.append(store)
        per_module_times.append(times_list)
        hooks.append(mod.register_forward_pre_hook(make_pre(store)))
        hooks.append(mod.register_forward_hook(make_post(store, times_list)))

    # Run decode steps
    generated = input_ids.clone()
    past_kv = None
    with torch.no_grad():
        for step in range(steps):
            if step == 0:
                out = model(generated, use_cache=True)
            else:
                out = model(generated[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

    for h in hooks:
        h.remove()

    total_self_attn_ms = sum(sum(t) for t in per_module_times)

    # attn_core = self_attn_total - qkv_proj - o_proj
    qkv_total = profiler.timings.get("qkv_proj", [])
    o_total = profiler.timings.get("o_proj", [])
    attn_core_ms = total_self_attn_ms - sum(qkv_total) - sum(o_total)
    return max(attn_core_ms, 0.0)


# ---------------------------------------------------------------------------
# Arithmetic intensity computation
# ---------------------------------------------------------------------------

def compute_arithmetic_intensity(precision: str) -> dict[str, dict]:
    """
    Compute FLOP and bytes accessed per decode step (batch=1, seq=1)
    for each component, returning arithmetic intensity (FLOP/byte).
    """
    d = MODEL_DIMS
    H = d["hidden"]
    L = d["layers"]
    V = d["vocab"]
    FFN = d["ffn_intermediate"]
    Nh = d["heads"]
    Nkv = d["kv_heads"]
    Hd = d["head_dim"]

    # Bytes per parameter
    if precision == "float16":
        bpp = 2
    elif precision == "int8":
        bpp = 1
    elif precision == "Q4":
        bpp = 0.5
    else:
        bpp = 2

    components = {}

    # Embedding: lookup, no matmul — 1 read of H floats
    emb_flops = 0
    emb_bytes = H * bpp
    components["embedding"] = {
        "flops_per_step": emb_flops,
        "bytes_per_step": emb_bytes,
        "arithmetic_intensity": 0.0,
    }

    # QKV projection: 3 matmuls [1,H] x [H, Nh*Hd], [H, Nkv*Hd], [H, Nkv*Hd]
    # Q: H -> Nh*Hd, K: H -> Nkv*Hd, V: H -> Nkv*Hd
    q_flops = 2 * H * (Nh * Hd)
    k_flops = 2 * H * (Nkv * Hd)
    v_flops = 2 * H * (Nkv * Hd)
    qkv_flops = (q_flops + k_flops + v_flops) * L
    q_bytes = H * (Nh * Hd) * bpp
    k_bytes = H * (Nkv * Hd) * bpp
    v_bytes = H * (Nkv * Hd) * bpp
    qkv_bytes = (q_bytes + k_bytes + v_bytes) * L
    components["qkv_proj"] = {
        "flops_per_step": qkv_flops,
        "bytes_per_step": qkv_bytes,
        "arithmetic_intensity": round(qkv_flops / qkv_bytes, 2) if qkv_bytes else 0,
    }

    # Attention core: QK^T + softmax + AV — depends on context length
    # For a single decode step at context C:
    #   QK^T: Nh * (2*Hd*C)  flops
    #   AV:   Nh * (2*Hd*C)  flops
    # We use C=128 as a representative context
    C = 128
    attn_flops = L * Nh * (2 * Hd * C + 2 * Hd * C)
    # Bytes: read K cache [Nkv, C, Hd], V cache [Nkv, C, Hd] per layer
    attn_bytes = L * 2 * Nkv * C * Hd * bpp
    components["attn_core"] = {
        "flops_per_step": attn_flops,
        "bytes_per_step": attn_bytes,
        "arithmetic_intensity": round(attn_flops / attn_bytes, 2) if attn_bytes else 0,
        "note": f"context_length={C}",
    }

    # O-projection: [1, Nh*Hd] x [Nh*Hd, H] per layer
    o_flops = 2 * (Nh * Hd) * H * L
    o_bytes = (Nh * Hd) * H * bpp * L
    components["o_proj"] = {
        "flops_per_step": o_flops,
        "bytes_per_step": o_bytes,
        "arithmetic_intensity": round(o_flops / o_bytes, 2) if o_bytes else 0,
    }

    # MLP: gate_proj [H, FFN] + up_proj [H, FFN] + SiLU + element-wise mul + down_proj [FFN, H]
    gate_flops = 2 * H * FFN
    up_flops = 2 * H * FFN
    down_flops = 2 * FFN * H
    mlp_flops = (gate_flops + up_flops + down_flops) * L
    gate_bytes = H * FFN * bpp
    up_bytes = H * FFN * bpp
    down_bytes = FFN * H * bpp
    mlp_bytes = (gate_bytes + up_bytes + down_bytes) * L
    components["mlp"] = {
        "flops_per_step": mlp_flops,
        "bytes_per_step": mlp_bytes,
        "arithmetic_intensity": round(mlp_flops / mlp_bytes, 2) if mlp_bytes else 0,
    }

    # LayerNorm (RMSNorm): 2*H flops per norm, 2 per layer + 1 final
    ln_flops = (2 * L + 1) * 2 * H
    ln_bytes = (2 * L + 1) * H * bpp  # read gamma
    components["layernorm"] = {
        "flops_per_step": ln_flops,
        "bytes_per_step": ln_bytes,
        "arithmetic_intensity": round(ln_flops / ln_bytes, 2) if ln_bytes else 0,
    }

    # LM Head: [1, H] x [H, V]
    lm_flops = 2 * H * V
    lm_bytes = H * V * bpp
    components["lm_head"] = {
        "flops_per_step": lm_flops,
        "bytes_per_step": lm_bytes,
        "arithmetic_intensity": round(lm_flops / lm_bytes, 2) if lm_bytes else 0,
    }

    return components


# ---------------------------------------------------------------------------
# Main decomposition runner
# ---------------------------------------------------------------------------

def run_decomposition(
    model_id: str,
    platform_name: str,
    precision: str,
    output_dir: Path,
    decode_steps: int = DECODE_STEPS,
    prompt: str = "The future of artificial intelligence is",
):
    device = get_device(platform_name)
    print(f"\n{'='*60}")
    print(f"  ArchX Phase 2 — Latency Decomposition")
    print(f"  Platform: {platform_name}  Device: {device}  Precision: {precision}")
    print(f"  Model: {model_id}")
    print(f"  Decode steps: {decode_steps}")
    print(f"{'='*60}\n")

    # --- Load model ---
    print("[1/5] Loading model...")
    dtype = torch.float16

    load_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "dtype": dtype,
        "device_map": {"": device},
    }

    if precision == "int8":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            del load_kwargs["dtype"]
            if device.type == "mps":
                # bitsandbytes doesn't support MPS, fall back to CPU
                print("  [warn] bitsandbytes not supported on MPS, using CPU for int8")
                device = torch.device("cpu")
                load_kwargs["device_map"] = {"": device}
        except ImportError:
            print("  [warn] bitsandbytes not installed, falling back to float16")
            precision = "float16"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"  Prompt tokens: {input_ids.shape[1]}")

    # --- Warmup ---
    print("\n[2/5] Warming up (3 forward passes)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    sync_device(device)

    # --- Profile with hooks ---
    print("\n[3/5] Profiling component latencies...")
    profiler = ComponentProfiler(model, device)

    # Run autoregressive decode
    generated = input_ids.clone()
    past_kv = None
    total_start = time.perf_counter()

    with torch.no_grad():
        for step in range(decode_steps):
            sync_device(device)
            if step == 0:
                out = model(generated, use_cache=True)
            else:
                out = model(generated[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

    sync_device(device)
    total_wall_ms = (time.perf_counter() - total_start) * 1000.0

    summary = profiler.get_summary()
    profiler.remove_hooks()

    # --- Derive attention core ---
    print("\n[4/5] Measuring attention core (self_attn - projections)...")
    # Re-run to measure full self_attn blocks
    profiler2 = ComponentProfiler(model, device)
    attn_core_ms = derive_attn_core(profiler2, model, device, input_ids, decode_steps)
    profiler2.remove_hooks()

    summary["attn_core"] = {
        "total_ms": round(attn_core_ms, 4),
        "mean_ms": round(attn_core_ms / max(decode_steps, 1), 4),
        "std_ms": 0.0,
        "calls": decode_steps,
    }

    # --- Compute framework overhead ---
    instrumented_total = sum(v["total_ms"] for v in summary.values())
    overhead_ms = max(total_wall_ms - instrumented_total, 0.0)
    summary["framework_overhead"] = {
        "total_ms": round(overhead_ms, 4),
        "mean_ms": round(overhead_ms / decode_steps, 4),
        "std_ms": 0.0,
        "calls": decode_steps,
    }
    summary["total"] = {
        "total_ms": round(total_wall_ms, 4),
        "mean_ms": round(total_wall_ms / decode_steps, 4),
        "std_ms": 0.0,
        "calls": decode_steps,
    }

    # --- Arithmetic intensity ---
    arith = compute_arithmetic_intensity(precision)

    # --- Build output ---
    component_order = [
        "embedding", "layernorm", "qkv_proj", "attn_core",
        "o_proj", "mlp", "lm_head", "framework_overhead", "total",
    ]

    results = {
        "platform": platform_name,
        "precision": precision,
        "model": model_id,
        "decode_steps": decode_steps,
        "device": str(device),
        "model_dimensions": MODEL_DIMS,
        "components": {},
    }

    for comp in component_order:
        entry = summary.get(comp, {"total_ms": 0, "mean_ms": 0, "std_ms": 0, "calls": 0})
        pct = (entry["total_ms"] / total_wall_ms * 100) if total_wall_ms > 0 else 0
        ai = arith.get(comp, {})
        results["components"][comp] = {
            "total_ms": round(entry["total_ms"], 4),
            "mean_per_step_ms": round(entry["mean_ms"], 4),
            "std_ms": round(entry["std_ms"], 4),
            "pct_of_total": round(pct, 2),
            "arithmetic_intensity_flop_per_byte": ai.get("arithmetic_intensity", None),
            "flops_per_step": ai.get("flops_per_step", None),
            "bytes_per_step": ai.get("bytes_per_step", None),
        }

    # --- Print console table ---
    print(f"\n[5/5] Results\n")
    print(f"{'Component':<22} {'Total (ms)':>10} {'Mean/step':>10} {'% Total':>8}  {'AI (F/B)':>8}")
    print("-" * 65)
    for comp in component_order:
        c = results["components"][comp]
        ai_str = f"{c['arithmetic_intensity_flop_per_byte']:.1f}" if c["arithmetic_intensity_flop_per_byte"] is not None else "-"
        if comp == "total":
            print("-" * 65)
        print(f"{comp:<22} {c['total_ms']:>10.2f} {c['mean_per_step_ms']:>10.2f} {c['pct_of_total']:>7.1f}%  {ai_str:>8}")

    print(f"\nWall time: {total_wall_ms:.2f} ms for {decode_steps} steps")
    print(f"Mean per-step: {total_wall_ms / decode_steps:.2f} ms")

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{platform_name}_{precision}_decomp.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Q4 simulation from Phase 1 data
# ---------------------------------------------------------------------------

def simulate_q4_from_phase1(
    raw_dir: Path,
    platform_name: str,
    output_dir: Path,
):
    """
    Use Phase 1 PTL data to estimate Q4 decomposition by scaling
    float16 decomposition proportionally to the Q4/F16 PTL ratio.
    """
    # Load float16 decomposition
    f16_path = output_dir / f"{platform_name}_float16_decomp.json"
    if not f16_path.exists():
        print(f"[Q4 sim] No float16 decomposition found at {f16_path}, skipping Q4 simulation")
        return None

    with open(f16_path) as f:
        f16_data = json.load(f)

    # Load Phase 1 PTL data for both F16 and Q4_K_M
    f16_ptl_path = raw_dir / f"{platform_name}_F16_per_token_latency_128.json"
    q4_ptl_path = raw_dir / f"{platform_name}_Q4_K_M_per_token_latency_128.json"

    if not f16_ptl_path.exists() or not q4_ptl_path.exists():
        print(f"[Q4 sim] Missing Phase 1 PTL data, skipping")
        return None

    with open(f16_ptl_path) as f:
        f16_ptl = json.load(f)["trial_result"]["median"]
    with open(q4_ptl_path) as f:
        q4_ptl = json.load(f)["trial_result"]["median"]

    ratio = q4_ptl / f16_ptl if f16_ptl > 0 else 1.0
    print(f"\n[Q4 sim] PTL ratio Q4/F16 = {q4_ptl:.2f}/{f16_ptl:.2f} = {ratio:.3f}")

    # Scale each component
    q4_results = {
        "platform": platform_name,
        "precision": "Q4_simulated",
        "model": f16_data["model"],
        "decode_steps": f16_data["decode_steps"],
        "device": f16_data["device"],
        "model_dimensions": f16_data["model_dimensions"],
        "note": f"Simulated from float16 decomposition scaled by Q4/F16 PTL ratio ({ratio:.3f})",
        "components": {},
    }

    # Compute-bound components scale less with quantization,
    # memory-bound components scale more
    arith = compute_arithmetic_intensity("Q4")
    f16_arith = compute_arithmetic_intensity("float16")

    total_scaled = 0.0
    for comp, f16_comp in f16_data["components"].items():
        if comp in ("framework_overhead", "total"):
            continue
        # Memory-bound components (low AI) benefit more from quantization
        f16_ai = f16_arith.get(comp, {}).get("arithmetic_intensity", 1.0) or 1.0
        q4_ai = arith.get(comp, {}).get("arithmetic_intensity", 1.0) or 1.0

        # Scale: memory-bound components scale with bytes ratio,
        # compute-bound stay similar.
        # AI < 2.0 FLOP/byte identifies memory-bound components (embedding, layernorm,
        # attn_core at typical context lengths) that benefit proportionally from
        # halving parameter size; compute-bound ops (MLP, QKV) gain less.
        if f16_ai < 2.0:  # memory-bound
            comp_ratio = ratio
        else:  # compute-bound
            comp_ratio = max(ratio, 0.8)  # at most 20% faster

        scaled_total = f16_comp["total_ms"] * comp_ratio
        total_scaled += scaled_total
        q4_results["components"][comp] = {
            "total_ms": round(scaled_total, 4),
            "mean_per_step_ms": round(scaled_total / f16_data["decode_steps"], 4),
            "pct_of_total": 0,  # recomputed below
            "arithmetic_intensity_flop_per_byte": arith.get(comp, {}).get("arithmetic_intensity"),
        }

    # Add overhead and total
    oh = f16_data["components"].get("framework_overhead", {}).get("total_ms", 0)
    total_scaled += oh
    q4_results["components"]["framework_overhead"] = {
        "total_ms": round(oh, 4),
        "mean_per_step_ms": round(oh / f16_data["decode_steps"], 4),
        "pct_of_total": 0,
    }
    q4_results["components"]["total"] = {
        "total_ms": round(total_scaled, 4),
        "mean_per_step_ms": round(total_scaled / f16_data["decode_steps"], 4),
        "pct_of_total": 100.0,
    }

    # Recompute percentages
    for comp in q4_results["components"]:
        if comp != "total":
            pct = q4_results["components"][comp]["total_ms"] / total_scaled * 100 if total_scaled > 0 else 0
            q4_results["components"][comp]["pct_of_total"] = round(pct, 2)

    out_path = output_dir / f"{platform_name}_Q4_simulated_decomp.json"
    with open(out_path, "w") as f:
        json.dump(q4_results, f, indent=2)
    print(f"[Q4 sim] Saved to {out_path}")

    return q4_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArchX Phase 2 — Per-component latency decomposition",
    )
    parser.add_argument(
        "--model", "-m",
        default=HF_MODEL_ID,
        help=f"HuggingFace model ID (default: {HF_MODEL_ID})",
    )
    parser.add_argument(
        "--precision", "-p",
        choices=["float16", "int8"],
        default="float16",
        help="Precision for HF model (default: float16)",
    )
    parser.add_argument(
        "--platform",
        choices=["M1", "M2", "M4"],
        default=None,
        help="Platform (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/decomposition",
        help="Output directory (default: results/decomposition)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DECODE_STEPS,
        help=f"Number of decode steps (default: {DECODE_STEPS})",
    )
    parser.add_argument(
        "--prompt",
        default="The future of artificial intelligence is",
        help="Prompt text for decoding",
    )
    parser.add_argument(
        "--simulate-q4",
        action="store_true",
        help="Also generate Q4 simulation from Phase 1 data + float16 decomposition",
    )
    parser.add_argument(
        "--raw-dir",
        default="results/raw",
        help="Phase 1 raw results dir (for Q4 simulation)",
    )

    args = parser.parse_args()
    plat = args.platform or detect_platform()

    run_decomposition(
        model_id=args.model,
        platform_name=plat,
        precision=args.precision,
        output_dir=Path(args.output_dir),
        decode_steps=args.steps,
        prompt=args.prompt,
    )

    if args.simulate_q4:
        simulate_q4_from_phase1(
            raw_dir=Path(args.raw_dir),
            platform_name=plat,
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()
