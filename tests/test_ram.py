#!/usr/bin/env python3
"""test_ram.py

Measure RSS (RAM) usage for loading ProtT5-XL (T5EncoderModel) under different
precisions.

Why subprocesses?
-----------------
Python/PyTorch memory allocators often *do not return* freed memory back to the
OS, so RSS stays high even after `del model` + `gc.collect()`. That makes
sequential "fp32 then fp16" measurements in one process misleading.

This script solves it by running each precision in a fresh subprocess.

Usage
-----
Driver mode (default): runs fp32, fp16, fp8 (when supported) in separate
processes and prints a comparison table.

    python tests/test_ram.py

Single-run mode (worker):

    python tests/test_ram.py --precision fp16

Optional:
    --model-path PATH   (local directory or HF id)
    --device cpu|cuda   (default: cpu)

Notes
-----
* fp8 is only attempted when a float8 dtype exists AND device=cuda.
* On CPU, fp16 loading may still allocate substantial temporary buffers; the
  subprocess approach ensures each precision starts from a clean baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, Optional

import psutil
import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers import logging as transformers_logging


# Silence noisy HF logging (e.g., unused decoder weights when loading encoder-only)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


DEFAULT_LOCAL_MODEL = "/home/saleh/webKinPred/models/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50"
DEFAULT_DOCKER_MODEL = "/app/models/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50"
DEFAULT_HF_ID = "Rostlab/prot_t5_xl_uniref50"


def _detect_default_model_path() -> str:
    if os.environ.get("KINFORM_MEDIA_PATH") and os.path.isdir(DEFAULT_DOCKER_MODEL):
        return DEFAULT_DOCKER_MODEL
    if os.path.isdir(DEFAULT_LOCAL_MODEL):
        return DEFAULT_LOCAL_MODEL
    return DEFAULT_HF_ID


def _rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 / 1024


def _model_param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers() if torch.is_tensor(b)
    )


def _dtype_for_precision(precision: str) -> Optional[torch.dtype]:
    precision = precision.lower()
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp8":
        # Different torch builds expose different float8 names
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        ):
            if hasattr(torch, name):
                return getattr(torch, name)
        return None
    raise ValueError(f"Unknown precision: {precision}")


def _fmt_mb(x: float) -> str:
    return f"{x:.2f} MB"


def run_single(precision: str, model_path: str, device: str) -> Dict[str, Any]:
    """Run one isolated measurement in the current process.

    Returns a JSON-serializable dict.
    """

    precision = precision.lower()
    device = device.lower()

    # Validate device
    if device == "cuda":
        if not torch.cuda.is_available():
            return {
                "ok": False,
                "precision": precision,
                "device": device,
                "reason": "cuda requested but torch.cuda.is_available() is False",
            }
    elif device != "cpu":
        return {
            "ok": False,
            "precision": precision,
            "device": device,
            "reason": "invalid device",
        }

    dtype = _dtype_for_precision(precision)

    # fp8 only realistically supported on cuda; do not pretend on cpu
    if precision == "fp8":
        if dtype is None:
            return {
                "ok": False,
                "precision": precision,
                "device": device,
                "reason": "fp8 dtype not available in this PyTorch build",
            }
        if device != "cuda":
            return {
                "ok": False,
                "precision": precision,
                "device": device,
                "reason": "fp8 test requires --device cuda",
            }

    # Clean start within the process
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    rss0 = _rss_mb()

    # Load tokenizer first (small but included consistently)
    tok = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    rss_tok = _rss_mb()

    # Load model
    model = T5EncoderModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # For fp16/bf16, ensure weights are actually that dtype (avoid mixed dtypes)
    if precision in ("fp16", "bf16"):
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype=dtype)
        for b in model.buffers():
            if torch.is_tensor(b) and b.is_floating_point() and b.dtype != dtype:
                b.data = b.data.to(dtype=dtype)

    rss_after_load = _rss_mb()

    # Move to device if requested
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    rss_after_move = _rss_mb()

    # Snapshot dtypes found
    dtypes_found = set()
    for _, p in model.named_parameters():
        dtypes_found.add(str(p.dtype))
    for _, b in model.named_buffers():
        if torch.is_tensor(b):
            dtypes_found.add(str(b.dtype))

    param_mb = _model_param_bytes(model) / 1024 / 1024

    gpu_alloc = None
    gpu_reserved = None
    if device == "cuda":
        gpu_alloc = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024

    # ── inference pass measurement ────────────────────────────────────────────
    # Use a short but realistic protein sequence (~50 AA) to measure how much
    # memory the forward pass itself allocates.
    SAMPLE_SEQ = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 residues
    spaced = " ".join(list(SAMPLE_SEQ))
    token_data = tok(spaced, return_tensors="pt", add_special_tokens=True).to(torch_device)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    rss_before_infer = _rss_mb()

    with torch.no_grad():
        out = model(**token_data)
    _ = out.last_hidden_state.cpu()  # force materialisation on CPU

    rss_after_infer = _rss_mb()

    del out, token_data
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    # ─────────────────────────────────────────────────────────────────────────

    # Cleanup (mostly for hygiene; process exits in worker mode anyway)
    del model
    del tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    t1 = time.time()

    return {
        "ok": True,
        "precision": precision,
        "device": device,
        "model_path": model_path,
        "rss_baseline_mb": rss0,
        "rss_after_tokenizer_mb": rss_tok,
        "rss_after_load_mb": rss_after_load,
        "rss_after_move_mb": rss_after_move,
        "delta_load_mb": rss_after_load - rss_tok,
        "delta_total_mb": rss_after_move - rss0,
        "theoretical_param_mb": param_mb,
        "ratio_rss_to_params": (rss_after_move - rss0) / param_mb if param_mb > 0 else None,
        "dtypes_in_model": sorted(dtypes_found),
        "rss_before_infer_mb": rss_before_infer,
        "rss_after_infer_mb": rss_after_infer,
        "delta_infer_mb": rss_after_infer - rss_before_infer,
        "gpu_alloc_mb": gpu_alloc,
        "gpu_reserved_mb": gpu_reserved,
        "seconds": t1 - t0,
    }


def _run_worker_subprocess(precision: str, model_path: str, device: str) -> Dict[str, Any]:
    """Run one precision in a fresh python process and parse its JSON output."""

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--precision",
        precision,
        "--model-path",
        model_path,
        "--device",
        device,
        "--json",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {
            "ok": False,
            "precision": precision,
            "device": device,
            "reason": f"worker exited with code {proc.returncode}",
            "stderr": proc.stderr.strip()[-4000:],
            "stdout": proc.stdout.strip()[-4000:],
        }

    try:
        return json.loads(proc.stdout)
    except Exception as e:
        return {
            "ok": False,
            "precision": precision,
            "device": device,
            "reason": f"failed to parse worker JSON: {e}",
            "stdout": proc.stdout.strip()[-4000:],
            "stderr": proc.stderr.strip()[-4000:],
        }


def _print_table(results: list[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("ProtT5-XL RAM comparison (subprocess-isolated)")
    print("=" * 70)

    headers = [
        ("precision", 10),
        ("device", 8),
        ("rss_total", 14),
        ("rss_load", 14),
        ("infer_60aa", 12),
        ("params", 12),
        ("ratio", 8),
        ("seconds", 10),
    ]

    def row_line(vals: list[str]) -> str:
        parts = []
        for (h, w), v in zip(headers, vals):
            parts.append(f"{v:<{w}}")
        return " ".join(parts)

    print(row_line([h for h, _ in headers]))
    print("-" * 70)

    for r in results:
        if not r.get("ok"):
            print(
                row_line(
                    [
                        r.get("precision", "?"),
                        r.get("device", "?"),
                        "ERROR",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    ]
                )
            )
            continue

        print(
            row_line(
                [
                    r["precision"],
                    r["device"],
                    _fmt_mb(r["delta_total_mb"]),
                    _fmt_mb(r["delta_load_mb"]),
                    _fmt_mb(r["delta_infer_mb"]) if r.get("delta_infer_mb") is not None else "N/A",
                    _fmt_mb(r["theoretical_param_mb"]),
                    f"{r['ratio_rss_to_params']:.2f}x" if r.get("ratio_rss_to_params") is not None else "N/A",
                    f"{r['seconds']:.1f}s",
                ]
            )
        )

    errors = [r for r in results if not r.get("ok")]
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e.get('precision')} ({e.get('device')}): {e.get('reason')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure ProtT5-XL RAM usage by precision")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "fp8"],
        default=None,
        help="Run a single precision in the current process (worker mode).",
    )
    parser.add_argument(
        "--model-path",
        default=_detect_default_model_path(),
        help="Local model directory or HuggingFace model id.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to move the model to. Default: cpu.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="In --precision mode: print JSON only (for the driver).",
    )

    args = parser.parse_args()

    # Worker mode
    if args.precision is not None:
        result = run_single(args.precision, args.model_path, args.device)
        if args.json:
            print(json.dumps(result))
        else:
            if not result.get("ok"):
                print(json.dumps(result, indent=2))
                return 2
            print("=" * 70)
            print(f"Precision: {result['precision']}  Device: {result['device']}")
            print(f"Model: {result['model_path']}")
            print("=" * 70)
            print(f"Baseline RSS:          {_fmt_mb(result['rss_baseline_mb'])}")
            print(f"After tokenizer RSS:   {_fmt_mb(result['rss_after_tokenizer_mb'])}")
            print(f"After load RSS:        {_fmt_mb(result['rss_after_load_mb'])}")
            print(f"After move RSS:        {_fmt_mb(result['rss_after_move_mb'])}")
            print(f"Load delta:            {_fmt_mb(result['delta_load_mb'])}")
            print(f"Total delta:           {_fmt_mb(result['delta_total_mb'])}")
            print(f"Theoretical params:    {_fmt_mb(result['theoretical_param_mb'])}")
            print(f"Ratio (rss/params):    {result['ratio_rss_to_params']:.2f}x")
            print(f"Infer RSS before:      {_fmt_mb(result['rss_before_infer_mb'])}")
            print(f"Infer RSS after:       {_fmt_mb(result['rss_after_infer_mb'])}")
            print(f"Infer delta (60AA):    {_fmt_mb(result['delta_infer_mb'])}")
            print(f"Dtypes in model:       {', '.join(result['dtypes_in_model'])}")
            if result.get("gpu_alloc_mb") is not None:
                print(f"GPU allocated:         {_fmt_mb(result['gpu_alloc_mb'])}")
                print(f"GPU reserved:          {_fmt_mb(result['gpu_reserved_mb'])}")
            print(f"Seconds:               {result['seconds']:.1f}s")
        return 0 if result.get("ok") else 2

    # Driver mode
    precisions = ["fp32", "fp16", "fp8"]
    results: list[Dict[str, Any]] = []

    for p in precisions:
        results.append(_run_worker_subprocess(p, args.model_path, args.device))

    _print_table(results)

    # Exit non-zero if fp32/fp16 fail.
    hard_fail = any((not r.get("ok")) and r.get("precision") in ("fp32", "fp16") for r in results)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
