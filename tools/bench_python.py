"""Latency baseline for the Python reference runtime.

Times OptionMarkerBackend.evaluate end to end (packing, tokenization, forward,
calibration) for a few golden cases, after warmup, on each requested device.

Usage (from the repo root):
    HF_HOME=.hf-cache uv run python von-rs/tools/bench_python.py --devices mps cpu
(HF_HOME is wherever the Hub cache lives; `just models` prints it.)
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.environ.get("VON_PY_SRC") or os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from von.backends.option_marker_backend import OptionMarkerBackend  # noqa: E402
from export_golden import HANDCRAFTED  # noqa: E402

# Same requests and iteration counts as examples/latency.rs.
BENCH = {"route-account-access": 100, "many-options": 100, "noul-zero-shot": 100, "score-detailed": 100,
         "long-600-tokens": 30}


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    cases = [c for c in HANDCRAFTED if c[0] in BENCH]
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    print(f"# {chip} · macOS {platform.mac_ver()[0]} · torch {torch.__version__} · threads {torch.get_num_threads()}")
    results = {}
    for device in args.devices:
        backend = OptionMarkerBackend(device=device)
        backend._get_model()
        for case_id, state, questions in cases:
            for _ in range(args.warmup):
                backend.evaluate(state, questions)
            sync(device)
            samples = []
            for _ in range(BENCH[case_id]):
                t0 = time.perf_counter()
                backend.evaluate(state, questions)
                sync(device)
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            p50 = statistics.median(samples)
            p95 = samples[int(0.95 * (len(samples) - 1))]
            results[f"{device}/{case_id}"] = {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2)}
            print(f"| {device} | {case_id} | {p50:.2f} | {p95:.2f} |")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
