"""Python server vs Rust `von serve` on identical payloads.

For each server: start it (optionally after purging the OS page cache, so the
weights come off disk), then measure
  - cold start: launch -> /health answers, and launch -> first /v1/systemone
    answer (Python loads the model lazily on the first request, Rust at startup);
  - sequential latency, p50/p95, for the same golden requests as
    examples/latency.rs and tools/bench_python.py;
  - throughput: CONCURRENCY clients sending THROUGHPUT_REQUESTS short requests;
  - resident memory (RSS) after the run.
Both servers load from local files only: Rust from VON_WEIGHTS, Python (with
HF_HUB_OFFLINE=1) from the pinned Von 1.2 Hub snapshot (tools/hub_pins.py).
The markdown result replaces this device's section of bench/RESULTS.md.

Usage (run by bench_compare.sh):
    python bench_compare.py --device cpu|metal --von target/release/von [--purge]
"""

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time

import httpx
import psutil

from hub_pins import snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.dirname(HERE)
# Converted checkpoint; `just` and tools/model_paths.sh set VON_WEIGHTS (VON_MODELS_DIR).
WEIGHTS = os.path.abspath(os.environ.get("VON_WEIGHTS") or os.path.join(RS, "checkpoints", "von-1.2"))
ROOT = os.path.dirname(RS)

# Same requests and iteration counts as examples/latency.rs (the long one fewer times).
CASES = {"route-account-access": 50, "many-options": 50, "noul-zero-shot": 50, "score-detailed": 50,
         "long-600-tokens": 20}
WARMUP = 5
CONCURRENCY = 8
THROUGHPUT_REQUESTS = 200
THROUGHPUT_CASE = "route-account-access"


def golden_payloads():
    golden = json.load(open(os.path.join(RS, "tests", "fixtures", "golden", "v1_2.json")))
    by_id = {c["id"]: c for c in golden["cases"]}
    return {cid: {"model": "von-latest", "state": by_id[cid]["state"], "questions": by_id[cid]["questions"]}
            for cid in CASES}


def purge():
    print("  purging the page cache (sudo purge)...", flush=True)
    subprocess.run(["sudo", "purge"], check=True)


def rss_mb(pid):
    try:
        return psutil.Process(pid).memory_info().rss / 2**20
    except psutil.Error:
        return float("nan")


def pct(samples, p):
    s = sorted(samples)
    return s[min(len(s) - 1, (len(s) - 1) * p // 100)]


def run_server(name, cmd, env, port, payloads, do_purge):
    base = f"http://127.0.0.1:{port}"
    if do_purge:
        purge()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=base, timeout=600) as client:
            while True:
                if proc.poll() is not None:
                    sys.exit(f"{name} server exited during startup (code {proc.returncode})")
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.01)
            ready = time.perf_counter() - t0
            r = client.post("/v1/systemone", json=payloads[THROUGHPUT_CASE])
            first = time.perf_counter() - t0
            r.raise_for_status()
            print(f"  {name}: ready {ready:.2f}s, first answer {first:.2f}s", flush=True)

            latency = {}
            for cid, iters in CASES.items():
                for _ in range(WARMUP):
                    client.post("/v1/systemone", json=payloads[cid]).raise_for_status()
                ms = []
                for _ in range(iters):
                    t = time.perf_counter()
                    client.post("/v1/systemone", json=payloads[cid]).raise_for_status()
                    ms.append((time.perf_counter() - t) * 1e3)
                latency[cid] = (pct(ms, 50), pct(ms, 95))
                print(f"  {name}: {cid} p50 {latency[cid][0]:.2f} ms", flush=True)

        errors, lock, next_i = [], threading.Lock(), [0]

        def worker():
            with httpx.Client(base_url=base, timeout=600) as c:
                while True:
                    with lock:
                        if next_i[0] >= THROUGHPUT_REQUESTS:
                            return
                        next_i[0] += 1
                    try:
                        c.post("/v1/systemone", json=payloads[THROUGHPUT_CASE]).raise_for_status()
                    except httpx.HTTPError as e:
                        errors.append(repr(e))

        t = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        rps = THROUGHPUT_REQUESTS / (time.perf_counter() - t)
        print(f"  {name}: {rps:.1f} req/s at concurrency {CONCURRENCY}, {len(errors)} errors", flush=True)
        return {"ready": ready, "first": first, "latency": latency, "rps": rps, "errors": len(errors),
                "rss": rss_mb(proc.pid)}
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def render(device, cold, env_line, py, rs):
    py_dev, rs_dev = ("MPS", "Metal") if device == "metal" else ("CPU", "CPU")
    start = "true cold start (page cache purged before each launch)" if cold else "warm page cache"
    lines = [
        f"<!-- bench_compare:{device} -->",
        f"## {rs_dev}: Python ({py_dev}) vs Rust ({rs_dev})",
        "",
        f"{env_line} · {datetime.date.today().isoformat()} · {start}.",
        "",
        "| Metric | Python | Rust | Rust / Python |",
        "|---|---|---|---|",
        f"| Launch → /health ready (s) | {py['ready']:.2f} | {rs['ready']:.2f} | {rs['ready'] / py['ready']:.2f}× |",
        f"| Launch → first answer (s) | {py['first']:.2f} | {rs['first']:.2f} | {rs['first'] / py['first']:.2f}× |",
    ]
    for cid in CASES:
        (pp50, pp95), (rp50, rp95) = py["latency"][cid], rs["latency"][cid]
        lines.append(f"| {cid}, p50 / p95 (ms) | {pp50:.2f} / {pp95:.2f} | {rp50:.2f} / {rp95:.2f} | {rp50 / pp50:.2f}× |")
    lines += [
        f"| Throughput, {CONCURRENCY} clients (req/s) | {py['rps']:.1f} | {rs['rps']:.1f} | {rs['rps'] / py['rps']:.2f}× |",
        f"| Errors | {py['errors']} | {rs['errors']} | |",
        f"| RSS after the run (MB) | {py['rss']:.0f} | {rs['rss']:.0f} | {rs['rss'] / py['rss']:.2f}× |",
        f"<!-- /bench_compare:{device} -->",
    ]
    return "\n".join(lines)


def write_section(path, device, section):
    text = open(path).read() if os.path.exists(path) else "# Python vs Rust server benchmarks\n"
    pattern = re.compile(rf"<!-- bench_compare:{device} -->.*?<!-- /bench_compare:{device} -->", re.S)
    text = pattern.sub(lambda _: section, text) if pattern.search(text) else text.rstrip("\n") + "\n\n" + section + "\n"
    open(path, "w").write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "metal"], required=True)
    ap.add_argument("--von", required=True, help="path to the release `von` binary")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--purge", action="store_true", help="sudo purge before each launch (true cold start)")
    ap.add_argument("--out", default=os.path.join(RS, "bench", "RESULTS.md"))
    args = ap.parse_args()

    payloads = golden_payloads()
    env = {k: v for k, v in os.environ.items() if k not in ("VON_API_KEY", "TYPESAFE_API_KEY", "VON_CORS_ORIGINS")}
    py_src = env.get("VON_PY_SRC") or os.path.join(ROOT, "bug-fix-fork-von", "src")
    # HF_HUB_OFFLINE: load from the local cache without the Hub's network checks, as Rust
    # loads from a local checkpoint; otherwise network latency lands in the cold start.
    py_env = {**env, "PYTHONPATH": py_src, "HF_HUB_OFFLINE": "1",
              "VON_DEVICE": "mps" if args.device == "metal" else "cpu"}
    # Python has no checkpoint setting; point its default lookup at the pinned snapshot.
    boot = ("import sys; from von.backends.option_marker_backend import OptionMarkerBackend as B; "
            "B.DEFAULT_CHECKPOINT_DIRS = (sys.argv.pop(1),); from von.cli import main; main()")
    py_cmd = [sys.executable, "-c", boot, snapshot("1.2"), "serve",
              "--host", "127.0.0.1", "--port", str(args.port)]
    rs_env = {**env, "VON_DEVICE": args.device, "VON_CHECKPOINT_DIR": WEIGHTS}
    rs_cmd = [args.von, "serve", "--host", "127.0.0.1", "--port", str(args.port)]

    import torch  # only for the version line; the server runs in its own process

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.split()[1]
    env_line = f"{chip} · macOS {platform.mac_ver()[0]} · torch {torch.__version__} · rustc {rustc}"
    print(f"# {env_line} · device {args.device}", flush=True)

    py = run_server("python", py_cmd, py_env, args.port, payloads, args.purge)
    rs = run_server("rust", rs_cmd, rs_env, args.port, payloads, args.purge)
    section = render(args.device, args.purge, env_line, py, rs)
    write_section(args.out, args.device, section)
    print(section)


if __name__ == "__main__":
    main()
