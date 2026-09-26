"""Soak test for `von serve`: many requests at high concurrency, watching memory.

Starts the release binary, sends REQUESTS requests from CONCURRENCY client
threads, and samples the server's RSS once a second. The mix is mostly valid
golden requests (the long ones excluded), plus error paths that must not leak
either: invalid JSON (422), a malformed question (422), an input over the
token limit (422) and a wrong Bearer token (401).

Passes when every response has the expected status, RSS at the end is within
GROWTH_LIMIT of RSS after the warm-up requests, and RSS is not still climbing:
the least-squares slope over the second half of the run must stay under
SLOPE_LIMIT MB per 1,000 requests. (A steady leak can stay under GROWTH_LIMIT
for a short run; the slope catches it.)

Usage (from the repo root):
    uv run python von-rs/tools/soak.py --device cpu|metal [--requests 10000] [--concurrency 64]
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time

import httpx
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.dirname(HERE)
# Converted checkpoint; `just` and tools/model_paths.sh set VON_WEIGHTS (VON_MODELS_DIR).
WEIGHTS = os.path.abspath(os.environ.get("VON_WEIGHTS") or os.path.join(RS, "checkpoints", "von-1.1"))
KEY = f"soak-{os.getpid()}"
GROWTH_LIMIT = 0.10  # 10% RSS growth after warm-up
SLOPE_LIMIT = 1.0  # MB per 1,000 requests over the second half of the run
ERROR_EVERY = 10  # one error-path request in every ten


def rss_mb(pid):
    try:
        return psutil.Process(pid).memory_info().rss / 2**20
    except psutil.Error:
        return None


def slope_mb_per_1k(samples):
    """Least-squares slope of RSS against requests done, in MB per 1,000 requests."""
    pts = [(done, rss) for _, done, rss in samples if rss is not None]
    if len(pts) < 3:
        return 0.0
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return 1000 * sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0.0


def request_mix():
    golden = json.load(open(os.path.join(RS, "tests", "fixtures", "golden", "v1.json")))
    ok = [(200, {"model": "von-latest", "state": c["state"], "questions": c["questions"]}, None)
          for c in golden["cases"] if not c["id"].startswith("long-")]
    q = {"q": {"type": "noul", "instructions": "Is this long?"}}
    bad = [
        (422, b'{"state": "x", "questions": ', None),  # invalid JSON
        (422, {"state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": 3}}}, None),
        (422, {"state": "word " * 9000, "questions": q}, None),  # over the 8192-token limit
        (401, {"state": "x", "questions": q}, "wrong"),
    ]
    mix = []
    for i, good in enumerate(itertools.islice(itertools.cycle(ok), len(ok) * ERROR_EVERY)):
        mix.append(good)
        if i % ERROR_EVERY == 0:
            mix.append(bad[(i // ERROR_EVERY) % len(bad)])
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "metal"], default="cpu")
    ap.add_argument("--requests", type=int, default=10_000)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1_000, help="requests before the RSS baseline")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--von", default=os.path.join(RS, "target", "release", "von"))
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "VON_API_KEY": KEY, "VON_DEVICE": args.device,
           "VON_CHECKPOINT_DIR": WEIGHTS}
    env.pop("VON_CORS_ORIGINS", None)
    proc = subprocess.Popen([args.von, "serve", "--host", "127.0.0.1", "--port", str(args.port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(1200):
            if proc.poll() is not None:
                sys.exit(f"server exited during startup (code {proc.returncode})")
            try:
                if httpx.get(f"{base}/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        print(f"server up, RSS {rss_mb(proc.pid):.0f} MB; {args.requests} requests "
              f"at concurrency {args.concurrency} on {args.device}", flush=True)

        mix = request_mix()
        lock = threading.Lock()
        state = {"next": 0, "done": 0, "baseline": None}
        failures, samples = [], []

        def worker():
            with httpx.Client(base_url=base, timeout=600) as c:
                while True:
                    with lock:
                        i = state["next"]
                        if i >= args.requests:
                            return
                        state["next"] += 1
                    want, body, key = mix[i % len(mix)]
                    headers = {"Authorization": f"Bearer {key or KEY}"}
                    try:
                        if isinstance(body, bytes):
                            headers["Content-Type"] = "application/json"
                            r = c.post("/v1/systemone", content=body, headers=headers)
                        else:
                            r = c.post("/v1/systemone", json=body, headers=headers)
                        if r.status_code != want:
                            failures.append(f"request {i}: status {r.status_code}, want {want}: {r.text[:200]}")
                    except httpx.HTTPError as e:
                        failures.append(f"request {i}: {e!r}")
                    with lock:
                        state["done"] += 1
                        if state["done"] == args.warmup:
                            state["baseline"] = rss_mb(proc.pid)

        stop = threading.Event()

        def sampler():
            t0 = time.perf_counter()
            while not stop.wait(1.0):
                rss = rss_mb(proc.pid)
                samples.append((time.perf_counter() - t0, state["done"], rss))
                if len(samples) % 30 == 0 and rss is not None:
                    print(f"  {samples[-1][0]:6.0f}s  {state['done']:6d} done  RSS {rss:.0f} MB", flush=True)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        mon = threading.Thread(target=sampler)
        mon.start()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        stop.set()
        mon.join()
        elapsed = time.perf_counter() - t0

        end = rss_mb(proc.pid)
        base_rss = state["baseline"] or end
        peak = max((s[2] for s in samples if s[2]), default=end)
        growth = (end - base_rss) / base_rss
        late = [smp for smp in samples if smp[1] >= args.warmup + (args.requests - args.warmup) // 2]
        slope = slope_mb_per_1k(late)
        print(f"{state['done']} requests in {elapsed:.0f}s ({state['done'] / elapsed:.1f} req/s), "
              f"{len(failures)} failures")
        print(f"RSS: {base_rss:.0f} MB after {args.warmup} requests, {end:.0f} MB at the end "
              f"({growth:+.1%}), peak {peak:.0f} MB; second-half slope {slope:+.2f} MB per 1k requests")
        if proc.poll() is not None:
            failures.append(f"server exited (code {proc.returncode})")
        for f in failures[:20]:
            print("  " + f)
        if growth > GROWTH_LIMIT:
            print(f"  RSS grew more than {GROWTH_LIMIT:.0%} after warm-up")
        if slope > SLOPE_LIMIT:
            print(f"  RSS still climbing: slope above {SLOPE_LIMIT} MB per 1k requests")
        if failures or growth > GROWTH_LIMIT or slope > SLOPE_LIMIT:
            sys.exit("SOAK FAILED")
        print("soak passed")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
