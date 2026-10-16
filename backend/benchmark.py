#!/usr/bin/env python3
"""
LAYADINO Inference Latency Benchmarker
Measures client round-trip latency, server inference time, jitter, and deadline miss rates.
"""

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def check_health(base_url: str) -> bool:
    """Verifies that the backend server is reachable and online."""
    try:
        req = urllib.request.Request(f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "ok"
    except Exception:
        return False


def send_predict(base_url: str, model: str, dist: float, grounded: bool = True, budget_ms: float = 0.0) -> Dict[str, Any]:
    """Sends a single prediction request to POST /predict and measures client round-trip time."""
    payload = {
        "id": int(time.time() * 1000),
        "model": model,
        "state": {
            "distance_to_cactus": dist,
            "dino_is_grounded": grounded,
            "cactus_width": 22,
            "cactus_height": 36,
            "speed": 6.0,
            "latency_budget_ms": budget_ms,
        },
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/predict",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        t1 = time.perf_counter()
        res_data = json.loads(resp.read().decode("utf-8"))
        client_rtt_ms = (t1 - t0) * 1000.0
        res_data["client_rtt_ms"] = client_rtt_ms
        return res_data


def percentile(data: List[float], p: float) -> float:
    """Computes the p-th percentile from an array of floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def render_ascii_histogram(data: List[float], bins: int = 10, width: int = 35) -> str:
    """Renders a compact text-based frequency distribution histogram."""
    if len(data) < 2:
        return "  (Insufficient samples for histogram)"
    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        return f"  All values identical: {min_val:.2f} ms"

    step = (max_val - min_val) / bins
    counts = [0] * bins
    for v in data:
        idx = min(int((v - min_val) / step), bins - 1)
        counts[idx] += 1

    max_count = max(counts) or 1
    lines = []
    for i in range(bins):
        low = min_val + i * step
        high = low + step
        bar_len = int((counts[i] / max_count) * width)
        bar = "█" * bar_len
        lines.append(f"  [{low:6.1f} - {high:6.1f} ms] | {bar:<{width}} ({counts[i]:2d})")
    return "\n".join(lines)


def run_benchmark_for_model(
    base_url: str,
    model: str,
    count: int,
    warmup: int,
    budget_ms: float,
) -> Dict[str, Any]:
    print(f"\n[*] Benchmarking model: [{model.upper()}] (Requests: {count}, Warmup: {warmup}, Budget: {budget_ms:.0f}ms)")

    # 1. Warmup
    if warmup > 0:
        print(f"    -> Warming up ({warmup} runs)...", end="", flush=True)
        for i in range(warmup):
            send_predict(base_url, model, dist=10.0, grounded=True, budget_ms=budget_ms)
            print(".", end="", flush=True)
        print(" done.")

    # 2. Main Benchmark Requests
    client_latencies: List[float] = []
    server_latencies: List[float] = []
    overheads: List[float] = []
    actions: Dict[str, int] = {"jump": 0, "run": 0}
    errors = 0

    print(f"    -> Dispatching {count} benchmark calls: [", end="", flush=True)
    for i in range(count):
        # Vary distances between 3.0 and 12.0 cells to simulate gameplay
        dist = 4.0 + (i % 6) * 1.5
        try:
            res = send_predict(base_url, model, dist=dist, grounded=True, budget_ms=budget_ms)
            client_ms = res["client_rtt_ms"]
            server_ms = res.get("inference_ms", client_ms)
            action = res.get("action", "run")

            client_latencies.append(client_ms)
            server_latencies.append(server_ms)
            overheads.append(max(0.0, client_ms - server_ms))
            actions[action] = actions.get(action, 0) + 1
            print("·", end="", flush=True)
        except Exception as e:
            errors += 1
            print("x", end="", flush=True)
    print("] Complete.")

    if not client_latencies:
        print(f"[!] No successful requests recorded for {model}.")
        return {}

    p50_client = percentile(client_latencies, 50)
    p90_client = percentile(client_latencies, 90)
    p95_client = percentile(client_latencies, 95)
    p99_client = percentile(client_latencies, 99)
    std_client = statistics.stdev(client_latencies) if len(client_latencies) > 1 else 0.0

    p50_server = percentile(server_latencies, 50)
    p90_server = percentile(server_latencies, 90)
    p95_server = percentile(server_latencies, 95)
    p99_server = percentile(server_latencies, 99)
    std_server = statistics.stdev(server_latencies) if len(server_latencies) > 1 else 0.0

    deadline = 45.0
    misses = sum(1 for ms in client_latencies if ms > deadline)
    miss_rate = (misses / len(client_latencies)) * 100.0

    return {
        "model": model,
        "count": len(client_latencies),
        "errors": errors,
        "actions": actions,
        "client": {
            "mean": statistics.mean(client_latencies),
            "std": std_client,
            "min": min(client_latencies),
            "max": max(client_latencies),
            "p50": p50_client,
            "p90": p90_client,
            "p95": p95_client,
            "p99": p99_client,
            "latencies": client_latencies,
        },
        "server": {
            "mean": statistics.mean(server_latencies),
            "std": std_server,
            "min": min(server_latencies),
            "max": max(server_latencies),
            "p50": p50_server,
            "p90": p90_server,
            "p95": p95_server,
            "p99": p99_server,
        },
        "overhead_mean_ms": statistics.mean(overheads) if overheads else 0.0,
        "deadline_ms": deadline,
        "misses": misses,
        "miss_rate_pct": miss_rate,
    }


def print_report(res: Dict[str, Any]):
    """Pretty prints structured benchmark results."""
    m = res["model"].upper()
    c = res["client"]
    s = res["server"]

    print("\n" + "=" * 68)
    print(f"  BENCHMARK REPORT // MODEL: {m}")
    print("=" * 68)
    print(f"  Total Requests:  {res['count']:<6} | Errors: {res['errors']} | Actions: {res['actions']}")
    print(f"  Transport Net:   {res['overhead_mean_ms']:.2f} ms average HTTP / IPC overhead")
    print(f"  45ms Deadline:   {res['misses']}/{res['count']} missed ({res['miss_rate_pct']:.1f}% violation rate)")
    print("-" * 68)
    print(f"  {'METRIC':<18} | {'SERVER INFERENCE':<20} | {'CLIENT ROUND-TRIP':<20}")
    print("-" * 68)
    print(f"  {'Mean ± Std':<18} | {s['mean']:>7.2f} ± {s['std']:<6.2f} ms   | {c['mean']:>7.2f} ± {c['std']:<6.2f} ms")
    print(f"  {'Min / Max':<18} | {s['min']:>7.2f} / {s['max']:<7.2f} ms  | {c['min']:>7.2f} / {c['max']:<7.2f} ms")
    print(f"  {'Median (p50)':<18} | {s['p50']:>16.2f} ms   | {c['p50']:>16.2f} ms")
    print(f"  {'p90':<18} | {s['p90']:>16.2f} ms   | {c['p90']:>16.2f} ms")
    print(f"  {'p95':<18} | {s['p95']:>16.2f} ms   | {c['p95']:>16.2f} ms")
    print(f"  {'p99':<18} | {s['p99']:>16.2f} ms   | {c['p99']:>16.2f} ms")
    print("-" * 68)
    print("  LATENCY DISTRIBUTION (Client RTT):")
    print(render_ascii_histogram(c["latencies"], bins=8, width=32))
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="LAYADINO Inference Latency Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="FastAPI Server base URL")
    parser.add_argument("--model", choices=["laya", "vega", "both"], default="laya", help="Model to benchmark")
    parser.add_argument("-n", "--count", type=int, default=20, help="Number of benchmark requests")
    parser.add_argument("--warmup", type=int, default=2, help="Number of initial warmup requests")
    parser.add_argument("--budget", type=float, default=0.0, help="Latency budget ms (0=Raw Async, 45=Fixed Hold)")
    parser.add_argument("--json", action="store_true", help="Print output as raw JSON")

    args = parser.parse_args()

    # Verify server health
    print(f"[*] Checking connection to {args.url}...")
    if not check_health(args.url):
        print(f"[!] Error: Could not reach backend server at {args.url}/health.")
        print("    Please start the server first with:")
        print("    .venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000")
        print("    or ./start.sh")
        sys.exit(1)
    print(f"[✓] Backend server is healthy.")

    models_to_test = ["laya", "vega"] if args.model == "both" else [args.model]
    results = {}

    for m in models_to_test:
        res = run_benchmark_for_model(
            base_url=args.url,
            model=m,
            count=args.count,
            warmup=args.warmup,
            budget_ms=args.budget,
        )
        if res:
            results[m] = res
            if not args.json:
                print_report(res)

    if args.json:
        # Strip raw arrays for clean JSON export unless requested
        json_clean = {}
        for k, v in results.items():
            entry = dict(v)
            entry["client"] = {ck: cv for ck, cv in entry["client"].items() if ck != "latencies"}
            json_clean[k] = entry
        print(json.dumps(json_clean, indent=2))


if __name__ == "__main__":
    main()
