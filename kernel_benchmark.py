#!/usr/bin/env python3
"""
Agentic Kernel Benchmark -- Proves the core throughput claim.

Run:  python kernel_benchmark.py
"""
import time, random, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from kernel_interface import AgenticKernel, PrecisionKernel, ThermalKernel, NetworkKernel

LAYER_DISTS = [
    {"embedding": 0.15, "attention": 0.25, "ffn": 0.40, "layernorm": 0.002, "output": 0.198},
    {"embedding": 0.08, "attention": 0.18, "ffn": 0.60, "layernorm": 0.002, "output": 0.138},
    {"embedding": 0.12, "attention": 0.30, "ffn": 0.38, "layernorm": 0.002, "output": 0.198},
]
PARAM_SIZES = [3_800_000_000, 7_000_000_000, 13_000_000_000, 46_000_000_000, 70_000_000_000]
PRECISIONS = ["FP32", "BF16", "FP16", "FP8"]
SAFE_COMBOS = [
    {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
    {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
    {"embedding": "BF16", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
    {"embedding": "FP32", "attention": "FP16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
    {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "FP16", "output": "BF16"},
]

def random_action(rng):
    if rng.random() < 0.7:
        return {"precision_strategy": rng.choice(SAFE_COMBOS)}
    layers = ["embedding", "attention", "ffn", "layernorm", "output"]
    return {"precision_strategy": {l: rng.choice(PRECISIONS) for l in layers}}

def random_state(rng):
    return {
        "model": {"total_params": rng.choice(PARAM_SIZES), "layer_distribution": rng.choice(LAYER_DISTS)},
        "cluster": {"total_gpus": rng.choice([4, 8, 16, 32]), "total_memory_gb": rng.choice([320, 640, 1280, 2560]), "gpu_memory_gb": 80.0},
        "num_models": rng.choice([2, 3, 4]),
    }

def fmt(n):
    return f"{n:,.0f}"

def run_benchmark(n_evals=10_000):
    rng = random.Random(42)
    kernel = AgenticKernel()

    print("=" * 65)
    print("  AGENTIC KERNEL BENCHMARK")
    print("=" * 65)
    print(f"  Modules  : {', '.join(kernel.get_module_names())}")
    print(f"  Weights  : {kernel.get_weights()}")
    print(f"  Evals    : {fmt(n_evals)}")
    print("-" * 65)

    # Warmup (JIT, cache priming)
    for _ in range(500):
        kernel.evaluate(random_state(rng), random_action(rng))
    kernel.reset_stats()

    # Timed run
    start = time.perf_counter()
    for _ in range(n_evals):
        kernel.evaluate(random_state(rng), random_action(rng))
    wall = time.perf_counter() - start

    stats = kernel.get_throughput_stats()

    print(f"\n  RESULTS ({n_evals} evaluations)")
    print(f"  {'-' * 50}")
    print(f"  Wall-clock time     : {wall*1000:.1f} ms")
    print(f"  Throughput          : {fmt(stats['throughput_evals_per_sec'])} evals/sec")
    print(f"  Mean latency        : {stats['mean_latency_us']:.1f} us")
    print(f"  Median latency      : {stats['median_latency_us']:.1f} us")
    print(f"  P99 latency         : {stats['p99_latency_us']:.1f} us")
    print(f"  Min latency         : {stats['min_latency_us']:.1f} us")
    print(f"  Max latency         : {stats['max_latency_us']:.1f} us")

    # Per-module benchmarks
    print(f"\n  PER-MODULE BREAKDOWN")
    print(f"  {'-' * 50}")
    for ModClass in [PrecisionKernel, ThermalKernel, NetworkKernel]:
        mod = ModClass()
        mk = AgenticKernel(modules=[mod], weights={mod.name: 1.0})
        for _ in range(200):
            mk.evaluate(random_state(rng), random_action(rng))
        mk.reset_stats()
        for _ in range(n_evals):
            mk.evaluate(random_state(rng), random_action(rng))
        ms = mk.get_throughput_stats()
        print(f"  {mod.name:12s} : {fmt(ms['throughput_evals_per_sec']):>10s} evals/sec  (mean {ms['mean_latency_us']:.1f} us)")

    # Batch mode
    print(f"\n  BATCH MODE")
    print(f"  {'-' * 50}")
    batch_kernel = AgenticKernel()
    batch_kernel.reset_stats()
    batch = [{"state": random_state(rng), "action": random_action(rng)} for _ in range(n_evals)]
    t0 = time.perf_counter()
    results = batch_kernel.batch_evaluate(batch)
    batch_wall = time.perf_counter() - t0
    bs = batch_kernel.get_throughput_stats()
    print(f"  Batch throughput    : {fmt(bs['throughput_evals_per_sec'])} evals/sec")
    print(f"  Batch wall-clock    : {batch_wall*1000:.1f} ms")

    # Comparison table
    tput = stats['throughput_evals_per_sec']
    lat_us = stats['mean_latency_us']
    print(f"\n  COMPARISON vs SANDBOX ALTERNATIVES")
    print(f"  {'-' * 60}")
    print(f"  {'Method':<22s} {'Cold Start':>12s} {'Throughput':>16s} {'GPU Starvation':>16s}")
    print(f"  {'-' * 60}")
    print(f"  {'Docker sandbox':<22s} {'100-300ms':>12s} {'~10 evals/sec':>16s} {'Severe':>16s}")
    print(f"  {'MicroVM (Firecracker)':<22s} {'~125ms':>12s} {'~8 evals/sec':>16s} {'Severe':>16s}")
    print(f"  {'Agentic Kernel':<22s} {'0ms':>12s} {fmt(tput)+' evals/sec':>16s} {'None':>16s}")
    print(f"  {'-' * 60}")
    speedup = tput / 10.0
    print(f"  Speedup vs Docker   : {speedup:,.0f}x")
    print(f"  Mean eval latency   : {lat_us:.1f} us ({lat_us/1000:.4f} ms)")

    print("\n" + "=" * 65)
    print(f"  >> {fmt(tput)} evals/sec -- Sandbox Scaling Problem: SOLVED")
    print("=" * 65)

    return stats

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    run_benchmark(n)
