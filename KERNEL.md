# The Agentic Kernel Architecture

> **TL;DR**: The Agentic Kernel evaluates RL trajectories at **7,600+ evals/sec** on a single CPU core by replacing Docker/MicroVM sandboxes with pure mathematical physics evaluation. Zero cold start. Zero GPU starvation.

---

## The Problem: Sandbox Scaling

In production RL training, GPUs generate millions of action trajectories per hour. Each trajectory must be evaluated inside an environment to compute a reward signal. Traditional approaches use Docker containers or MicroVMs (Firecracker, gVisor) to isolate the evaluation — but the overhead destroys throughput:

| Method | Cold Start | Throughput | GPU Starvation |
|---|---|---|---|
| Docker sandbox | 100-300ms | ~10 evals/sec | Severe |
| MicroVM (Firecracker) | ~125ms | ~8 evals/sec | Severe |
| **Agentic Kernel** | **0ms** | **7,600+ evals/sec** | **None** |

The **sandbox ratio problem**: GPUs generate data faster than sandboxes can evaluate it. Result: GPU utilization drops below 30%, training costs balloon, and wall-clock time explodes.

## The Insight

For trusted, domain-specific environments — where the dynamics are deterministic mathematical functions grounded in published hardware specs — you don't need OS-level isolation. You need a **kernel**: a minimal, pure-math evaluation function that runs in-process.

## Architecture

```
                    RL Training Loop (GRPO/PPO)
                           |
                    +------v------+
                    | batch_evaluate([rollouts]) |
                    +------+------+
                           |
              +------------+------------+
              |            |            |
    +---------v--+ +-------v----+ +----v--------+
    | Precision  | |  Thermal   | |  Network    |
    |   Kernel   | |   Kernel   | |   Kernel    |
    +------+-----+ +-----+------+ +------+------+
           |              |               |
    NVIDIA TE       H100 TDP        NCCL Bandwidth
    Constants     Specifications      Benchmarks
```

### Module Composition

The `AgenticKernel` composes pluggable `KernelModule` instances:

```python
from kernel_interface import AgenticKernel, PrecisionKernel, ThermalKernel, NetworkKernel

# Default: all three modules
kernel = AgenticKernel()

# Custom: precision-only (fastest)
kernel = AgenticKernel(
    modules=[PrecisionKernel()],
    weights={"precision": 1.0}
)

# Custom blend
kernel = AgenticKernel(
    modules=[PrecisionKernel(), ThermalKernel()],
    weights={"precision": 0.7, "thermal": 0.3}
)
```

### Kernel Modules

| Module | What It Evaluates | Physics Source | Throughput |
|---|---|---|---|
| `PrecisionKernel` | Layer-level mixed precision (FP32/BF16/FP8) | NVIDIA Transformer Engine | ~10,000 evals/sec |
| `ThermalKernel` | Junction temperature, power draw, throttle risk | NVIDIA H100 SXM5 TDP specs | ~42,000 evals/sec |
| `NetworkKernel` | Communication overhead (All-Reduce, NVLink) | NCCL benchmarks | ~93,000 evals/sec |

### API

```python
# Single evaluation
result = kernel.evaluate(state, action)
# Returns: {"score": 0.87, "feedback": "...", "breakdown": {...}, "latency_us": 65.3}

# Batch evaluation (for GRPO rollouts)
results = kernel.batch_evaluate([
    {"state": s1, "action": a1},
    {"state": s2, "action": a2},
    ...
])

# Performance introspection
stats = kernel.get_throughput_stats()
# Returns: throughput, mean/median/p99 latency, total evals
```

## What Each Module Does

### PrecisionKernel (60% weight)

Evaluates whether a layer-by-layer precision assignment (embedding=FP32, attention=BF16, ffn=FP8, etc.) is:
- **Numerically stable**: FP8 on embedding layers = guaranteed NaN crash
- **Memory efficient**: stays within fair share of cluster VRAM
- **Hardware safe**: doesn't exceed memory/power utilization danger thresholds

Includes **Inverse Reward Design (IRD)**: detects degenerate strategies (all-FP8, all-FP32, empty) and immediately blocks them with score=0.01.

### ThermalKernel (25% weight)

Models GPU junction temperature under sustained training load:
- Calculates power draw from precision-dependent power multipliers
- Estimates junction temperature using thermal resistance model
- Accounts for memory utilization's effect on chip heating
- Risk levels: OPTIMAL < MODERATE < HIGH < THROTTLING < SHUTDOWN

### NetworkKernel (15% weight)

Evaluates communication overhead for distributed training:
- Supports topology types: PCIe Gen4/5, NVLink v3/v4, InfiniBand NDR
- Models All-Reduce communication volume (Megatron-LM formula)
- Penalizes poor topology/parallelism combinations that waste compute

## Running the Benchmark

```bash
# Default: 10,000 evaluations
python kernel_benchmark.py

# Custom count
python kernel_benchmark.py 50000
```

The benchmark outputs:
1. **Overall throughput** (all modules composed)
2. **Per-module throughput** (isolated performance)
3. **Batch mode performance** (GRPO-style evaluation)
4. **Comparison table** vs Docker/MicroVM alternatives

## Adding a New Kernel Module

```python
from kernel_interface import KernelModule, _clamp

class StorageKernel(KernelModule):
    """Checkpoint I/O scheduling evaluation."""
    name = "storage"

    def evaluate(self, state: dict, action: dict) -> dict:
        # Your physics evaluation here
        score = compute_checkpoint_overhead(...)
        return {
            "score": _clamp(score),
            "feedback": f"storage={score:.3f}",
            "details": {...},
        }

# Register it
kernel = AgenticKernel(
    modules=[PrecisionKernel(), ThermalKernel(), StorageKernel()],
    weights={"precision": 0.5, "thermal": 0.25, "storage": 0.25}
)
```

## Why This Matters

The Agentic Kernel pattern is generalizable to any domain where environment dynamics can be expressed as deterministic mathematical functions:

- **Energy grid management**: physics of power flow, thermal limits
- **Supply chain optimization**: logistics cost functions, capacity constraints
- **Network routing**: bandwidth, latency, congestion models
- **Autonomous vehicle planning**: kinematic models, collision physics

In all these domains, the environment evaluation is the RL bottleneck. The Agentic Kernel eliminates it.
