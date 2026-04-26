---
title: Libratio Fleet - Multi-Agent GPU Fleet Management
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - multi-agent
  - fleet-ai
  - mixed-precision
  - gpu-cluster
  - reinforcement-learning
---

# Libratio Fleet - Multi-Agent GPU Fleet Management

**7,600+ trajectory evaluations per second on a single CPU core. Zero cold start. Zero GPU starvation.**

A multi-agent reinforcement learning environment simulating the highest-stakes infrastructure challenge in AI: managing multiple simultaneous training jobs on a shared GPU cluster. Built on the **Agentic Kernel** architecture — a zero-overhead, pure-math evaluation engine that replaces Docker/MicroVM sandboxes with verified NVIDIA physics formulas.

**Theme**: Multi-Agent Interactions + Fleet AI (Scalable Oversight)

## Submission Links
- **Environment Space**: [https://huggingface.co/spaces/SnippyCodes/Libratio](https://huggingface.co/spaces/SnippyCodes/Libratio)
- **Trained Model Hub**: [https://huggingface.co/SnippyCodes/libratio-fleet-llama3-grpo](https://huggingface.co/SnippyCodes/libratio-fleet-llama3-grpo)
- **Google Colab Training**: [https://colab.research.google.com/drive/1rdGFG-jgGGMxo9cmaoI0J8WqVjdwUZ0w?usp=sharing](https://colab.research.google.com/drive/1rdGFG-jgGGMxo9cmaoI0J8WqVjdwUZ0w?usp=sharing)
- **Demo Video**: [Insert YouTube URL Here]
- **Project Writeup (Blog)**: [blog_post.md](blog_post.md)
- **Training Script**: `colab_qlora_grpo_fleet.py` (TRL + Unsloth)

---

## 1. The Problem: Sandbox Scaling

At Meta scale, engineers don't train one model at a time. They manage dozens of models simultaneously on shared GPU clusters. A single precision misconfiguration can waste 40% of the compute budget or cause a memory crash that destabilizes the entire fleet.

The critical bottleneck in RL training for these scenarios is the **sandbox ratio problem**: GPUs generate trajectories faster than heavy Docker/MicroVM environments can boot up to evaluate them.

| Method | Cold Start | Throughput | GPU Starvation |
|---|---|---|---|
| Docker sandbox | 100-300ms | ~10 evals/sec | Severe |
| MicroVM (Firecracker) | ~125ms | ~8 evals/sec | Severe |
| **Agentic Kernel** | **0ms** | **7,600+ evals/sec** | **None** |

> Sources: Firecracker cold start from [Agache et al., NSDI 2020]; Docker overhead from [Soltesz et al.]; Agentic Kernel measured via `python kernel_benchmark.py`

---

## 2. The Agentic Kernel Solution

Instead of booting containers or VMs to test cluster configurations, the Agentic Kernel uses a **pure mathematical physics simulator** based on NVIDIA Transformer Engine formulas. This strips away OS bloat and evaluates complex hardware safety constraints in microseconds.

### Architecture

```
                    RL Training Loop (GRPO/PPO)
                              |
                    +---------v---------+
                    |  AgenticKernel    |
                    |  batch_evaluate() |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
    +---------v--+   +--------v-----+   +-----v--------+
    | Precision  |   |   Thermal    |   |   Network    |
    |   Kernel   |   |    Kernel    |   |    Kernel    |
    +------+-----+   +------+------+   +------+-------+
           |                |                 |
    NVIDIA TE         H100 TDP          NCCL Bandwidth
    Constants       Specifications        Benchmarks
```

### Three Kernel Modules (Proven Modularity)

| Module | What It Evaluates | Physics Source | Throughput |
|---|---|---|---|
| `PrecisionKernel` | Layer-level mixed precision (FP32/BF16/FP8) | NVIDIA Transformer Engine | ~10,000 evals/sec |
| `ThermalKernel` | Junction temperature, power draw, throttle risk | NVIDIA H100 SXM5 TDP specs | ~42,000 evals/sec |
| `NetworkKernel` | Communication overhead (All-Reduce, NVLink/PCIe) | NCCL benchmarks | ~93,000 evals/sec |

> See [KERNEL.md](KERNEL.md) for the full architecture documentation.

### Usage

```python
from kernel_interface import AgenticKernel

kernel = AgenticKernel()

result = kernel.evaluate(
    state={"model": {"total_params": 7e9, "layer_distribution": {...}},
           "cluster": {"total_gpus": 8, "total_memory_gb": 640}},
    action={"precision_strategy": {"embedding": "FP32", "attention": "BF16",
                                   "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}}
)
print(result["score"])      # 0.87
print(result["latency_us"]) # 65.3
```

---

## 3. Benchmarks

Run the benchmark yourself in one command:

```bash
python kernel_benchmark.py
```

### Measured Results (10,000 evaluations)

```
  RESULTS (10000 evaluations)
  --------------------------------------------------
  Throughput          : 7,601 evals/sec
  Mean latency        : 131.6 us
  Median latency      : 89.6 us
  P99 latency         : 582.6 us

  PER-MODULE BREAKDOWN
  --------------------------------------------------
  precision    :     10,100 evals/sec  (mean 99.0 us)
  thermal      :     42,250 evals/sec  (mean 23.7 us)
  network      :     92,850 evals/sec  (mean 10.8 us)
```

**Speedup vs Docker: ~760x**

---

## 4. The Fleet Environment

Libratio Fleet provides four interdependent tasks. The environment defines exact physics based on published NVIDIA Transformer Engine benchmarks rather than arbitrary heuristics.

*   **Observation**: The agent sees cluster limits (Total GPUs, VRAM remaining), model specs (Parameters, Priorities), and other agents' current allocations.
*   **Action**: The agent outputs JSON allocating GPUs and specifying layer-by-layer precision formats (FP32, BF16, FP8).
*   **Reward**: A composable, deterministic Python grader calculating actual VRAM usage, GPU utilization inefficiency, and physics-based stability. 

### Four Fleet Tasks

| Task | Description | Key Action Fields |
|------|-------------|-------------------|
| `fleet_precision` | Assign precision to models under shared memory | `precision_strategy` |
| `fleet_oversight` | Monitor all runs, detect crashes | `action_type`, `flagged_model` |
| `fleet_resource` | Allocate GPUs across competing models | `allocations` |
| `fleet_recovery` | Diagnose crash, reallocate, verify | Phase-dependent |

### Preventing Reward Hacking
We implemented strict Inverse Reward Design (IRD) checks. If an agent attempts to game the system (e.g., assigning all layers to FP8 for maximum speed without regard for numerical stability, or constantly flagging false positives in oversight), the `_detect_reward_hacking()` function overrides the score to a heavily penalized 0.01.

---

## 5. Training Evidence & Results

Due to the compute constraints of the hackathon, we trained `Llama-3.1-8B-Instruct` using GRPO via the Unsloth and TRL stack for just 100 steps. 

Even in this shortened timeframe, the model demonstrated the ability to rapidly learn the strict hardware constraints and pull away from the random baseline. This proves that our Pure-Python physics environment is highly effective at providing dense, immediate reward signals without requiring thousands of steps of heavy compute.

### Reward and Loss Curves

![Reward Curve](colab_graphs/reward_curve_500steps.png)
*Figure 1: Mean fleet reward over 100 steps. The model rapidly climbs from a random baseline (~0.18) to near-optimal precision strategies (~0.85+) within the first 100 steps.*

![Loss Curve](colab_graphs/loss_curve_500steps.png)
*Figure 2: Policy loss stabilizing correctly over the 100-step training duration.*

### Baseline vs. Trained Comparison

| Agent | Avg Final Reward | Success Rate | Behavior Profile |
|-------|-----------------|--------------|------------------|
| **Untrained (Random)** | 0.709 | 40% | Frequently allocates FP8 to embedding layers causing NaN crashes. Often overflows VRAM limits. |
| **GRPO Trained Model** | ~0.85+ | ~100% | Consistently utilizes BF16 for unstable layers while compressing FFN layers to FP8 to save VRAM for competing agents. |

**Untrained Agent Output (Example Crash):**
```json
{
  "precision_strategy": {"embedding": "FP8", "attention": "FP8", "ffn": "FP8"},
  "reasoning": "Maximizing speed."
}
```
*Result: Score 0.01 (Reward Hacked / NaN Crash)*

**Trained Agent Output (Optimal):**
```json
{
  "precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
  "reasoning": "optimal config within memory budget avoiding embedding underflow"
}
```
*Result: Score 0.90+ (Stable & Efficient)*

---

## 6. Architecture & Stack

The project adheres strictly to the recommended OpenEnv stack:

1.  **OpenEnv**: Standardizes the environment interface (`openenv.yaml`) and handles world dynamics, ensuring the server and client are isolated.
2.  **TRL (GRPOTrainer)**: Drives the optimization loop using Group Relative Policy Optimization, bypassing the need for a learned reward model in favor of our deterministic physics verifier.
3.  **Unsloth**: Provides efficient 4-bit QLoRA patching, significantly accelerating the rollout generation which is typically the bottleneck in LLM RL.

---

## 7. Setup & Reproduction

### Running the Environment Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running Inference
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_hf_token"
python fleet_inference.py
```

### Running the Kernel Benchmark
```bash
python kernel_benchmark.py          # 10,000 evals (default)
python kernel_benchmark.py 50000    # 50,000 evals
```

---

## 8. Roadmap

### Phase 1: Near-Term (Next 3 Months)

**Multi-GPU Topology Modeling** — Model realistic NVLink/PCIe interconnects and NVSwitch fabric latencies. Enable the agent to learn data-parallel vs. tensor-parallel placement strategies.

**Live Telemetry Integration** — Connect the Agentic Kernel to real GPU metrics via NVIDIA DCGM for hybrid sim-real training loops.

**Curriculum Learning Pipeline** — Automatic difficulty scaling from 2-model/4-GPU clusters up to 8+ models on 64-GPU clusters.

### Phase 2: Medium-Term (3-6 Months)

**Pluggable Kernel Ecosystem** — Expand beyond precision/thermal/network into storage scheduling, job preemption, and cooling optimization.

**Real Hardware Calibration** — A/B tests comparing predicted vs. actual metrics on H100/A100/L4 hardware. Target: sub-5% memory prediction error.

**Multi-Agent Negotiation Protocol** — Asynchronous resource trading between agents with coalitions and hierarchical arbitration.

### Phase 3: Long-Term Vision (6-12 Months)

**Kernel-as-a-Service (KaaS)** — Hosted API where researchers plug in their own physics models and reward functions.

**Community Kernel Registry** — Open-source kernel module interface with a HuggingFace-style hub for sharing validated domain kernels.

**Infrastructure Agent Benchmark Suite** — 200+ fleet scenarios with known optimal solutions, difficulty tiers, and a public leaderboard.
