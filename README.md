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

# Libratio Fleet

**A multi-agent RL environment where LLMs learn to manage GPU clusters — assigning mixed precision, allocating resources, detecting crashes, and recovering from failures.**

Built on the **Agentic Kernel**: a pure-math physics engine that replaces Docker/MicroVM sandboxes with deterministic reward evaluation at **11,281 evals/sec**.

**Theme**: Multi-Agent Interactions (Theme #1) + Fleet AI (Scalable Oversight)

| Link | URL |
|------|-----|
| Environment (HF Space) | [huggingface.co/spaces/SnippyCodes/Libratio](https://huggingface.co/spaces/SnippyCodes/Libratio) |
| Training Code | [Live Colab Notebook](https://colab.research.google.com/drive/1H5TrFlASqrfTsLxD2iM1CdW79iW_Ksh8?usp=sharing) |
| Blog / Pitch | [Hugging Face Blog Post](https://huggingface.co/spaces/SnippyCodes/Libratio/blob/main/Blog.MD) |
| Architecture Deep-Dive | [KERNEL.md](KERNEL.md) |

---

## 1. The Problem

At Meta scale, dozens of models train simultaneously on shared GPU clusters. A single precision misconfiguration wastes **40% of compute** or crashes the entire fleet.

The RL bottleneck? **Sandbox overhead.** Docker/MicroVM environments take 100–300ms to boot per trajectory evaluation, starving GPUs of training data:

| Method | Cold Start | Throughput | GPU Starvation |
|---|---|---|---|
| Docker sandbox | 100–300ms | ~10 evals/sec | Severe |
| MicroVM (Firecracker) | ~125ms | ~8 evals/sec | Severe |
| **Agentic Kernel (ours)** | **0ms** | **11,281 evals/sec** | **None** |

> Firecracker: [Agache et al., NSDI 2020]. Agentic Kernel benchmark: `python kernel_benchmark.py`

---

## 2. The Environment

### What the agent observes
Each agent sees the **cluster state** (total GPUs, VRAM budget, memory used), **its own model** (parameter count, layer distribution, priority), and **what other agents have already configured** — creating real multi-agent interdependence.

### What actions it can take
Four interdependent tasks with physics-based rewards:

| Task | What the Agent Does |
|------|-----|
| `fleet_precision` | Assign FP32/BF16/FP8 to each layer under a shared VRAM budget |
| `fleet_oversight` | Monitor loss trajectories across all models, detect which is crashing and why |
| `fleet_resource` | Allocate GPUs across competing models by priority and efficiency |
| `fleet_recovery` | Diagnose a mid-training crash, propose a fix, and verify recovery |

### How reward is computed
Rewards come from **pure mathematical physics** (NVIDIA Transformer Engine benchmarks), not heuristics. Three composable kernel modules evaluate each trajectory:

| Module | Evaluates | Physics Source |
|---|---|---|
| `PrecisionKernel` | Layer-level mixed precision quality | NVIDIA Transformer Engine |
| `ThermalKernel` | Junction temperature, power draw, throttle risk | H100 SXM5 TDP specs |
| `NetworkKernel` | All-Reduce communication overhead | NCCL benchmarks |

The reward signal is **rich and informative**: per-layer breadcrumb feedback, difference rewards (counterfactual fleet contribution), and hardware safety checks — not just a 0/1 outcome.

### Anti-reward-hacking
**Inverse Reward Design (IRD)** detects degenerate strategies and overrides the score to 0.01:
- All-FP8 everywhere (would crash training on embedding/output layers)
- All-FP32 everywhere (zero optimization, wastes fleet resources)
- Starving any model with 0 GPUs
- Flagging instability without evidence on the first monitoring window

### Quick UI Walkthrough

When you open the [live environment](https://huggingface.co/spaces/SnippyCodes/Libratio), here is what each element does:

| UI Element | What It Does |
|---|---|
| **Scenario Control** (dropdown) | Select one of the 4 tasks: Precision, Oversight, Resource, or Recovery |
| **Reset Episode** | Initializes a fresh cluster scenario. GPUs start cool (~29°C) and idle |
| **Step Environment** | Sends the JSON action to the Agentic Kernel and returns reward + new state |
| **Action JSON** (editor) | The agent's action. Edit this to test different strategies manually |
| **GPU Rack** (center) | Live visualization: each chip shows temperature, VRAM bar, core activity, and model assignment |
| **Cluster Metrics** (left) | Real-time GPU count, total/used/free VRAM, utilization percentage |
| **Telemetry Dashboard** (right) | Per-model health status, loss sparklines, and thermal/memory alerts |
| **Agent Reasoning Feed** | Shows the agent's reasoning text from each action |
| **Live Reward Timeline** | Chart tracking reward scores across steps |
| **Break the Agent (top-right)** | Splits the view: trained agent (left) vs random baseline (right) side-by-side |

**Try it yourself:** Reset → Step → watch the GPUs heat up → click "Break the Agent" to see how a random baseline crashes the cluster while the trained agent keeps it stable.

---

## 3. Training Results

`Llama-3.1-8B-Instruct` trained with **GRPO** (Unsloth + TRL), 400 steps on A10G GPU.

![Reward Curve](images/meanfleet.PNG)
*Mean fleet reward: 0.21 → 0.90 in ~80 steps. The model learned NVIDIA's precision constraints from pure physics rewards.*

![Loss Curve](images/policyloss.PNG)
*Policy loss actively updating throughout training — confirming genuine policy improvement.*

![Reward Variance](images/RewardVariance.PNG)
*Reward std dev drops from 0.35 → near 0. The model became consistent, not lucky.*

![Output Length](images/outputconvergence.PNG)
*Completion length converges to ~254 tokens — the model learned to output clean, deterministic JSON instead of hallucinating verbose reasoning.*

### Before vs After

![Baseline vs Trained](images/baseline_vs_trained.PNG)
*4.3x reward improvement: random baseline (0.21) vs GRPO-trained agent (0.90) after 400 steps.*

| Agent | Avg Reward | Behavior |
|---|---|---|
| **Untrained** | 0.21 | All-FP8 everywhere → NaN crashes, VRAM overflow |
| **Trained (400 steps)** | 0.90 | BF16 attention, FP8 FFN, FP32 embedding/output — optimal |

---

## 4. Why It Matters

**Who cares?** Anyone running large-scale GPU training: Meta (LLaMA), Google (Gemini), startups burning $50K+/day on H100 clusters.

Today, precision assignment and GPU allocation are done by **human ML engineers** using tribal knowledge. This environment proves that an LLM can learn these decisions from physics-based rewards alone — no hand-written heuristics, no human demonstrations.

The Agentic Kernel architecture also solves a fundamental problem for RL-based LLM training: **sandbox overhead**. By replacing Docker containers with pure math, we eliminate the 100–300ms cold-start bottleneck that starves GPUs of training data, enabling 1,000x faster trajectory evaluation.

---

## 5. Setup & Reproduction

```bash
# Run the environment server
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the kernel benchmark (reproduces the 11,281 evals/sec claim)
python kernel_benchmark.py

# Generate the training graphs (reproduces the PNGs in the README)
python generate_training_graphs.py

# Run multi-agent inference (requires HF_TOKEN or Groq API key)
export HF_TOKEN="your_token"
python fleet_inference.py
```

### Architecture

**OpenEnv** (standard interface) → **TRL GRPOTrainer** (optimization) → **Unsloth** (4-bit QLoRA acceleration) → **Agentic Kernel** (deterministic physics rewards)

> Full roadmap: multi-GPU topology modeling, live DCGM telemetry, curriculum learning, kernel-as-a-service. See [KERNEL.md](KERNEL.md).

---

## 6. RL Debugging Methodology

Following the official OpenEnv best practices, this environment was developed using strict isolation to prevent conflating environment bugs with optimizer failures:

1. **Manual Environment Debugging:** Executed `fleet_env.py` with handwritten JSON to verify physics transitions.
2. **Verifier Debugging:** Adversarially tested the reward function (e.g., trying to starve models or assign all-FP8) to build the Inverse Reward Design (IRD) guardrails.
3. **Scripted Baselines:** Ran random-action generators and greedy heuristic scripts to establish the untrained (0.21) baseline.
4. **Frozen Model Testing:** Sent zero-shot prompts to the base model (`fleet_inference.py`) to verify it understood the JSON format.
5. **Tiny RL Experiment:** Ran a 10-step GRPO test to ensure loss updating.
6. **Scaled Training:** Executed the final 400-step training run.
