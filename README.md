---
title: Libratio — Mixed Precision Training Env
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - mixed-precision
  - reinforcement-learning
  - ml-engineering
---

# Libratio — Mixed Precision Training Environment

> *Libratio (Latin): "the act of balancing"* — the art of balancing precision, speed, and memory in modern AI training.

A reinforcement learning environment simulating one of the highest-stakes decisions in large-scale AI infrastructure: **configuring floating-point precision formats (FP32, BF16, FP16, FP8) for neural network layers** during training.

This is the exact problem Meta engineers face when training LLaMA at scale — assign the wrong precision to the wrong layer and you waste millions of dollars in GPU compute, or worse, crash an entire training run with NaN losses.

---

## Motivation

Training a 70B parameter model on H100 GPUs costs ~$1M+. Precision misconfiguration is one of the leading causes of:
- **Training crashes** (NaN loss from FP8 on embedding layers)
- **Wasted compute** (using FP32 everywhere when BF16 is safe)
- **Memory overflows** (underestimating the memory footprint of your precision strategy)

Libratio provides a structured RL environment where agents can learn, be evaluated, and improve at these decisions — grounded in real empirical benchmarks from NVIDIA Transformer Engine and Meta's LLaMA-3 technical reports.

---

## Environment Architecture

```
POST /reset  ->  { task_id }   ->  Initial Observation
POST /step   ->  { action }    ->  { observation, reward, done, info }
POST /state  ->  {}            ->  Current State
GET  /tasks  ->  {}            ->  Task Definitions (4 tasks)
```

The environment follows the [OpenEnv](https://openenv.dev) specification with typed models for all actions and observations.

---

## Tasks and Difficulty

| Task | Difficulty | Steps | What the Agent Does |
|---|---|---|---|
| `precision_assignment` | Easy | 5 | Assign FP32/BF16/FP16/FP8 to each layer one at a time |
| `instability_detection` | Medium | 5 | Observe loss trajectory windows, detect precision-induced crashes |
| `multi_objective_optimization` | Hard | 5 | Design a precision strategy satisfying memory, time, and accuracy constraints |
| `precision_transfer` | Medium-Hard | 3 | **Adapt** a working config from one model to a different architecture |

---

## Observation and Action Spaces

### Task 1: precision_assignment (Easy)

**Observation:**
```json
{
  "task_id": "precision_assignment",
  "model_name": "LLaMA-3-70B",
  "current_layer": {
    "name": "ffn_layer",
    "layer_type": "ffn",
    "num_params": 28000000000,
    "gradient_sensitivity": "low",
    "activation_range": "bounded"
  },
  "assigned_so_far": { "embedding": "FP32", "attention": "BF16" },
  "available_precisions": ["FP32", "BF16", "FP16", "FP8"],
  "memory_budget_gb": 80.0,
  "memory_used_gb": 12.4
}
```

**Action:**
```json
{ "precision": "FP8", "reasoning": "FFN layers have bounded activations safe for FP8" }
```

---

### Task 2: instability_detection (Medium)

**Observation:**
```json
{
  "task_id": "instability_detection",
  "precision_config": { "embedding": "FP8", "attention": "BF16", "ffn": "FP8" },
  "loss_trajectory_window": [2.34, 2.18, 2.05, null, null, null],
  "window_index": 1,
  "windows_remaining": 3
}
```
> `null` values represent NaN — a training crash.

**Action:**
```json
{
  "action_type": "flag_instability",
  "analysis": "NaN values detected starting at step 23",
  "flagged_step": 23,
  "root_cause": "fp8 embedding underflow causing gradient collapse"
}
```

---

### Task 3: multi_objective_optimization (Hard)

**Observation:**
```json
{
  "task_id": "multi_objective_optimization",
  "constraints": {
    "memory_budget_gb": 35.0,
    "time_budget_days": 5.5,
    "accuracy_threshold": 0.95
  },
  "iterations_remaining": 4,
  "best_score_so_far": 0.718,
  "previous_feedback": "Valid! Mem=27.5GB, Time=3.8d, Acc=0.991, Speedup=1.81x"
}
```

**Action:**
```json
{
  "precision_strategy": {
    "embedding": "FP32", "attention": "BF16",
    "ffn": "FP8", "layernorm": "BF16", "output": "FP32"
  },
  "reasoning": "Baseline strategy satisfies all constraints with maximum FFN throughput"
}
```

---

### Task 4: precision_transfer (Medium-Hard)

This task tests **generalization**: can the agent adapt a precision config that works on one model architecture to a completely different one?

**Observation:**
```json
{
  "task_id": "precision_transfer",
  "scenario_id": "small_to_large",
  "source_model": {
    "name": "LLaMA-3-7B",
    "total_params": 7000000000,
    "working_config": {"embedding": "FP32", "attention": "FP32", "ffn": "BF16", "layernorm": "FP32", "output": "FP32"},
    "metrics": {"memory_gb": 22.4, "accuracy": 0.998, "speedup_vs_fp32": 1.22}
  },
  "target_model": {
    "name": "LLaMA-3-70B",
    "total_params": 70000000000,
    "layer_distribution": {"embedding": 0.12, "attention": 0.30, "ffn": 0.42, "layernorm": 0.002, "output": 0.158},
    "constraints": {"memory_budget_gb": 180.0, "time_budget_days": 25.0, "accuracy_threshold": 0.96}
  },
  "iterations_remaining": 3
}
```

The agent sees:
- The source config uses FP32 for attention (safe but wasteful on a 7B model)
- The target is 10x larger (70B) with a 180GB memory budget
- Blindly copying FP32 attention would waste compute
- The agent should adapt: switch attention to BF16 and ffn to FP8 for the larger model

**Action:**
```json
{
  "precision_strategy": {
    "embedding": "FP32", "attention": "BF16",
    "ffn": "FP8", "layernorm": "BF16", "output": "FP32"
  },
  "reasoning": "Source used FP32 attention on 7B but 70B target needs more aggressive precision. Switching attention to BF16 and ffn to FP8 to fit memory budget while maintaining accuracy."
}
```

**Grading bonus:** The agent earns extra points for each layer it intelligently *changes* from the source config, up to +0.15. Blindly copying scores less.

---

## Reward Functions

**Task 1 — Per-step stability + efficiency:**
```
FP32 on embedding  ->  1.0  (IEEE 754 baseline, zero underflow risk)
BF16 on attention  ->  1.0  (Kalamkar2019: 1.85x throughput, stable)
FP8 on ffn         ->  1.0  (NVIDIA-TE 2023: 2.5x throughput, bounded activation)
FP8 on embedding   ->  0.0  (catastrophic: gradient underflow)
```

**Task 2 — Timing + root cause accuracy:**
- `continue_monitoring` on stable data: 0.6 per step
- Correct `flag_instability`: 0.4 base + 0.3 timing + 0.3 root cause = up to 1.0
- False alarm: 0.1 (heavy penalty)
- Missed crash: 0.1 (heavy penalty)

**Task 3 — Pareto frontier proximity:**
```python
score = 0.5 + 0.2 * mem_efficiency + 0.15 * time_efficiency + 0.15 * accuracy_margin
# 0.0 if any hard constraint is violated
```

**Task 4 — Transfer quality:**
```python
score = 0.4 (base) + efficiency_bonus (up to 0.45) + adaptation_bonus (up to 0.15)
# adaptation_bonus = 0.05 per layer intelligently changed from source
# 0.0 if target constraints violated
```

---

## Baseline Scores — Multi-Model Benchmark

Evaluated across 3 models using OpenAI-compatible APIs:

| Model | Provider | Task 1 (Precision) | Task 2 (Instability) | Task 3 (Optimization) | Average |
|---|---|---|---|---|---|
| llama-3.1-8b-instant | Groq | 1.000 | 0.667 | 0.486 | 0.718 |
| **llama-3.3-70b-versatile** | **Groq** | **1.000** | **0.625** | **0.719** | **0.781** |
| gemma-4-31b-it | Google AI Studio | 1.000 | 0.625 | 0.608 | 0.744 |

> Scores vary between runs due to random scenario selection. The physics model ensures graders are deterministic given the same inputs. Task 4 (precision_transfer) baseline expected: ~0.65.

---

## Setup and Usage

### Option A: Docker (Recommended)

```bash
docker build -t libratio .
docker run -p 7860:7860 libratio
```

### Option B: Local Python

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Running the Baseline Agent

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_api_key_here"
python inference.py
```

---

## Physics Model Sources

| Claim | Source |
|---|---|
| BF16 attention: 1.85x speedup | Kalamkar et al. (2019), BFloat16 for Deep Learning |
| FP8 FFN: 2.5x speedup | NVIDIA Transformer Engine (2023) |
| FP8 embedding: NaN crash | IEEE 754 analysis, confirmed in LLaMA-3 training logs |
| H100 cost model | AWS/GCP H100 cloud pricing (2024) |
| LLaMA-3 FLOPs estimate | Meta AI (2024), LLaMA 3 Technical Report |

---

## Project Structure

```
Libratio/
  app/
    main.py              # FastAPI: /reset, /step, /state, /tasks
  environment/
    mixed_precision_env.py   # Core state machine and graders
    physics_model.py         # Empirical benchmark constants
  scenarios/
    task1_scenarios.py   # Layer architectures (LLaMA-3 7B/13B/70B)
    task2_scenarios.py   # Loss trajectories (stable/unstable/spike)
    task3_scenarios.py   # Optimization scenarios (tight/balanced/generous)
    task4_scenarios.py   # Transfer scenarios (small-to-large, dense-to-MoE)
  inference.py           # Baseline LLM agent (OpenAI-compatible)
  openenv.yaml           # OpenEnv spec (4 tasks, 4 graders)
  Dockerfile             # Container for HF Spaces
  requirements.txt
```

---

## Why This Problem Matters

Every frontier AI lab (Meta, Google DeepMind, Anthropic) hits this problem at scale:

- GPT-4 training cost: estimated $100M+
- Precision misconfiguration: can waste 10-40% of compute budget
- Manual tuning: takes expert ML engineers days per model

An agent that can reliably configure mixed precision could save millions per training run and is directly applicable to production ML infrastructure.

---

*Built for the Meta x Scalar Hackathon 2026 | OpenEnv Specification*
