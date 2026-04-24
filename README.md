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

> *Libratio (Latin): "the act of balancing"* - balancing precision, cost, and stability across an entire GPU fleet.

A **multi-agent** reinforcement learning environment simulating the highest-stakes infrastructure challenge in AI: **managing multiple simultaneous training jobs on a shared GPU cluster**. Agents cooperate to configure precision formats, compete for GPU resources, and an oversight agent monitors the entire fleet for training crashes.

**Theme**: Multi-Agent Interactions + Fleet AI (Scalable Oversight)

---

## Hackathon Submission Links
- **Hugging Face Space**: [Insert Space URL Here]
- **2-Minute Demo Video**: [Insert YouTube URL Here]
- **Training Script (Colab)**: [Run in Google Colab](https://colab.research.google.com/drive/1O74FkE8YCkren_sJHmx_p0i3hLAQ9Ylv)

---

## Training Evidence

![RL Training Reward Curve](./training_plot.png)
*Figure 1: GRPO training log from Colab showing the Llama-3.1-8B model's reward steadily increasing as it learns to statelessly detect precision-induced NaN crashes and format valid JSON actions.*

---

## The Problem

At Meta scale, you don't train ONE model - you train dozens simultaneously on shared GPU clusters:

- **LLaMA-70B** training cost: $1M+ in GPU compute
- **Precision misconfiguration**: wastes 10-40% of budget per training run
- **Fleet-wide crashes**: one misconfigured model can destabilize the entire cluster
- **Manual management**: takes expert engineers days per model

**Libratio Fleet** simulates this exact problem. Multiple AI agents learn to cooperate, coordinate resources, and catch failures - the same decisions Meta's infrastructure team makes every day, now trainable via RL.

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  GPU CLUSTER                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Model A │  │ Model B │  │ Model C │  Shared     │
│  │ LLaMA-7B│  │LLaMA-13B│  │Code-7B  │  Memory    │
│  └────┬────┘  └────┬────┘  └────┬────┘  Pool      │
│       │            │            │                   │
│  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐            │
│  │Agent A  │  │Agent B  │  │Agent C  │  Training   │
│  │Precision│  │Precision│  │Precision│  Agents     │
│  └─────────┘  └─────────┘  └─────────┘            │
│                     │                               │
│              ┌──────┴───────┐                      │
│              │  Oversight   │  Fleet AI             │
│              │    Agent     │  (monitors all runs)  │
│              └──────────────┘                      │
└─────────────────────────────────────────────────────┘

POST /fleet/reset  ->  { task_id }   ->  Initial Observation
POST /fleet/step   ->  { action }    ->  { observation, reward, done, info }
POST /fleet/state  ->  {}            ->  Current Fleet State
GET  /fleet/tasks  ->  {}            ->  Task Definitions (4 tasks)
```

---

## Fleet Tasks

| Task | Difficulty | Steps | What Happens |
|---|---|---|---|
| `fleet_precision` | Medium | 2-3 | Multiple agents assign precision to their models under shared memory |
| `fleet_oversight` | Hard | 1-5 | Oversight agent monitors ALL training runs, detects which model is crashing |
| `fleet_resource` | Hard | 5 | Agent allocates GPUs across competing models with priorities |
| `fleet_recovery` | Medium-Hard | 3 | Diagnose a crash -> fix precision -> verify recovery plan |

---

## Fleet Task Details

### Task 1: Fleet Precision Assignment

Each training agent configures precision for ONE model, but they share a cluster-wide GPU memory pool. Using too much memory affects other agents.

**Observation:**
```json
{
  "task_id": "fleet_precision",
  "cluster": {"total_gpus": 8, "total_memory_gb": 640, "memory_remaining_gb": 480},
  "your_model": {
    "model_id": "model_a", "name": "LLaMA-3-7B",
    "total_params": 7000000000, "priority": 3
  },
  "other_agents": {"model_b": {"memory_used_gb": 26.5, "status": "configured"}},
  "agents_remaining": 1
}
```

**Action:**
```json
{"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "optimal config within memory budget"}
```

---

### Task 2: Fleet Oversight Monitoring (Fleet AI)

The oversight agent sees loss trajectories from ALL models simultaneously. Must detect **which** model is crashing, **when**, and **why**.

**Observation:**
```json
{
  "task_id": "fleet_oversight",
  "model_trajectories": {
    "model_a": {"loss_window": [2.34, 2.18, 2.05, 1.98], "precision_config": {"embedding": "FP32", "attention": "BF16"}},
    "model_b": {"loss_window": [2.50, 2.30, null, null], "precision_config": {"embedding": "FP8", "attention": "BF16"}},
    "model_c": {"loss_window": [3.10, 2.95, 2.80, 2.70], "precision_config": {"embedding": "FP32", "attention": "BF16"}}
  },
  "windows_remaining": 3
}
```

**Action:**
```json
{
  "action_type": "flag_instability",
  "flagged_model": "model_b",
  "flagged_step": 42,
  "root_cause": "fp8 embedding underflow causing gradient collapse in model_b",
  "analysis": "model_b shows NaN values starting at step ~42, consistent with FP8 on embedding layer"
}
```

---

### Task 3: Fleet Resource Negotiation

Divide GPU resources across competing models with different priorities and sizes.

**Action:**
```json
{
  "allocations": {
    "model_a": {"gpus": 4, "precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}},
    "model_b": {"gpus": 3, "precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}},
    "model_c": {"gpus": 1, "precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"}}
  },
  "reasoning": "Priority-weighted allocation with memory-efficient precision strategies"
}
```

---

### Task 4: Fleet Recovery

One model crashes. Diagnose -> fix -> verify in 3 steps.

---

## Baseline Scores

| Task | Score | Model |
|---|---|---|
| fleet_precision | **0.994** | LLaMA-3.1-8B (Groq) |
| fleet_oversight | **0.830** | LLaMA-3.1-8B (Groq) |
| fleet_resource | **0.912** | LLaMA-3.1-8B (Groq) |
| fleet_recovery | **0.797** | LLaMA-3.1-8B (Groq) |
| **Average** | **0.883** | |

---

## Reward Functions

**Fleet Precision:** Per-agent layer scoring (physics model) + memory fairness + fleet stability bonus

**Fleet Oversight:** Correct model ID (+0.25) + correct timing (+0.20) + root cause (+0.25) + base detection (+0.30). False alarm = 0.10.

**Fleet Resource:** Cost efficiency (30%) + GPU utilization (25%) + priority alignment (25%) + validity (10%) + stability (10%)

**Fleet Recovery:** Diagnosis accuracy (phase 1) + config stability (phase 2) + reasoning quality (phase 3)

---

## Physics Model Sources

All reward functions are grounded in empirical benchmarks, not heuristics:

| Claim | Source |
|---|---|
| BF16 attention: 1.85x speedup | Kalamkar et al. (2019), BFloat16 for Deep Learning |
| FP8 FFN: 2.5x speedup | NVIDIA Transformer Engine (2023) |
| FP8 embedding: NaN crash | IEEE 754 analysis + LLaMA-3 training logs |
| H100 cost model | AWS/GCP H100 cloud pricing (2024) |
| LLaMA-3 FLOPs estimate | Meta AI (2024), LLaMA 3 Technical Report |

---

## Setup

### Docker (Recommended)
```bash
docker build -t libratio-fleet .
docker run -p 7860:7860 libratio-fleet
```

### Local Python
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running the Fleet Agent
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_api_key"
python fleet_inference.py
```

---

## Project Structure

```text
Libratio/
  environment/
    fleet_env.py             # Multi-agent fleet environment (4 tasks)
    physics_model.py         # Empirical benchmark constants
  scenarios/
    fleet_scenarios.py       # Fleet configs + oversight scenarios
    task1-4_scenarios.py     # Single-agent scenarios
  server/
    app.py                   # FastAPI: /fleet/* and /* endpoints
    models.py                # Pydantic models
  notebooks/
    final_colab_training.ipynb # GRPO RL training pipeline
  fleet_inference.py         # Multi-agent LLM inference script
  openenv.yaml               # OpenEnv spec (fleet tasks)
  Dockerfile
```

---

## Why Multi-Agent?

Most ML environments train a single agent on a single task. Real GPU infrastructure involves:

- **Cooperation**: Models share memory pools - one greedy agent affects all
- **Competition**: Higher-priority models should get more resources
- **Oversight**: Someone needs to watch ALL runs for crashes simultaneously
- **Recovery**: When things break, agents must diagnose and adapt in real-time

Libratio Fleet captures **all four** dynamics in a single environment.

---

*Built for the Meta × Scaler PyTorch OpenEnv Hackathon 2026 | Theme 1: Multi-Agent + Fleet AI*
