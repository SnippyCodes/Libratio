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

---

## 5. MongoDB Intelligent Core

Instead of treating MongoDB as a basic JSON dump, Libratio Fleet integrates MongoDB Atlas as its intelligent MLOps backend using advanced native features:

```
                                 ┌────────────────────────────────────┐
                                 │       Agentic RL Environment       │
                                 └─────────────────┬──────────────────┘
                                                   │ Step Telemetry &
                                                   │ Run Metrics
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            MongoDB Atlas                                             │
│                                                                                                      │
│  ┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌────────────────────────┐  │
│  │     trajectories (flat)       │   │ gpu_telemetry_metrics (Time)  │   │      runs (telemetry)  │  │
│  │ - Gemini 768d Embeddings     │   │ - Auto-compressed Metrics     │   │ - Average scores/runs  │  │
│  │ - $vectorSearch Index        │   │ - Sorted by step/timestamp    │   │ - Change Stream target │  │
│  └───────────────▲───────────────┘   └───────────────▲───────────────┘   └───────────┬────────────┘  │
│                  │                                   │                               │               │
└──────────────────┼───────────────────────────────────┼───────────────────────────────┼───────────────┘
                   │                                   │                               │
                   │ Semantic RAG                      │ Fetch Profile                 │ Trigger event
                   │                                   │                               │
         ┌─────────┴─────────┐               ┌─────────┴─────────┐                     │
         │  Predictive Agent │               │ mongodb_streams.py│◄────────────────────┘
         │ (Gemini RAG SRE)  │               │  (Change Stream)  │
         └───────────────────┘               └─────────┬─────────┘
                                                       │ Log Incident
                                                       ▼
                                             ┌───────────────────┐
                                             │ incident_reports  │
                                             └───────────────────┘
```

### 🧠 1. Atlas Vector Search (Semantic RAG)
- **Data Serialization**: Trajectory data (model, layer precisions, memory, thermals, outcomes) is serialized into semantic natural language.
- **Gemini Embeddings**: Serialized text is embedded into a 768-dim space via Google's `models/gemini-embedding-001`.
- **Cosine Similarity Match**: At query time, the live cluster state is embedded and searched via MongoDB Atlas `$vectorSearch` (`trajectory_vector_index`) against past crash documents.
- **RAG Recovery**: Rather than simple exact filters (e.g. `memory > 200`), Vector Search finds records that *feel* semantically similar, enabling the agent to learn from fuzzy historic contexts.

### ⏱️ 2. Native Time Series Metrics Logging
- **Optimized Storage**: Step-by-step telemetry (VRAM, thermal status, power, reward) is saved to a native MongoDB Time Series collection (`gpu_telemetry_metrics`).
- **Granular Grouping**: Documents are partition-organized by `timeField: "timestamp"` and `metaField: "metadata"`. MongoDB automatically compresses the data (saving up to 95% disk space) and indexes it for rapid historical querying.

### ⚡ 3. Real-Time MLOps Change Streams
- **Event-Driven Diagnostics**: A background daemon (`mongodb_streams.py`) listens to a MongoDB `$changeStream` on the `runs` collection.
- **Auto-Triaging**: If a run completes with a poor score (<0.60), the listener catches the insertion event, queries the Time Series collection to build a step-by-step performance profile of that execution, runs a heuristic diagnostic, and posts a detailed post-incident report to the `incident_reports` collection.

### 📊 4. Aggregation Analytics Dashboard
- Exposes an `/api/analytics` endpoint on the FastAPI server.
- Uses complex multi-stage `$group`, `$sort`, and conditional `$cond` aggregation pipelines to calculate running average scores per task, success percentages, and cross-model performance benchmarks directly inside the database.

---

## 6. Setup & Reproduction

```bash
# 1. Install dependencies (includes Google ADK + MongoDB MCP support)
pip install -r requirements.txt

# 2. Copy and fill in your secrets
cp .env.example .env
# Edit .env: set MONGO_URI, GEMINI_API_KEY, HF_TOKEN, GROQ_API_KEY

# 3. Run the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 4. Run the kernel benchmark (reproduces the 11,281 evals/sec claim)
python kernel_benchmark.py

# 5. Generate the training graphs (reproduces the PNGs in the README)
python generate_training_graphs.py

# 6. Run multi-agent inference (requires HF_TOKEN or Groq API key)
export HF_TOKEN="your_token"
python fleet_inference.py
```

### Google ADK Agent (MongoDB MCP Integration)

```bash
# Run the Google ADK agent directly (uses Gemini + MongoDB MCP Server)
# Requires: GEMINI_API_KEY + MONGO_URI in .env, and npx available
python agent/adk_agent.py

# Or trigger via the API while the server is running:
curl -X POST http://localhost:7860/adk/run \
  -H "Content-Type: application/json" \
  -d '{"task_id": "fleet_precision"}'

# Check ADK + MongoDB MCP status:
curl http://localhost:7860/adk/status
```

### Architecture

**OpenEnv** (standard interface) → **TRL GRPOTrainer** (optimization) → **Unsloth** (4-bit QLoRA acceleration) → **Agentic Kernel** (deterministic physics rewards)

**ADK Layer**: Gemini 2.0 Flash → Google ADK MCPToolset → MongoDB MCP Server → MongoDB Atlas

---

> See **Section 5** above for the full MongoDB Atlas Intelligent Core architecture.


## 7. RL Debugging Methodology

Following the official OpenEnv best practices, this environment was developed using strict isolation to prevent conflating environment bugs with optimizer failures:

1. **Manual Environment Debugging:** Executed `fleet_env.py` with handwritten JSON to verify physics transitions.
2. **Verifier Debugging:** Adversarially tested the reward function (e.g., trying to starve models or assign all-FP8) to build the Inverse Reward Design (IRD) guardrails.
3. **Scripted Baselines:** Ran random-action generators and greedy heuristic scripts to establish the untrained (0.21) baseline.
4. **Frozen Model Testing:** Sent zero-shot prompts to the base model (`fleet_inference.py`) to verify it understood the JSON format.
5. **Tiny RL Experiment:** Ran a 10-step GRPO test to ensure loss updating.
6. **Scaled Training:** Executed the final 400-step training run.

---

## 8. Using Your Custom Trained Model

The project ships with a GRPO-fine-tuned **LLaMA-3.1-8B** adapter published at [`SnippyCodes/libratio-fleet-llama3-grpo`](https://huggingface.co/SnippyCodes/libratio-fleet-llama3-grpo) and served via [Featherless AI](https://featherless.ai). You can swap in your own retrained adapter in three ways:

### Option A — HuggingFace (default, cloud)

The agent automatically routes `fleet_precision` and `fleet_recovery` tasks to your custom model. Set these in `.env`:

```bash
HF_TOKEN=your_hf_token
HF_MODEL_URL=https://router.huggingface.co/v1/chat/completions
HF_MODEL_NAME=YOUR_HF_USERNAME/your-model-name:featherless-ai
```

### Option B — Local vLLM Server (fastest, GPU required)

After training your adapter with `colab_qlora_grpo_fleet.py`:

```bash
# Serve the merged model locally
pip install vllm
vllm serve results/qlora_grpo_fleet_merged_16bit \
  --host 0.0.0.0 --port 8000 \
  --served-model-name libratio-fleet-local
```

Then update `.env`:

```bash
API_BASE_URL=http://localhost:8000/v1
MODEL_NAME=libratio-fleet-local
HF_TOKEN=not-needed
HF_MODEL_URL=http://localhost:8000/v1/chat/completions
HF_MODEL_NAME=libratio-fleet-local
```

### Option C — Route ALL Tasks to Your Custom Model

By default, `fleet_oversight` and `fleet_resource` use Groq (larger model). To route everything to your model, edit `fleet_inference.py` line ~152:

```python
# Change this:
if task_id in ["fleet_oversight", "fleet_resource"] or "groq.com" in current_base_url:
# To this (disable Groq routing):
if False:
```

### Retraining from Scratch

```bash
# In Google Colab (free T4 GPU):
# 1. Open colab_qlora_grpo_fleet.py
# 2. Update REPO_URL to your fork
# 3. Run — training takes ~2-3 hours on T4
# 4. Adapter is saved to results/qlora_grpo_fleet/

# Push your new adapter to HuggingFace Hub:
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="results/qlora_grpo_fleet",
    repo_id="YOUR_USERNAME/libratio-fleet-v2",
    repo_type="model"
)
```
