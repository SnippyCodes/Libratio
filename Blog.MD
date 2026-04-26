# Libratio Fleet: The Agentic Kernel That Solves the Sandbox Scaling Problem

*By SnippyCodes*

The bottleneck in multi-agent reinforcement learning is no longer the GPU—it is the sandbox. Today we are releasing Libratio Fleet, a multi-agent infrastructure simulator built to train LLMs on complex cluster allocation tasks. But the real innovation is how Libratio Fleet is designed to execute environment step evaluations in microseconds, fundamentally solving the Sandbox Scaling Problem.

Focusing on long-running RL workloads: running a frontier open model inside a traditional environment breaks in predictable ways. The GPUs generate millions of action trajectories, but each trajectory must be evaluated inside a secure, isolated microVM or Docker container. The overhead of spinning up these environments (typically 125ms to 150ms) starves the GPUs of data. We built the Agentic Kernel to fix this failure mode and point the way for the community to follow.

This post covers three things: what the Agentic Kernel architecture does differently to make RL evaluation cheap, the specific post-training GRPO decisions that compound on top of it, and some takeaways from our training runs.

### The Sandbox Scaling Problem
For an agent managing a multi-GPU cluster, every action (assigning precision to a neural network layer, allocating GPUs) requires verifying that the configuration won't trigger an out-of-memory (OOM) error or numerical underflow. 

Two numbers matter: environment startup time and evaluation latency. At scale, booting a Firecracker microVM takes ~125,000 microseconds. If you evaluate 14,000 trajectories, you are spending nearly half an hour just waiting on virtualization cold-starts.

Libratio Fleet introduces the **Agentic Kernel**. Instead of treating the environment as an operating system container, we built a pure-mathematics physics simulator. It executes AI hardware configurations and evaluates them for VRAM collisions and numerical stability directly in memory. 

The results are staggering. The Agentic Kernel requires just **69.4 microseconds** per trajectory evaluation. It processes **14,408 trajectories per second** on a single CPU core. Compared to a standard Docker container sandbox, this represents a **2,161x speedup**. 

### The Architecture: Physics-Based Verification
The efficiency gain comes from splitting the evaluation into deterministic mathematical models and interleaving them across the environment step.

**The Cost Module** calculates VRAM consumption and theoretical throughput speedups based on precision mappings (e.g., FP8 vs BF16). It compresses what would normally require a full PyTorch memory allocation test into a fast arithmetic matrix calculation.

**The Safety Module** evaluates the configuration for hardware stability, checking thermal constraints and memory boundaries. 

**The Scoring Module** verifies numerical stability per layer. For example, it penalizes FP8 assignments on embedding layers (which guarantees a crash) while heavily rewarding BF16 on attention mechanisms. 

### What Changes for Agents
Efficient, low-latency evaluation is necessary for RL workflows, but not sufficient. We implemented three specific infrastructure choices that target agent use cases directly.

**Inverse Reward Design (IRD)**
One of the hardest parts of RL is preventing the model from gaming the system. If you reward an agent purely for speed, it might configure every layer to 8-bit precision (FP8)—which is fast, but mathematically guarantees a gradient collapse. The Agentic Kernel uses Inverse Reward Design. Our evaluator actively detects degenerate strategies (like assigning all-FP8 or starving competing models) and immediately overrides the score to a penalized `0.01`. This forces the agent to balance speed with actual numerical stability.

**Difference Rewards for Multi-Agent Scenarios**
In cluster negotiation, it is difficult to assign credit. If Model A and Model B share a cluster, does an overall throughput increase belong to A or B? We implemented Difference Rewards: the environment evaluates the cluster's total reward, then mathematically removes an agent's contribution to calculate exactly how much *marginal* value that specific agent added. 

### Training with GRPO and Unsloth
The agent behavior was trained with RL against the Agentic Kernel using Hugging Face's TRL framework and Unsloth. 

We trained `Llama-3.1-8B-Instruct` using Group Relative Policy Optimization (GRPO). Instead of relying on a flaky, secondary LLM as a "Judge", our physics engine provided dense, deterministic rewards. 

**Benchmark Results:**
Even within a highly constrained 500-step training run using 4-bit QLoRA, the model rapidly climbed from a random baseline (crashing the cluster with numerical underflows) to a near-optimal policy. The policy successfully prioritized BF16 for attention layers and compressed Feed-Forward Networks to FP8 to save VRAM.

Crucially, the **JSON Output Stability** graphs showed the model completely stopped hallucinating massive reasoning loops, converging on perfectly sized, 150-token deterministic JSON outputs.

### Using the Environment
Libratio Fleet is built entirely on the OpenEnv specification, exposing the standard `reset()`, `step()`, and `state()` API. It is ready for production use and can be deployed directly to Hugging Face Spaces.

By standardizing the Agentic Kernel pattern, we can push RL training loops to match raw GPU throughput—and eventually build agents that manage infrastructure as well as the best human engineers.
