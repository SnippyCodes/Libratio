"""
FastAPI application for Libratio Fleet OpenEnv.
Includes API endpoints and a lightweight frontend for demos.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import sys, os
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── ADK Agent (Google Cloud Agent Builder integration) ──
try:
    from agent.adk_agent import run_fleet_task_with_adk, ADK_AVAILABLE
except ImportError:
    ADK_AVAILABLE = False
    async def run_fleet_task_with_adk(task_id: str = "fleet_precision"):
        return {"error": "google-adk not installed", "adk_available": False}

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "libratio"

from environment.fleet_env import FleetEnvironment
from server.models import (
    ResetRequest, ResetResponse, StepResponse, StateResponse,
    TaskDefinition, RewardPayload,
)

app = FastAPI(
    title="Libratio Fleet — Multi-Agent GPU Fleet Management",
    description="OpenEnv for multi-agent GPU cluster precision optimization and fleet oversight",
    version="3.0.0"
)

# Allow browser frontends on any origin to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
COLAB_GRAPHS_DIR = ROOT_DIR / "colab_graphs"
HF_GRAPHS_DIR = ROOT_DIR / "hf_graphs"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
if COLAB_GRAPHS_DIR.exists():
    app.mount("/colab_graphs", StaticFiles(directory=str(COLAB_GRAPHS_DIR)), name="colab_graphs")
if HF_GRAPHS_DIR.exists():
    app.mount("/hf_graphs", StaticFiles(directory=str(HF_GRAPHS_DIR)), name="hf_graphs")


# Multi-agent fleet environment (new)
fleet_env = FleetEnvironment()


def _clamp_score(raw_score) -> float:
    """Defense-in-depth: clamp score to strict (0.01, 0.99) interval."""
    try:
        val = float(raw_score)
    except (ValueError, TypeError):
        val = 0.01
    return float(max(0.01, min(0.99, val)))


@app.get("/")
def root():
    if FRONTEND_DIR.exists():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    return {
        "status": "ok",
        "environment": "Libratio Fleet — Multi-Agent GPU Fleet Management",
        "version": "3.0.0",
        "modes": {
            "fleet": "Multi-agent fleet environment (/fleet/reset, /fleet/step, /fleet/state, /fleet/tasks)",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "Libratio Fleet — Multi-Agent GPU Fleet Management",
        "version": "3.0.0",
    }




# ═══════════════════════════════════════════════════════════
# FLEET MODE — Multi-agent fleet endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/fleet/tasks", response_model=List[TaskDefinition])
def get_fleet_tasks() -> List[TaskDefinition]:
    return [TaskDefinition(**t) for t in fleet_env.TASK_DEFS]


@app.post("/fleet/reset", response_model=ResetResponse)
def reset_fleet(payload: dict = {}):
    task_id = payload.get("task_id", "fleet_precision")
    obs = fleet_env.reset(task_id)
    return ResetResponse(observation=obs)


@app.post("/fleet/step", response_model=StepResponse)
def step_fleet(payload: dict = {}):
    action = payload.get("action", payload)
    result = fleet_env.step(action)
    clamped_score = _clamp_score(result["reward"]["score"])
    return StepResponse(
        observation=result["observation"],
        reward=RewardPayload(
            score=clamped_score,
            feedback=result["reward"]["feedback"],
        ),
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/fleet/state", response_model=StateResponse)
def get_fleet_state():
    s = fleet_env.state()
    return StateResponse(**s)


@app.get("/api/analytics")
def get_fleet_analytics():
    """
    Exposes real-time fleet analytics aggregated via MongoDB aggregation pipelines.
    Provides performance stats per task and compares avg scores per model.
    """
    if not MONGO_URI:
        return {"database_status": "not_configured", "error": "MONGO_URI not set in env"}

    client_mongo = None
    try:
        client_mongo = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
        db_mongo = client_mongo[DB_NAME]
        runs_col = db_mongo["runs"]

        # ── Pipeline 1: Task summary statistics ──
        pipeline_tasks = [
            {
                "$group": {
                    "_id": "$task_id",
                    "avg_score": {"$avg": "$score"},
                    "max_score": {"$max": "$score"},
                    "min_score": {"$min": "$score"},
                    "success_rate": {
                        "$avg": {
                            "$cond": [{"$eq": ["$success", True]}, 1.0, 0.0]
                        }
                    },
                    "total_runs": {"$sum": 1}
                }
            },
            {"$sort": {"avg_score": -1}}
        ]
        task_stats = list(runs_col.aggregate(pipeline_tasks))

        # ── Pipeline 2: Model performance comparisons ──
        pipeline_models = [
            {
                "$group": {
                    "_id": "$model_name",
                    "avg_score": {"$avg": "$score"},
                    "total_runs": {"$sum": 1}
                }
            },
            {"$sort": {"avg_score": -1}}
        ]
        model_stats = list(runs_col.aggregate(pipeline_models))

        # Format aggregation outputs into user-friendly stats
        formatted_tasks = []
        for t in task_stats:
            formatted_tasks.append({
                "task_id": t["_id"],
                "avg_score": round(t["avg_score"], 3),
                "max_score": round(t["max_score"], 3),
                "min_score": round(t["min_score"], 3),
                "success_rate_pct": round(t["success_rate"] * 100, 1),
                "total_runs": t["total_runs"]
            })

        formatted_models = []
        for m in model_stats:
            formatted_models.append({
                "model_name": m["_id"],
                "avg_score": round(m["avg_score"], 3),
                "total_runs": m["total_runs"]
            })

        return {
            "database_status": "connected",
            "total_runs_logged": sum(t["total_runs"] for t in task_stats),
            "task_analytics": formatted_tasks,
            "model_analytics": formatted_models
        }
    except Exception as e:
        return {
            "database_status": "failed_connection",
            "error": str(e)
        }
    finally:
        if client_mongo:
            client_mongo.close()


@app.get("/api/search")
def search_trajectories(query: str, type: str = "hybrid", limit: int = 5):
    """
    Search historical trajectories using MongoDB Atlas Search.
    Supports 'text' (full-text search), 'vector' (semantic vector search),
    or 'hybrid' (combined RRF fusion).
    """
    if not MONGO_URI:
        return {"error": "MONGO_URI not set in env"}
        
    client_mongo = None
    try:
        client_mongo = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
        db_mongo = client_mongo[DB_NAME]
        collection = db_mongo["trajectories"]
        
        if type == "text":
            from mongodb_vector import text_search_similar_trajectories
            results = text_search_similar_trajectories(collection, query, top_k=limit)
        elif type == "vector":
            from mongodb_vector import generate_embedding
            query_vector = generate_embedding(query, task_type="RETRIEVAL_QUERY")
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "trajectory_vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 10,
                        "limit": limit,
                    }
                },
                {
                    "$addFields": {
                        "vector_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "embedding": 0
                    }
                }
            ]
            results = list(collection.aggregate(pipeline))
            for res in results:
                res["_id"] = str(res["_id"])
        else: # hybrid
            from mongodb_vector import generate_embedding
            query_vector = generate_embedding(query, task_type="RETRIEVAL_QUERY")
            
            # Vector Search Pipeline
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "trajectory_vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 10,
                        "limit": limit * 2,
                    }
                },
                {
                    "$addFields": {
                        "vector_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "embedding": 0
                    }
                }
            ]
            
            # Text Search Pipeline
            text_pipeline = [
                {
                    "$search": {
                        "index": "trajectory_text_search_index",
                        "text": {
                            "query": query,
                            "path": ["embedding_source_text", "failure_reason", "model_name", "outcome"]
                        }
                    }
                },
                {
                    "$addFields": {
                        "search_score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$limit": limit * 2
                },
                {
                    "$project": {
                        "embedding": 0
                    }
                }
            ]
            
            vector_results = list(collection.aggregate(vector_pipeline))
            text_results = list(collection.aggregate(text_pipeline))
            
            # RRF merge
            K = 60
            rrf_scores = {}
            doc_map = {}
            
            for rank, doc in enumerate(vector_results):
                doc_id = str(doc["_id"])
                doc_map[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (K + rank + 1))
                doc["vector_rank"] = rank + 1
                
            for rank, doc in enumerate(text_results):
                doc_id = str(doc["_id"])
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                else:
                    if "search_score" in doc:
                        doc_map[doc_id]["search_score"] = doc["search_score"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (K + rank + 1))
                doc_map[doc_id]["text_rank"] = rank + 1
                
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            results = []
            for doc_id in sorted_ids[:limit]:
                doc = doc_map[doc_id]
                doc["rrf_score"] = rrf_scores[doc_id]
                doc["_id"] = doc_id
                results.append(doc)
                
        return {"status": "success", "query": query, "type": type, "count": len(results), "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if client_mongo:
            client_mongo.close()


# ═══════════════════════════════════════════════════════════
# AGENTIC KERNEL — Live Performance Benchmarking
# Proves the core Agentic Kernel thesis: pure-math physics
# evaluation runs in microseconds, not the 100ms+ of Docker/VM
# sandboxes. These endpoints let the frontend visualize this
# in real time.
# ═══════════════════════════════════════════════════════════

@app.get("/kernel/benchmark")
def kernel_benchmark():
    """
    Run a live benchmark of the Agentic Kernel physics engine.
    Evaluates N trajectories and returns precise timing data
    to prove microsecond-latency evaluation.
    """
    import time
    import random as _rnd
    from environment.physics_model import (
        compute_training_cost,
        compute_hardware_safety,
        score_precision_layer,
    )

    NUM_TRAJECTORIES = 1000
    precisions = ["FP32", "BF16", "FP16", "FP8"]
    layers = ["embedding", "attention", "ffn", "layernorm", "output"]
    layer_dist = {
        "embedding": 0.05, "attention": 0.35, "ffn": 0.45,
        "layernorm": 0.05, "output": 0.10,
    }

    # --- Phase 1: Benchmark compute_training_cost ---
    start = time.perf_counter_ns()
    for _ in range(NUM_TRAJECTORIES):
        strategy = {l: _rnd.choice(precisions) for l in layers}
        params = _rnd.choice([3_000_000_000, 7_000_000_000, 13_000_000_000, 70_000_000_000])
        compute_training_cost(
            total_params=params,
            precision_strategy=strategy,
            layer_distribution=layer_dist,
        )
    cost_ns = time.perf_counter_ns() - start

    # --- Phase 2: Benchmark hardware safety checks ---
    start = time.perf_counter_ns()
    for _ in range(NUM_TRAJECTORIES):
        strategy = {l: _rnd.choice(precisions) for l in layers}
        params = _rnd.choice([3_000_000_000, 7_000_000_000, 13_000_000_000])
        compute_hardware_safety(
            total_params=params,
            precision_strategy=strategy,
            layer_distribution=layer_dist,
            num_gpus=_rnd.choice([1, 2, 4, 8]),
            gpu_memory_gb=80.0,
        )
    safety_ns = time.perf_counter_ns() - start

    # --- Phase 3: Benchmark per-layer scoring ---
    start = time.perf_counter_ns()
    for _ in range(NUM_TRAJECTORIES):
        for l in layers:
            score_precision_layer(l, _rnd.choice(precisions))
    scoring_ns = time.perf_counter_ns() - start

    # --- Phase 4: Full environment reset+step cycle ---
    from environment.fleet_env import FleetEnvironment
    bench_env = FleetEnvironment()
    start = time.perf_counter_ns()
    cycle_count = 200
    for _ in range(cycle_count):
        bench_env.reset("fleet_precision")
        bench_env.step({
            "precision_strategy": {
                "embedding": "FP32", "attention": "BF16", "ffn": "FP8",
                "layernorm": "BF16", "output": "FP32"
            },
            "reasoning": "benchmark"
        })
    env_ns = time.perf_counter_ns() - start

    total_ns = cost_ns + safety_ns + scoring_ns
    per_trajectory_us = (total_ns / NUM_TRAJECTORIES) / 1000

    # Docker/VM comparison (published industry data)
    docker_cold_start_ms = 150  # Typical Docker container cold start
    microvm_cold_start_ms = 125  # Firecracker MicroVM

    return {
        "kernel_version": "1.0.0",
        "trajectories_evaluated": NUM_TRAJECTORIES,
        "timing": {
            "cost_model_total_us": round(cost_ns / 1000, 1),
            "cost_model_per_eval_us": round(cost_ns / NUM_TRAJECTORIES / 1000, 2),
            "safety_check_total_us": round(safety_ns / 1000, 1),
            "safety_check_per_eval_us": round(safety_ns / NUM_TRAJECTORIES / 1000, 2),
            "layer_scoring_total_us": round(scoring_ns / 1000, 1),
            "layer_scoring_per_eval_us": round(scoring_ns / NUM_TRAJECTORIES / 1000, 2),
            "full_env_cycle_per_eval_us": round(env_ns / cycle_count / 1000, 2),
            "total_per_trajectory_us": round(per_trajectory_us, 2),
        },
        "throughput": {
            "trajectories_per_second": round(1_000_000 / per_trajectory_us) if per_trajectory_us > 0 else 999999,
            "speedup_vs_docker": round(docker_cold_start_ms * 1000 / per_trajectory_us, 0) if per_trajectory_us > 0 else 999999,
            "speedup_vs_microvm": round(microvm_cold_start_ms * 1000 / per_trajectory_us, 0) if per_trajectory_us > 0 else 999999,
        },
        "comparison": {
            "agentic_kernel_us": round(per_trajectory_us, 2),
            "docker_container_ms": docker_cold_start_ms,
            "firecracker_microvm_ms": microvm_cold_start_ms,
            "sandbox_ratio_solved": per_trajectory_us < 1000,
        },
    }


@app.get("/kernel/profile")
def kernel_profile():
    """
    Profile each kernel module (physics sub-system) individually.
    Returns per-module timing and a breakdown of what the kernel evaluates.
    """
    import time
    from environment.physics_model import (
        BYTES_PER_PARAM, STABILITY_SCORE, THROUGHPUT_MULTIPLIER,
        ACCURACY_PENALTY, POWER_DRAW_MULTIPLIER,
        compute_training_cost, compute_hardware_safety, score_precision_layer,
        compute_network_topology,
    )

    layers = ["embedding", "attention", "ffn", "layernorm", "output"]
    layer_dist = {
        "embedding": 0.05, "attention": 0.35, "ffn": 0.45,
        "layernorm": 0.05, "output": 0.10,
    }
    strategy = {
        "embedding": "FP32", "attention": "BF16", "ffn": "FP8",
        "layernorm": "BF16", "output": "FP32",
    }
    params = 7_000_000_000  # 7B model

    # Profile each function
    modules = {}

    start = time.perf_counter_ns()
    result = compute_training_cost(params, strategy, layer_dist)
    modules["precision_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "VRAM, throughput, cost, stability evaluation",
        "outputs": {
            "memory_gb": result["memory_gb"],
            "speedup_vs_fp32": result["speedup_vs_fp32"],
            "cost_usd": result["cost_usd"],
            "estimated_stable": result["estimated_stable"],
            "accuracy_retention": result["accuracy_retention"],
        }
    }

    start = time.perf_counter_ns()
    hw = compute_hardware_safety(params, strategy, layer_dist, num_gpus=4, gpu_memory_gb=80.0)
    modules["safety_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "Thermal, power, memory utilization checks",
        "outputs": {
            "memory_per_gpu_gb": hw["memory_per_gpu_gb"],
            "memory_utilization_pct": hw["memory_utilization_pct"],
            "thermal_risk": hw["thermal_risk"],
            "overall_safe": hw["overall_safe"],
        }
    }

    start = time.perf_counter_ns()
    layer_results = {}
    for l in layers:
        score, feedback = score_precision_layer(l, strategy.get(l, "FP32"))
        layer_results[l] = {"score": score, "precision": strategy.get(l, "FP32")}
    modules["scoring_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "Per-layer precision quality scoring",
        "outputs": layer_results,
    }

    start = time.perf_counter_ns()
    net = compute_network_topology("PCIe_Gen4", tensor_parallel_size=8, pipeline_parallel_size=1)
    modules["network_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "Multi-GPU topology bottleneck analysis",
        "outputs": {
            "topology": net["topology"],
            "effective_throughput_pct": net["effective_throughput_pct"],
            "bottleneck_risk": net["bottleneck_risk"],
        }
    }

    total_latency = sum(m["latency_us"] for m in modules.values())

    return {
        "profile_model": "LLaMA-3-7B",
        "profile_strategy": strategy,
        "modules": modules,
        "total_kernel_latency_us": round(total_latency, 2),
        "physics_constants_loaded": {
            "precision_formats": len(BYTES_PER_PARAM),
            "layer_types": len(STABILITY_SCORE),
            "stability_rules": sum(len(v) for v in STABILITY_SCORE.values()),
            "throughput_rules": sum(len(v) for v in THROUGHPUT_MULTIPLIER.values()),
            "source": "NVIDIA Transformer Engine + Micikevicius et al. 2018/2022",
        }
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ═══════════════════════════════════════════════════════════════════════════
# GOOGLE ADK AGENT ENDPOINTS
# Demonstrates Google Cloud Agent Builder + MongoDB MCP integration
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/adk/status")
def adk_status():
    """
    Check whether the Google ADK agent and MongoDB MCP server are available.
    """
    return {
        "adk_available": ADK_AVAILABLE,
        "mongodb_mcp": "enabled" if ADK_AVAILABLE else "requires google-adk and mcp packages",
        "agent_model": os.getenv("ADK_MODEL", "gemini-2.0-flash"),
        "install_command": "pip install google-adk mcp" if not ADK_AVAILABLE else None,
        "description": "Google Cloud Agent Builder (ADK) + MongoDB MCP Server integration",
    }


@app.post("/adk/run")
async def run_adk_agent(payload: dict = {}):
    """
    Run the Libratio Fleet Agent using Google ADK + MongoDB MCP Server.

    This is the core hackathon demo endpoint:
    - Uses Google ADK (Agent Development Kit) as the orchestration framework
    - Gemini 2.0 Flash as the reasoning model
    - MongoDB MCP Server as the partner superpower (tool calls via Model Context Protocol)
    - Fleet physics tools for environment interaction

    Body: {"task_id": "fleet_precision"}
    Valid task_ids: fleet_precision, fleet_oversight, fleet_resource, fleet_recovery
    """
    if not ADK_AVAILABLE:
        return {
            "error": "Google ADK not installed",
            "fix": "pip install google-adk mcp",
            "adk_available": False,
        }

    task_id = payload.get("task_id", "fleet_precision")
    valid_tasks = ["fleet_precision", "fleet_oversight", "fleet_resource", "fleet_recovery"]

    if task_id not in valid_tasks:
        return {"error": f"Invalid task_id. Choose from: {valid_tasks}"}

    try:
        import asyncio
        result = await run_fleet_task_with_adk(task_id)
        return {
            "status": "completed",
            "agent": "LibratioFleetCommander",
            "framework": "Google ADK (Agent Development Kit)",
            "partner_integration": "MongoDB MCP Server (@mongodb-js/mongodb-mcp-server)",
            **result,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "task_id": task_id,
        }


if __name__ == "__main__":
    main()

