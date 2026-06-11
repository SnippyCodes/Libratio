"""
Libratio Predictive MLOps Agent
================================
Uses MongoDB Atlas Vector Search + LLM reasoning to proactively
prevent GPU cluster crashes by recommending precision strategy changes.

Retrieval pipeline (RAG):
  1. Live cluster state → Gemini text-embedding-004 → 768-dim query vector
  2. MongoDB Atlas $vectorSearch → top-5 semantically similar past crashes
  3. Historical context + live state → LLM reasoning → preventive action

This is true semantic RAG: instead of exact field matching, we find
historically similar situations by meaning — even if the exact numbers differ.

Fallback chain:
  Atlas Vector Search → basic $match aggregation → local JSON file
"""
import os
import json
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

# Vector Search module (Gemini embeddings + Atlas $vectorSearch)
try:
    from mongodb_vector import (
        embed_query,
        vector_search_similar_trajectories,
        format_vector_results_as_context,
        hybrid_search_similar_trajectories,
        format_hybrid_results_as_context,
        _GEMINI_AVAILABLE,
    )
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    _GEMINI_AVAILABLE = False
    print("[WARN] mongodb_vector module not found. Using basic $match fallback.")

load_dotenv()

# ── MongoDB Setup (with local fallback) ──
MONGO_URI = os.getenv("MONGO_URI")
USE_LOCAL_FALLBACK = False
local_data = None

try:
    import certifi
    mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    db = mongo_client["libratio"]
    trajectories_col = db["trajectories"]
    # Quick connectivity check
    mongo_client.admin.command("ping")
    print("[OK] Connected to MongoDB Atlas")
except Exception as e:
    print(f"[WARN] MongoDB unreachable ({type(e).__name__}). Using local JSON fallback.")
    USE_LOCAL_FALLBACK = True
    with open("synthetic_trajectories.json", "r") as f:
        local_data = json.load(f)

# ── LLM Setup ──
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_MODEL_URL = os.getenv("HF_MODEL_URL", "https://router.huggingface.co/v1/chat/completions")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "SnippyCodes/libratio-fleet-llama3-grpo:featherless-ai")
def retrieve_historical_context(live_state: dict) -> str:
    """
    RAG Retrieval Step: Find semantically similar historical trajectories.

    Retrieval hierarchy:
    1. MongoDB Atlas Vector Search ($vectorSearch) — semantic similarity via Gemini embeddings
       "Find past situations that FEEL like this one" — best results
    2. MongoDB basic aggregation ($match) — exact field matching fallback
       "Find past situations with exactly these field values"
    3. Local JSON file — offline fallback when MongoDB is unreachable

    Args:
        live_state: Dict with current cluster state (model_name, memory_used_gb,
                    cluster_capacity_gb, thermal_risk, power_util, etc.)
    Returns:
        Formatted string of historical context for the LLM prompt.
    """
    thermal_risk = live_state.get("thermal_risk", "HIGH")
    memory_used = live_state.get("memory_used_gb", 0)
    memory_gb_threshold = memory_used * 0.8

    print(f"[*] RAG SEARCH: Finding semantically similar past crashes...")
    print(f"    Query: {live_state.get('model_name', '?')} | "
          f"{memory_used}GB VRAM | thermal={thermal_risk}")

    # ── Path 1: Atlas Hybrid Search (Vector + Full-Text Search with RRF) (best) ──
    if VECTOR_SEARCH_AVAILABLE and not USE_LOCAL_FALLBACK:
        try:
            model_name = live_state.get("model_name", "")
            text_query = f"{model_name} {thermal_risk} thermal crash"
            results = hybrid_search_similar_trajectories(
                collection=trajectories_col,
                query_text=text_query,
                live_state=live_state,
                top_k=5,
            )
            if results:
                return format_hybrid_results_as_context(results)
            else:
                print("    [WARN] Hybrid search returned no results. Falling back to vector search.")
                results = vector_search_similar_trajectories(
                    collection=trajectories_col,
                    live_state=live_state,
                    top_k=5,
                )
                if results:
                    return format_vector_results_as_context(results)
        except Exception as e:
            print(f"    [WARN] Hybrid search error ({type(e).__name__}): {e}")
            print("    Falling back to basic $match query...")

    # ── Path 2: Basic MongoDB $match (fallback) ──
    if not USE_LOCAL_FALLBACK:
        try:
            print("    (Source: MongoDB Atlas — basic $match aggregation)")
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"thermal_risk": thermal_risk},
                            {"memory_used_gb": {"$gt": memory_gb_threshold}},
                            {"outcome": {"$ne": "SUCCESS"}}
                        ]
                    }
                },
                {"$sort": {"memory_used_gb": -1}},
                {"$limit": 5},
                {"$project": {"embedding": 0, "_id": 0}},
            ]
            results = list(trajectories_col.aggregate(pipeline))
            if results:
                context_lines = ["HISTORICAL CRASH DATA FROM DATABASE ($match fallback):"]
                for t in results:
                    line = (
                        f"- Model: {t.get('model_name', t.get('model_id', '?'))} "
                        f"({t.get('total_params_b', '?')}B params) | "
                        f"Outcome: {t['outcome']} | "
                        f"Strategy: {t['precision_strategy']} | "
                        f"VRAM: {t['memory_used_gb']}GB | "
                        f"Thermal: {t['thermal_risk']} | "
                        f"Power: {t.get('estimated_power_pct', '?')}%"
                    )
                    if t.get("failure_reason"):
                        line += f" | Cause: {t['failure_reason']}"
                    context_lines.append(line)
                return "\n".join(context_lines)
        except Exception as e:
            print(f"    [WARN] $match query failed ({type(e).__name__}): {e}")

    # ── Path 3: Local JSON fallback ──
    print("    (Source: local synthetic_trajectories.json — offline fallback)")
    results = []
    for episode in (local_data or []):
        for t in episode.get("trajectories", []):
            if (t.get("thermal_risk") == thermal_risk or
                t.get("memory_used_gb", 0) > memory_gb_threshold or
                    t.get("outcome") != "SUCCESS"):
                results.append(t)
                if len(results) >= 5:
                    break
        if len(results) >= 5:
            break

    if not results:
        return "No similar historical trajectories found in database."

    context_lines = ["HISTORICAL CRASH DATA (local JSON fallback):"]
    for t in results:
        line = (
            f"- Model: {t.get('model_name', t.get('model_id', '?'))} "
            f"({t.get('total_params_b', '?')}B params) | "
            f"Outcome: {t.get('outcome', 'UNKNOWN')} | "
            f"Strategy: {t.get('precision_strategy', {})} | "
            f"VRAM: {t.get('memory_used_gb', '?')}GB | "
            f"Thermal: {t.get('thermal_risk', '?')} | "
            f"Power: {t.get('estimated_power_pct', '?')}%"
        )
        if t.get("failure_reason"):
            line += f" | Cause: {t['failure_reason']}"
        context_lines.append(line)
    return "\n".join(context_lines)

def call_llm(prompt):
    """
    Call the custom HuggingFace model first. 
    If HF API is blocked or model is loading, fallback to Groq.
    """
    system_prompt = "You are the Libratio Fleet Commander, an expert MLOps SRE agent. You prevent GPU cluster crashes by analyzing historical failure data and recommending optimal mixed-precision strategies. Always output valid JSON."
    
    # Attempt 1: Hugging Face Custom Fine-Tuned Model
    print(f"    [Routing inference to custom model: {HF_MODEL_NAME}]")
    try:
        hf_headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        hf_payload = {
            "model": HF_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        hf_res = requests.post(HF_MODEL_URL, headers=hf_headers, json=hf_payload, timeout=30)
        hf_res.raise_for_status()
        result = hf_res.json()
        if "choices" in result and len(result["choices"]) > 0:
            print("    [SUCCESS: Inference powered by Hugging Face Custom Model]")
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"    [WARN] Hugging Face API unavailable or warming up ({type(e).__name__}). Falling back to Groq.")

    # Attempt 2: Fallback to Groq (llama-3.1-8b base)
    groq_headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    groq_payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }

    response = requests.post(
        f"{GROQ_API_BASE}/chat/completions",
        headers=groq_headers,
        json=groq_payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def predict_and_recover(live_state):
    """
    The Predictive MLOps Agent.
    1. Retrieves semantically similar historical crashes from MongoDB Atlas
       using Vector Search (Gemini embeddings → $vectorSearch)
    2. Sends the retrieved context + live state to LLM for reasoning
    3. Outputs a preventive precision strategy
    """
    retrieval_mode = (
        "Atlas Vector Search (Gemini embeddings)" if (VECTOR_SEARCH_AVAILABLE and _GEMINI_AVAILABLE and not USE_LOCAL_FALLBACK)
        else "MongoDB $match" if not USE_LOCAL_FALLBACK
        else "Local JSON Fallback"
    )

    print("\n" + "=" * 60)
    print("  LIBRATIO PREDICTIVE MLOPS AGENT")
    print(f"  Retrieval: {retrieval_mode}")
    print("=" * 60)

    # ── Step 1: RAG Retrieval from MongoDB (Vector Search) ──
    historical_context = retrieve_historical_context(live_state)

    print("\n[STEP 1] RAG CONTEXT FROM MONGODB:")
    print("-" * 40)
    print(historical_context)

    # ── Step 2: Build prompt with live state + history ──
    prompt = f"""
CURRENT LIVE CLUSTER STATE (DANGER):
- Model: {live_state['model_name']}
- Parameters: {live_state.get('total_params_b', '?')}B
- Current Memory Used: {live_state['memory_used_gb']} GB
- Cluster Total Memory: {live_state['cluster_capacity_gb']} GB
- Memory Utilization: {round(100 * live_state['memory_used_gb'] / live_state['cluster_capacity_gb'], 1)}%
- Thermal Risk: {live_state['thermal_risk']}
- Power Utilization: {live_state['power_util']}%

{historical_context}

Based on the historical crash data above, what precision strategy should we apply to this model RIGHT NOW to prevent a crash?

Rules:
- Available layers: embedding, attention, ffn, layernorm, output
- Available precisions: FP32, BF16, FP16, FP8
- Embedding and output layers should NEVER use FP8 (causes numerical instability)
- FFN layers are the safest target for FP8 optimization
- The goal is to REDUCE memory and thermal pressure while maintaining training stability

Respond with a JSON object containing:
1. "precision_strategy": the recommended precision per layer
2. "reasoning": a short explanation citing the historical data
3. "expected_outcome": what you predict will happen after this change
"""

    print("\n[STEP 2] AGENT REASONING (via LLM)...")
    print("-" * 40)

    try:
        response = call_llm(prompt)
        print("\n[STEP 3] AGENT DECISION:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        return response
    except Exception as e:
        print(f"\n[ERROR] LLM call failed: {e}")
        return None


if __name__ == "__main__":
    # Simulate a live cluster state that is heading toward a crash
    live_danger_state = {
        "model_name": "LLaMA-3-70B",
        "total_params_b": 70.0,
        "memory_used_gb": 220.0,
        "cluster_capacity_gb": 320.0,
        "thermal_risk": "HIGH",
        "power_util": 88.0,
    }

    predict_and_recover(live_danger_state)
