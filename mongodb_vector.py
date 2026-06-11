"""
MongoDB Atlas Vector Search — Libratio Fleet
=============================================
Upgrades the basic $match RAG retrieval into true semantic search.

How it works:
1. Each GPU trajectory record is converted to a descriptive text string
2. Google Gemini generates a 768-dim embedding vector for that text
3. The vector is stored alongside the document in MongoDB
4. At query time, the live cluster state is embedded the same way
5. Atlas $vectorSearch finds the top-k most semantically similar past crashes

Why Gemini embeddings?
- We already have GEMINI_API_KEY in .env
- 768-dim vectors are compact (good for Atlas free tier)
- Gemini Embedding-001 is specifically designed for semantic retrieval tasks

Why this beats $match:
- $match finds: thermal_risk == "HIGH" AND memory_used_gb > 180 (exact)
- $vectorSearch finds: "situations semantically similar to high thermal stress
  with memory pressure, regardless of exact field values"
"""

import os
import json
from typing import Optional
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

# ── Gemini Embedding Setup ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"  # 768-dim, supports output_dimensionality
EMBEDDING_DIM = 768

# Fallback if Gemini is unavailable — use a simple deterministic hash embedding
_GEMINI_AVAILABLE = False
_gemini_client = None

if GEMINI_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        _GEMINI_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] Gemini embedding setup failed: {e}. Using fallback embedder.")
else:
    print("[WARN] GEMINI_API_KEY not set. Using fallback embedder (no semantic search).")


# ──────────────────────────────────────────────────────────────────────────────
# TEXT SERIALIZATION
# Convert a trajectory record or live cluster state into a rich text description
# that captures the semantic meaning of the GPU cluster situation.
# ──────────────────────────────────────────────────────────────────────────────

def trajectory_to_text(traj: dict) -> str:
    """
    Convert a stored trajectory record into a descriptive text for embedding.

    The text intentionally uses natural language so Gemini's semantic embedding
    captures the *meaning* of the situation — not just the field names.

    Example output:
    "Model LLaMA-3-70B (70.0B params) training outcome: CRASH_NUMERICAL_INSTABILITY.
     Precision strategy: embedding=FP8, attention=BF16, ffn=FP8, layernorm=FP8, output=FP32.
     Memory usage: 185.4GB VRAM, 92% utilization. Thermal status: HIGH risk, 94% power draw.
     Speedup vs FP32: 2.1x. Failure cause: embedding at FP8 (stability=0.1)."
    """
    strategy = traj.get("precision_strategy", {})
    strategy_str = ", ".join(f"{k}={v}" for k, v in strategy.items()) if strategy else "unknown"

    outcome = traj.get("outcome", "UNKNOWN")
    failure = traj.get("failure_reason", "")
    failure_str = f" Failure cause: {failure}." if failure else ""

    text = (
        f"Model {traj.get('model_name', traj.get('model_id', 'unknown'))} "
        f"({traj.get('total_params_b', '?')}B params) training outcome: {outcome}. "
        f"Precision strategy: {strategy_str}. "
        f"Memory usage: {traj.get('memory_used_gb', '?')}GB VRAM, "
        f"{traj.get('memory_utilization_pct', '?')}% utilization. "
        f"Thermal status: {traj.get('thermal_risk', 'UNKNOWN')} risk, "
        f"{traj.get('estimated_power_pct', '?')}% power draw. "
        f"Speedup vs FP32: {traj.get('speedup_vs_fp32', '?')}x. "
        f"Accuracy retention: {traj.get('accuracy_retention', '?')}%."
        f"{failure_str}"
    )
    return text


def live_state_to_query_text(live_state: dict) -> str:
    """
    Convert the current live cluster state into a query text for vector search.

    This should describe the DANGER/RISK we're currently experiencing so that
    Vector Search finds historically similar dangerous situations.

    Example output:
    "Current cluster EMERGENCY: Model LLaMA-3-70B (70.0B params) using 220.0GB of
     320.0GB cluster memory (68.8% utilization). Thermal risk: HIGH. Power draw: 88%.
     Need to find similar historical crashes to inform precision strategy decisions."
    """
    memory_pct = round(
        100 * live_state.get("memory_used_gb", 0) / max(live_state.get("cluster_capacity_gb", 1), 1),
        1
    )
    text = (
        f"Current cluster EMERGENCY: "
        f"Model {live_state.get('model_name', 'unknown')} "
        f"({live_state.get('total_params_b', '?')}B params) "
        f"using {live_state.get('memory_used_gb', '?')}GB of "
        f"{live_state.get('cluster_capacity_gb', '?')}GB cluster memory "
        f"({memory_pct}% utilization). "
        f"Thermal risk: {live_state.get('thermal_risk', 'UNKNOWN')}. "
        f"Power draw: {live_state.get('power_util', '?')}%. "
        f"Need to find similar historical crashes to inform precision strategy decisions."
    )
    return text


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_embedding(text: str) -> list[float]:
    """
    Deterministic fallback when Gemini is unavailable.
    Uses a simple character-based hash to produce a 768-dim vector.
    NOT semantically meaningful, but allows the pipeline to run without Gemini.
    """
    import hashlib
    import math

    # Generate a seeded pseudo-random vector from the text hash
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng_state = seed
    result = []
    for i in range(EMBEDDING_DIM):
        # Linear congruential generator — fast, deterministic, reproducible
        rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
        # Map to [-1, 1] and apply sine for spread
        val = math.sin(rng_state / 0xFFFFFFFF * math.pi * 2 + i)
        result.append(round(val, 6))

    # Normalize to unit vector (required for cosine similarity)
    magnitude = math.sqrt(sum(v * v for v in result))
    if magnitude > 0:
        result = [v / magnitude for v in result]
    return result


def generate_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Generate a 768-dimensional embedding vector for the given text.

    Args:
        text: The text to embed (trajectory description or query)
        task_type: Gemini task type — use "RETRIEVAL_DOCUMENT" when indexing,
                   "RETRIEVAL_QUERY" when searching. This optimizes the embedding
                   for asymmetric retrieval (short query vs longer documents).

    Returns:
        List of 768 floats representing the text in semantic space.
    """
    if not _GEMINI_AVAILABLE or _gemini_client is None:
        return _fallback_embedding(text)

    try:
        result = _gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,   # Pin to 768 dims explicitly
            ),
        )
        return list(result.embeddings[0].values)
    except Exception as e:
        print(f"  [WARN] Gemini embedding failed ({type(e).__name__}), using fallback: {e}")
        return _fallback_embedding(text)


def embed_trajectory(traj: dict) -> list[float]:
    """Generate an embedding vector for a stored trajectory document."""
    text = trajectory_to_text(traj)
    return generate_embedding(text, task_type="RETRIEVAL_DOCUMENT")


def embed_query(live_state: dict) -> list[float]:
    """Generate an embedding vector for a live cluster state query."""
    text = live_state_to_query_text(live_state)
    return generate_embedding(text, task_type="RETRIEVAL_QUERY")


# ──────────────────────────────────────────────────────────────────────────────
# VECTOR SEARCH QUERY
# ──────────────────────────────────────────────────────────────────────────────

def vector_search_similar_trajectories(
    collection,
    live_state: dict,
    top_k: int = 5,
    filter_outcome: Optional[str] = None,
) -> list[dict]:
    """
    Find the most semantically similar historical trajectories to the current
    live cluster state using MongoDB Atlas $vectorSearch.

    The Atlas Vector Search index must exist (run setup_vector_index.py first).
    Index name: "trajectory_vector_index"
    Field: "embedding"
    Similarity: "cosine"

    Args:
        collection: PyMongo collection object (trajectories collection)
        live_state: Current live cluster state dict
        top_k: Number of similar trajectories to return
        filter_outcome: Optional — filter results to only CRASH outcomes

    Returns:
        List of trajectory dicts (without the embedding field for readability)

    How Atlas $vectorSearch works:
        1. Your query vector is compared against all stored document vectors
        2. Atlas uses Hierarchical Navigable Small Worlds (HNSW) graph — an
           approximate nearest-neighbor algorithm that runs in O(log n) time
        3. Results are ranked by cosine similarity score (1.0 = identical, 0.0 = unrelated)
        4. Only the top_k most similar documents are returned
    """
    # Embed the live cluster state
    print("    (Source: MongoDB Atlas Vector Search — semantic similarity)")
    query_vector = embed_query(live_state)

    # Build the $vectorSearch pipeline
    # numCandidates: how many candidates to consider before ranking (must be >= top_k)
    # More candidates = more accurate but slower. 10x top_k is a good balance.
    pipeline = [
        {
            "$vectorSearch": {
                "index": "trajectory_vector_index",   # Index name (created by setup_vector_index.py)
                "path": "embedding",                   # Field storing the vector
                "queryVector": query_vector,           # Our query embedding
                "numCandidates": top_k * 10,           # Candidate pool for ranking
                "limit": top_k,                        # How many to return
            }
        },
        # Add the similarity score to each result
        {
            "$addFields": {
                "vector_score": {"$meta": "vectorSearchScore"}
            }
        },
        # Optionally filter to only crash outcomes
        *(
            [{"$match": {"outcome": {"$ne": "SUCCESS"}}}]
            if filter_outcome == "crashes_only"
            else []
        ),
        # Remove the embedding vector from results (it's huge and unreadable)
        {
            "$project": {
                "embedding": 0,
                "_id": 0,
            }
        }
    ]

    try:
        results = list(collection.aggregate(pipeline))
        if results:
            print(f"    Found {len(results)} semantically similar trajectories "
                  f"(top score: {results[0].get('vector_score', 'N/A'):.3f})")
        return results
    except Exception as e:
        # If vector search index doesn't exist yet, fall back to basic search
        print(f"    [WARN] Vector search failed ({type(e).__name__}): {e}")
        print("    [HINT] Run setup_vector_index.py to create the Atlas Vector Search index.")
        return []


def format_vector_results_as_context(results: list[dict]) -> str:
    """
    Format vector search results into a rich context string for the LLM prompt.
    Includes the semantic similarity score so the LLM knows how relevant each
    historical record is.
    """
    if not results:
        return "No semantically similar historical trajectories found."

    lines = ["HISTORICAL CRASH DATA (MongoDB Atlas Vector Search — ranked by semantic similarity):"]
    for i, t in enumerate(results, 1):
        score = t.get("vector_score", 0)
        relevance = "HIGH" if score > 0.85 else "MEDIUM" if score > 0.70 else "LOW"

        strategy = t.get("precision_strategy", {})
        strategy_str = ", ".join(f"{k}={v}" for k, v in strategy.items()) if strategy else "unknown"

        line = (
            f"\n[{i}] Similarity: {score:.3f} ({relevance} relevance)\n"
            f"    Model: {t.get('model_name', t.get('model_id', '?'))} "
            f"({t.get('total_params_b', '?')}B params)\n"
            f"    Outcome: {t.get('outcome', 'UNKNOWN')} | "
            f"VRAM: {t.get('memory_used_gb', '?')}GB | "
            f"Thermal: {t.get('thermal_risk', '?')} | "
            f"Power: {t.get('estimated_power_pct', '?')}%\n"
            f"    Strategy: {strategy_str}"
        )
        if t.get("failure_reason"):
            line += f"\n    Failure: {t['failure_reason']}"

        lines.append(line)

    return "\n".join(lines)


def text_search_similar_trajectories(
    collection,
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Find historical trajectories matching a text query using MongoDB Atlas Search (Lucene full-text search).
    
    The Atlas Search index must exist (run setup_vector_index.py first).
    Index name: "trajectory_text_search_index"
    """
    print("    (Source: MongoDB Atlas Search — full-text keyword search)")
    pipeline = [
        {
            "$search": {
                "index": "trajectory_text_search_index",
                "text": {
                    "query": query_text,
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
            "$limit": top_k
        },
        {
            "$project": {
                "embedding": 0
            }
        }
    ]
    try:
        results = list(collection.aggregate(pipeline))
        for res in results:
            res["_id"] = str(res["_id"])
        if results:
            print(f"    Found {len(results)} keyword matches (top score: {results[0].get('search_score', 'N/A'):.3f})")
        return results
    except Exception as e:
        print(f"    [WARN] Full-text search failed ({type(e).__name__}): {e}")
        return []


def hybrid_search_similar_trajectories(
    collection,
    query_text: str,
    live_state: dict,
    top_k: int = 5,
) -> list[dict]:
    """
    Combines Vector Search (Gemini embeddings) and Full-Text Search (keyword matching)
    using Reciprocal Rank Fusion (RRF) to produce a combined relevance score.
    
    RRF Score(d) = sum_{m in M} 1 / (60 + rank_m(d))
    """
    # 1. Run Vector Search (get more than top_k so we have a good candidate pool for ranking)
    candidate_limit = max(top_k * 4, 20)
    
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "trajectory_vector_index",
                "path": "embedding",
                "queryVector": embed_query(live_state),
                "numCandidates": candidate_limit * 10,
                "limit": candidate_limit,
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
    
    # 2. Run Text Search
    text_pipeline = [
        {
            "$search": {
                "index": "trajectory_text_search_index",
                "text": {
                    "query": query_text,
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
            "$limit": candidate_limit
        },
        {
            "$project": {
                "embedding": 0
            }
        }
    ]
    
    vector_results = []
    text_results = []
    
    try:
        vector_results = list(collection.aggregate(vector_pipeline))
    except Exception as e:
        print(f"    [WARN] Hybrid: Vector search step failed: {e}")
        
    try:
        text_results = list(collection.aggregate(text_pipeline))
    except Exception as e:
        print(f"    [WARN] Hybrid: Text search step failed: {e}")
        
    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Constant
    K = 60
    rrf_scores = {}
    doc_map = {}
    
    # Vector ranking
    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (K + rank + 1))
        doc["vector_rank"] = rank + 1
        
    # Text ranking
    for rank, doc in enumerate(text_results):
        doc_id = str(doc["_id"])
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        else:
            if "search_score" in doc:
                doc_map[doc_id]["search_score"] = doc["search_score"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (K + rank + 1))
        doc_map[doc_id]["text_rank"] = rank + 1
        
    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    hybrid_results = []
    for doc_id in sorted_ids[:top_k]:
        doc = doc_map[doc_id]
        doc["rrf_score"] = rrf_scores[doc_id]
        doc["_id"] = doc_id
        hybrid_results.append(doc)
        
    print(f"    Hybrid Search (RRF) merged {len(vector_results)} vector & {len(text_results)} text results -> returned top {len(hybrid_results)}")
    return hybrid_results


def format_hybrid_results_as_context(results: list[dict]) -> str:
    """
    Format hybrid RRF search results into a rich context string for the LLM prompt.
    """
    if not results:
        return "No similar historical trajectories found via Hybrid Search."

    lines = ["HISTORICAL CRASH DATA (MongoDB Atlas Hybrid Search — Vector + Full-Text RRF Fusion):"]
    for i, t in enumerate(results, 1):
        rrf = t.get("rrf_score", 0.0)
        v_rank = t.get("vector_rank", "N/A")
        t_rank = t.get("text_rank", "N/A")
        
        strategy = t.get("precision_strategy", {})
        strategy_str = ", ".join(f"{k}={v}" for k, v in strategy.items()) if strategy else "unknown"

        line = (
            f"\n[{i}] RRF Score: {rrf:.4f} (Vector Rank: {v_rank}, Text Rank: {t_rank})\n"
            f"    Model: {t.get('model_name', t.get('model_id', '?'))} "
            f"({t.get('total_params_b', '?')}B params)\n"
            f"    Outcome: {t.get('outcome', 'UNKNOWN')} | "
            f"VRAM: {t.get('memory_used_gb', '?')}GB | "
            f"Thermal: {t.get('thermal_risk', '?')} | "
            f"Power: {t.get('estimated_power_pct', '?')}%\n"
            f"    Strategy: {strategy_str}"
        )
        if t.get("failure_reason"):
            line += f"\n    Failure: {t['failure_reason']}"

        lines.append(line)

    return "\n".join(lines)
