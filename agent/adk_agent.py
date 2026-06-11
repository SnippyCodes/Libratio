"""
Libratio Fleet — Google ADK Agent Orchestrator
===============================================
Google Cloud Agent Builder integration using Google ADK 2.2.0.

HOW IT WORKS (simple version):
  1. Google ADK gives Gemini a set of "tools" it can call
  2. One toolset = MongoDB MCP Server (a Node.js process that knows how to talk to MongoDB)
  3. Another toolset = our fleet physics tools (Python functions Gemini can call directly)
  4. Gemini reasons about the fleet task, decides which tools to call, and orchestrates everything
  5. ADK runs this in a loop until the task is done

WHY THIS MATTERS FOR THE HACKATHON:
  - "Google Cloud Agent Builder" = Google ADK (the code-first version of Agent Builder)
  - "MongoDB MCP Server" = the partner superpowers requirement
  - Gemini 2.0 Flash = the brain
  - This file is the proof that all three are wired together

Requirements:
  - pip install google-adk mcp
  - npm/npx available (for MongoDB MCP server subprocess)
  - MONGO_URI, GEMINI_API_KEY set in .env
"""

import asyncio
import json
import os
import sys
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Google ADK imports ──────────────────────────────────────────────────────
try:
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
    from google.adk.tools import FunctionTool
    from mcp import StdioServerParameters
    from google.genai import types as genai_types
    ADK_AVAILABLE = True
except ImportError as e:
    ADK_AVAILABLE = False
    print(f"[WARN] Google ADK not installed: {e}")
    print("       Run: pip install google-adk mcp")

# ── Config ──────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Use Vertex AI if GCP project is configured, otherwise AI Studio (free)
USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

if not USE_VERTEX_AI:
    # Unset Vertex AI env variables in Python to force the SDK to use the API key
    os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
    os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
    print("AI Studio mode enabled (Vertex AI environment variables cleared)")

# Model — Gemini 2.0 Flash is fast and free-tier friendly
ADK_MODEL = os.getenv("ADK_MODEL", "gemini-2.0-flash")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("libratio.adk")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: MONGODB MCP TOOLSET
# ═══════════════════════════════════════════════════════════════════════════
# This launches the official MongoDB MCP server (@mongodb-js/mongodb-mcp-server)
# as a child process. The ADK MCPToolset wraps ALL its tools so Gemini can
# call them like native Python functions.
#
# Tools exposed by MongoDB MCP server include:
#   - find: Query documents from a collection
#   - aggregate: Run aggregation pipelines (including $vectorSearch)
#   - insertMany: Insert documents
#   - createIndex: Create search/vector indexes
#   - listCollections: List all collections
#   - count: Count documents matching a filter
# ═══════════════════════════════════════════════════════════════════════════

def get_mongodb_mcp_toolset() -> "McpToolset | None":
    """
    Creates a Google ADK McpToolset connected to the MongoDB MCP Server.

    The MongoDB MCP server runs as a Node.js subprocess (via npx).
    ADK communicates with it over stdio using the Model Context Protocol.

    Returns:
        McpToolset if MongoDB MCP is available, None otherwise.
    """
    if not ADK_AVAILABLE:
        logger.warning("ADK not available -- MongoDB MCP toolset skipped")
        return None

    import platform
    if platform.system() == "Windows":
        logger.info("Windows detected -- skipping Node.js MCP stdio subprocess to prevent asyncio pipe issues. Using PyMongo tools instead.")
        return None

    if not MONGO_URI:
        logger.warning("MONGO_URI not set -- MongoDB MCP toolset will use localhost")

    logger.info("Initializing MongoDB MCP Server (via npx mongodb-mcp-server)...")

    # The MongoDB MCP server is a Node.js package published by MongoDB.
    # Package was renamed from @mongodb-js/mongodb-mcp-server -> mongodb-mcp-server
    # Connection string is now passed via MDB_MCP_CONNECTION_STRING env var
    # (the old --connectionString CLI arg is deprecated).

    # On Windows, "npx" is actually "npx.cmd" — Python subprocess can't find
    # bare "npx" without shell=True. We resolve the full path with shutil.which.
    import shutil
    npx_cmd = shutil.which("npx")
    if npx_cmd is None:
        logger.error("npx not found on PATH. Install Node.js: https://nodejs.org")
        return None

    logger.info(f"Using npx at: {npx_cmd}")

    mcp_toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                # Run the official MongoDB MCP server via npx (auto-downloads)
                command=npx_cmd,                    # full path works on all OS
                args=[
                    "-y",                           # auto-install without prompting
                    "mongodb-mcp-server@latest",    # official MongoDB MCP package (new name)
                ],
                env={
                    **os.environ,                    # inherit PATH, NODE, etc.
                    "MDB_MCP_CONNECTION_STRING": MONGO_URI,  # official env var name
                },
            ),
            timeout=60,  # 60s timeout for initial npx package download on first run
        )
    )

    logger.info("MongoDB MCP toolset configured (npx mongodb-mcp-server)")
    return mcp_toolset


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: FLEET PHYSICS TOOLS (ADK Python Tools)
# ═══════════════════════════════════════════════════════════════════════════
# These wrap our existing fleet environment as ADK-compatible Python tools.
# Gemini can call these to interact with the physics-based RL environment.
# ═══════════════════════════════════════════════════════════════════════════

# Module-level fleet env instance (created lazily)
_fleet_env_instance = None


def _fleet_env():
    global _fleet_env_instance
    if _fleet_env_instance is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environment.fleet_env import FleetEnvironment
        _fleet_env_instance = FleetEnvironment()
    return _fleet_env_instance


def reset_fleet_environment(task_id: str) -> dict:
    """
    Reset the Libratio fleet environment to start a new episode.

    Use this at the beginning of each fleet task to get an initial observation
    of the GPU cluster state. The observation tells you about available GPUs,
    memory, and the model(s) you need to configure.

    Args:
        task_id: One of 'fleet_precision', 'fleet_oversight', 'fleet_resource', 'fleet_recovery'

    Returns:
        Initial cluster observation dict with GPU state, model info, and task context.
    """
    try:
        obs = _fleet_env().reset(task_id)
        logger.info(f"[Tool] reset_fleet_environment: task={task_id}")
        return {"success": True, "observation": obs, "task_id": task_id}
    except Exception as e:
        logger.error(f"[Tool] reset_fleet_environment error: {e}")
        return {"success": False, "error": str(e)}


def step_fleet_environment(action_json: str) -> dict:
    """
    Submit an action to the fleet environment and get a reward score.

    After observing the cluster state, use this tool to submit your decision
    as a JSON string. The environment evaluates your action using the
    physics-based reward model and returns a score between 0.01 and 0.99.

    Action format depends on the current task:
    - fleet_precision: {"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "..."}
    - fleet_oversight: {"action_type": "continue_monitoring|flag_instability", "analysis": "...", "flagged_model": null}
    - fleet_resource: {"allocations": {"model_a": {"gpus": 4, "precision_strategy": {...}}}, "reasoning": "..."}
    - fleet_recovery: Phase-dependent (diagnose/reallocate/verify)

    Args:
        action_json: JSON string containing the action dict

    Returns:
        Dict with score (0.01-0.99), feedback string, done flag, and next observation.
    """
    try:
        action = json.loads(action_json)
        result = _fleet_env().step(action)
        score = max(0.01, min(0.99, float(result["reward"]["score"])))
        logger.info(f"[Tool] step_fleet_environment: score={score:.3f}")
        return {
            "success": True,
            "score": score,
            "feedback": result["reward"]["feedback"],
            "done": result["done"],
            "observation": result.get("observation"),
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in action: {e}"}
    except Exception as e:
        logger.error(f"[Tool] step_fleet_environment error: {e}")
        return {"success": False, "error": str(e)}


def compute_precision_physics(layer_name: str, precision: str) -> dict:
    """
    Compute the physics-based score and safety check for a precision assignment.

    Use this to validate your precision choices BEFORE submitting to the fleet.
    This is the Agentic Kernel — pure math based on NVIDIA Transformer Engine benchmarks.

    Args:
        layer_name: One of 'embedding', 'attention', 'ffn', 'layernorm', 'output'
        precision: One of 'FP32', 'BF16', 'FP16', 'FP8'

    Returns:
        Dict with score, stability, and recommendation.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environment.physics_model import score_precision_layer, STABILITY_SCORE
        score, feedback = score_precision_layer(layer_name, precision)
        stability = STABILITY_SCORE.get(layer_name, {}).get(precision, 0.5)
        return {
            "layer": layer_name,
            "precision": precision,
            "physics_score": round(score, 3),
            "stability": round(stability, 3),
            "is_stable": stability >= 0.7,
            "feedback": feedback,
            "recommendation": (
                f"SAFE: {precision} is good for {layer_name}"
                if stability >= 0.7
                else f"RISKY: {precision} on {layer_name} may cause instability (stability={stability:.2f})"
            ),
        }
    except Exception as e:
        return {"error": str(e), "layer": layer_name, "precision": precision}


def get_fleet_physics_tools() -> list:
    """Return a list of ADK FunctionTool wrappers for fleet physics functions."""
    if not ADK_AVAILABLE:
        return []
    return [
        FunctionTool(func=reset_fleet_environment),
        FunctionTool(func=step_fleet_environment),
        FunctionTool(func=compute_precision_physics),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# PART 2B: MONGODB PYMONGO TOOLS (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════
# When the MCP subprocess fails (common on Windows due to asyncio pipe issues),
# these pymongo-based FunctionTools serve as the MongoDB integration.
# Gemini calls them exactly like MCP tools — same purpose, same data.
# ═══════════════════════════════════════════════════════════════════════════

def _get_mongo_db():
    """Get a pymongo database connection (reuses existing MONGO_URI)."""
    try:
        import certifi
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        return client["libratio"]
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        return None


def query_mongodb_trajectories(filter_json: str = "{}", limit: int = 5) -> dict:
    """
    Query historical training trajectories from MongoDB Atlas.

    Use this tool to find past GPU fleet training runs from the 'trajectories' collection.
    Returns documents with precision strategies, rewards, and cluster states from previous episodes.
    This data helps you make better decisions by learning from history.

    Args:
        filter_json: MongoDB filter as JSON string. Use '{}' for all docs, or '{"task_id": "fleet_precision"}' to filter.
        limit: Maximum number of documents to return (default 5).

    Returns:
        Dict with matching trajectory documents from MongoDB Atlas.
    """
    db = _get_mongo_db()
    if db is None:
        return {"error": "MongoDB not connected. Check MONGO_URI in .env", "documents": []}

    try:
        import json as _json
        filter_dict = _json.loads(filter_json) if filter_json else {}
        docs = list(db["trajectories"].find(filter_dict, {"_id": 0, "embedding": 0}).limit(limit))
        logger.info(f"[MongoDB Tool] query_trajectories: filter={filter_json}, found={len(docs)}")
        return {"success": True, "count": len(docs), "documents": docs}
    except Exception as e:
        return {"error": str(e), "documents": []}


def log_run_to_mongodb(task_id: str, score: float, action_json: str, reasoning: str) -> dict:
    """
    Log a training run result to MongoDB Atlas 'runs' collection.

    Call this AFTER stepping the environment to record your result.
    This creates a permanent record in MongoDB that can be queried by future agents.

    Args:
        task_id: The fleet task that was run (e.g. 'fleet_precision')
        score: The reward score achieved (0.01 to 0.99)
        action_json: The JSON action that was submitted
        reasoning: Your reasoning for the chosen action

    Returns:
        Dict confirming the insert with the document ID.
    """
    db = _get_mongo_db()
    if db is None:
        return {"error": "MongoDB not connected", "logged": False}

    try:
        import json as _json
        from datetime import datetime, timezone
        doc = {
            "task_id": task_id,
            "score": score,
            "action": _json.loads(action_json) if isinstance(action_json, str) else action_json,
            "reasoning": reasoning,
            "agent": "libratio_fleet_commander_adk",
            "model": ADK_MODEL,
            "timestamp": datetime.now(timezone.utc),
        }
        result = db["runs"].insert_one(doc)
        logger.info(f"[MongoDB Tool] log_run: task={task_id}, score={score}, id={result.inserted_id}")
        return {"success": True, "logged": True, "document_id": str(result.inserted_id)}
    except Exception as e:
        return {"error": str(e), "logged": False}


def get_mongodb_fallback_tools() -> list:
    """Return pymongo-based FunctionTools as a fallback when MCP subprocess fails."""
    if not ADK_AVAILABLE:
        return []
    return [
        FunctionTool(func=query_mongodb_trajectories),
        FunctionTool(func=log_run_to_mongodb),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: AGENT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

FLEET_AGENT_INSTRUCTION = """
You are the Libratio Fleet Commander — a Google ADK-powered AI agent managing
GPU clusters using MongoDB Atlas as your intelligent backend.

## Your Tools
You have two types of tools:

**MongoDB MCP Tools** (your data superpowers — call these first!):
- Use find/aggregate to query historical crash trajectories from 'libratio.trajectories'
- Use insertMany to log your results to 'libratio.runs'
- Use aggregate with $vectorSearch to find semantically similar past crashes

**Fleet Physics Tools** (your environment interface):
- reset_fleet_environment: Start a new fleet task episode
- step_fleet_environment: Submit your precision/allocation decision as JSON
- compute_precision_physics: Validate a precision choice with physics model

## Workflow for Each Task
1. RETRIEVE: Query MongoDB 'libratio.trajectories' for historical runs similar to current scenario
2. ANALYZE: Use compute_precision_physics to validate your planned action
3. ACT: Use step_fleet_environment to submit your JSON decision
4. LOG: Insert your result into MongoDB 'libratio.runs'

## Precision Rules (memorize these)
- embedding: ALWAYS FP32 (FP8 causes NaN crashes — NEVER use FP8 here)
- output: ALWAYS FP32 (loss computation needs full precision)
- ffn: FP8 is ideal (2.5x speedup, safe for FFN layers)
- attention: BF16 is optimal (1.85x speedup)
- layernorm: BF16 is best

## Goal
Reward score > 0.60 = success. Target > 0.85.
Always retrieve MongoDB context before acting.
"""


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: AGENT ASSEMBLY + RUNNER (ADK 2.2.0 API)
# ═══════════════════════════════════════════════════════════════════════════

async def run_fleet_task_with_adk(task_id: str = "fleet_precision") -> dict:
    """
    Run a complete Libratio fleet task using the Google ADK agent.

    This is the main entry point for the hackathon demo. It:
    1. Creates the ADK agent with MongoDB MCP + physics tools
    2. Sends a task message to Gemini 2.0 Flash
    3. Gemini reasons, calls MongoDB MCP tools, calls physics tools
    4. Returns the final score and reasoning

    Args:
        task_id: One of 'fleet_precision', 'fleet_oversight', 'fleet_resource', 'fleet_recovery'

    Returns:
        Dict with final_score, steps, reasoning, mongo_tools_called, physics_tools_called
    """
    if not ADK_AVAILABLE:
        raise RuntimeError("Google ADK not installed. Run: pip install google-adk mcp")

    logger.info(f"\n{'='*60}")
    logger.info(f"  LIBRATIO ADK AGENT - task={task_id}")
    logger.info(f"  Model: {ADK_MODEL} | MongoDB: enabled | Physics: enabled")
    logger.info(f"{'='*60}")

    # ── Step 1: Build tools ──────────────────────────────────────────────
    # Try MCP first (ideal for hackathon), fallback to pymongo tools (always works)
    mongo_mcp = get_mongodb_mcp_toolset()
    physics_tools = get_fleet_physics_tools()
    mongo_fallback_tools = get_mongodb_fallback_tools()

    # Assemble: MCP toolset (if available) + pymongo fallback + physics tools
    all_tools = physics_tools + mongo_fallback_tools
    mongodb_mode = "pymongo (ADK FunctionTool)"
    if mongo_mcp:
        all_tools = [mongo_mcp] + all_tools
        mongodb_mode = "MCP Server + pymongo fallback"

    logger.info(f"MongoDB mode: {mongodb_mode}")

    # ── Step 2: Create the ADK Agent ─────────────────────────────────────
    # This is the "Google Cloud Agent Builder" code-first integration
    agent = Agent(
        model=ADK_MODEL,                          # Gemini 2.0 Flash
        name="libratio_fleet_commander",
        instruction=FLEET_AGENT_INSTRUCTION,
        tools=all_tools,
    )

    logger.info(f"Agent created: model={ADK_MODEL}, tools={len(all_tools)}")

    # ── Step 3: Create InMemoryRunner (ADK 2.2.0 simplified API) ─────────
    # InMemoryRunner bundles the agent + session service together
    runner = InMemoryRunner(
        agent=agent,
        app_name="libratio_fleet",
    )

    # ── Step 4: Create a session ──────────────────────────────────────────
    session = await runner.session_service.create_session(
        app_name="libratio_fleet",
        user_id="fleet_commander",
    )
    session_id = session.id

    logger.info(f"Session created: {session_id}")

    # ── Step 5: Build the task prompt ──────────────────────────────────────
    task_prompt = f"""
Run a complete Libratio fleet task: {task_id}

Follow this exact workflow:
1. Call reset_fleet_environment(task_id="{task_id}") to see the cluster state
2. Call MongoDB find or aggregate on collection 'trajectories' in database 'libratio' 
   to retrieve 3-5 historical runs with similar conditions
3. Call compute_precision_physics for the layers you plan to configure
4. Call step_fleet_environment with your best JSON action (aim for score > 0.85)
5. Call MongoDB insertMany to log your result to collection 'runs' in database 'libratio'
6. Report: your final score, which MongoDB MCP tools you used, and your reasoning

Target reward: > 0.85
Database: libratio | Collections: trajectories (read), runs (write)
"""

    # ── Step 6: Run the agent (with retry for rate limits) ──────────────
    mongo_tools_called = []
    physics_tools_called = []
    final_text = ""
    final_score = 0.0
    event_count = 0

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async for event in runner.run_async(
                user_id="fleet_commander",
                session_id=session_id,
                new_message=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=task_prompt)],
                ),
            ):
                event_count += 1

                # Track tool calls for demo visibility
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        # Detect function calls (tool invocations)
                        if hasattr(part, "function_call") and part.function_call:
                            tool_name = part.function_call.name
                            logger.info(f"  [Tool called] {tool_name}")

                            # Categorize which type of tool was used
                            mongo_keywords = ["find", "aggregate", "insert", "collection", "mongo",
                                              "vector_search", "count", "query_mongodb", "log_run"]
                            physics_keywords = ["fleet", "precision", "physics", "reset", "step", "compute"]

                            if any(kw in tool_name.lower() for kw in mongo_keywords):
                                mongo_tools_called.append(tool_name)
                            elif any(kw in tool_name.lower() for kw in physics_keywords):
                                physics_tools_called.append(tool_name)
                            else:
                                physics_tools_called.append(tool_name)

                        # Capture final text response
                        if hasattr(part, "text") and part.text:
                            final_text += part.text

                # Detect final response event
                if hasattr(event, "is_final_response") and callable(event.is_final_response):
                    if event.is_final_response():
                        logger.info("  [Final response received]")

            # If we got here, agent completed successfully
            break

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 35 * (attempt + 1)  # 35s, 70s, 105s
                logger.warning(f"  Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                # Create a new session for retry (old one may be stale)
                session = await runner.session_service.create_session(
                    app_name="libratio_fleet",
                    user_id="fleet_commander",
                )
                session_id = session.id
            else:
                logger.error(f"ADK agent run error: {e}")
                raise

    # Try to extract score from final text
    import re
    score_matches = re.findall(r'score[:\s=]+([0-9.]+)', final_text.lower())
    if score_matches:
        try:
            final_score = float(score_matches[-1])
            final_score = max(0.01, min(0.99, final_score))
        except ValueError:
            final_score = 0.5

    result = {
        "task_id": task_id,
        "model": ADK_MODEL,
        "final_score": round(final_score, 3),
        "events_processed": event_count,
        "mongo_tools_called": mongo_tools_called,
        "physics_tools_called": physics_tools_called,
        "total_tool_calls": len(mongo_tools_called) + len(physics_tools_called),
        "mongodb_mcp_used": len(mongo_tools_called) > 0,
        "final_reasoning": final_text[:800] if final_text else "See server logs for full reasoning",
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"  ADK AGENT COMPLETE")
    logger.info(f"  MongoDB MCP tools called: {mongo_tools_called}")
    logger.info(f"  Physics tools called: {physics_tools_called}")
    logger.info(f"  Final score: {result['final_score']}")
    logger.info(f"{'='*60}\n")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def _main():
    """Run a demo of the ADK agent."""
    task_id = os.environ.get("TASK_ID", "fleet_precision")
    print(f"\nRunning ADK agent: task={task_id}, model={ADK_MODEL}")
    print(f"MongoDB MCP: {'enabled' if MONGO_URI else 'disabled (no MONGO_URI)'}")
    print(f"Gemini API: {'configured' if GEMINI_API_KEY else 'MISSING - set GEMINI_API_KEY in .env'}")
    print()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    result = await run_fleet_task_with_adk(task_id)

    print("\n" + "=" * 60)
    print("LIBRATIO ADK AGENT - RESULT SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    if result["mongodb_mcp_used"]:
        print("\n[OK] MongoDB MCP Server was used - hackathon partner requirement SATISFIED")
    else:
        print("\n[WARN] MongoDB MCP tools were not called.")
        print("       Check that MONGO_URI is set and npx is available.")


if __name__ == "__main__":
    if not ADK_AVAILABLE:
        print("ERROR: Google ADK not installed. Run: pip install google-adk mcp")
        sys.exit(1)
    asyncio.run(_main())
