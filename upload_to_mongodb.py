"""
Upload Synthetic Trajectory Data to MongoDB Atlas (with Vector Embeddings)
===========================================================================
This script uploads the synthetic_trajectories.json dataset to MongoDB Atlas
AND generates Gemini embedding vectors for each trajectory record.

The embeddings enable Atlas Vector Search — converting your MongoDB database
from a plain JSON store into a semantic similarity search engine.

What this script does:
1. Reads synthetic_trajectories.json (1,000 episodes, ~3,000 trajectory records)
2. For each trajectory record, generates a 768-dim embedding via Gemini API
3. Inserts all records WITH their embedding vectors into MongoDB
4. Creates the 'trajectories' flat collection (unnested from episodes)
5. Also stores the full episode documents in 'episodes' collection

Collections created:
- libratio.trajectories  — flat trajectory records with embedding vectors (for Vector Search)
- libratio.episodes      — full episode documents with all trajectories nested
- libratio.runs          — live inference run telemetry (written by fleet_inference.py)

Usage:
    python upload_to_mongodb.py

Note: Embedding 3,000 trajectories takes ~5 minutes (Gemini API rate limits).
      A progress bar is shown as embeddings are generated.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from mongodb_vector import embed_trajectory, trajectory_to_text, EMBEDDING_DIM, _GEMINI_AVAILABLE

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "libratio"
TRAJECTORIES_COLLECTION = "trajectories"
EPISODES_COLLECTION = "episodes"
BATCH_SIZE = 50       # Insert this many documents at a time
EMBED_BATCH_DELAY = 0.1  # Seconds between batches to avoid API rate limits


def upload_data():
    print("=" * 60)
    print("  MongoDB Atlas Upload — With Vector Embeddings")
    print("=" * 60)

    if not MONGO_URI:
        print("[ERROR] MONGO_URI not set in .env")
        return

    # ── Connect ──
    print("\n[1/5] Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
        print("  [OK] Connected")
    except Exception as e:
        print(f"  [ERROR] Connection failed: {e}")
        return

    db = client[DB_NAME]
    trajectories_col = db[TRAJECTORIES_COLLECTION]
    episodes_col = db[EPISODES_COLLECTION]

    # ── Read dataset ──
    print("\n[2/5] Reading synthetic dataset...")
    with open("synthetic_trajectories.json", "r") as f:
        dataset = json.load(f)

    total_episodes = len(dataset)
    total_trajs = sum(len(ep.get("trajectories", [])) for ep in dataset)
    print(f"  Episodes: {total_episodes}")
    print(f"  Trajectory records: {total_trajs}")
    print(f"  Embedding model: {'Gemini text-embedding-004 (semantic)' if _GEMINI_AVAILABLE else 'Fallback hash (no Gemini key)'}")
    print(f"  Embedding dimensions: {EMBEDDING_DIM}")

    # ── Prepare flat trajectory documents with embeddings ──
    print(f"\n[3/5] Generating embeddings for {total_trajs} trajectory records in parallel...")
    print("  Accelerated using ThreadPoolExecutor for concurrent Gemini API requests.")

    flat_trajectories = []
    for ep in dataset:
        episode_id = ep.get("episode_id", 0)
        cluster_capacity = ep.get("cluster_capacity_gb", 0)
        episode_memory_overflow = ep.get("memory_overflow", False)

        for traj in ep.get("trajectories", []):
            enriched = {
                **traj,
                "episode_id": episode_id,
                "cluster_capacity_gb": cluster_capacity,
                "episode_memory_overflow": episode_memory_overflow,
            }
            flat_trajectories.append(enriched)

    start_time = time.time()
    processed = 0

    def process_record(index, record):
        record["embedding"] = embed_trajectory(record)
        record["embedding_source_text"] = trajectory_to_text(record)
        return index, record

    # Process records concurrently in a thread pool (max 20 workers to balance speed & rate limits)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_record, i, rec): i for i, rec in enumerate(flat_trajectories)}
        
        for future in as_completed(futures):
            index, record = future.result()
            flat_trajectories[index] = record
            processed += 1

            # Progress update every 100 records
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (total_trajs - processed) / rate
                print(f"  [{processed}/{total_trajs}] "
                      f"{processed/total_trajs*100:.0f}% — "
                      f"{rate:.1f} records/sec — "
                      f"~{remaining:.0f}s remaining")

    elapsed_total = time.time() - start_time
    print(f"\n  [OK] Generated {len(flat_trajectories)} embeddings in {elapsed_total:.0f}s")

    # ── Upload trajectories (with embeddings) ──
    print(f"\n[4/5] Uploading to MongoDB Atlas...")

    # Drop and recreate collections for a clean state
    # (safe to do — this is synthetic training data, not production data)
    print(f"  Clearing existing 'trajectories' collection...")
    trajectories_col.drop()

    print(f"  Inserting {len(flat_trajectories)} trajectory documents with embeddings...")
    inserted_count = 0
    for i in range(0, len(flat_trajectories), BATCH_SIZE):
        batch = flat_trajectories[i:i + BATCH_SIZE]
        result = trajectories_col.insert_many(batch)
        inserted_count += len(result.inserted_ids)

    print(f"  [OK] Inserted {inserted_count} trajectory records into 'trajectories'")

    # ── Upload episodes ──
    print(f"\n  Uploading {total_episodes} full episode documents to 'episodes'...")
    episodes_col.drop()
    result = episodes_col.insert_many(dataset)
    print(f"  [OK] Inserted {len(result.inserted_ids)} episodes into 'episodes'")

    # ── Print stats ──
    print(f"\n[5/5] Upload Summary")
    print("-" * 40)
    total_crashes = sum(
        1 for t in flat_trajectories if t.get("outcome") != "SUCCESS"
    )
    total_success = len(flat_trajectories) - total_crashes
    has_embeddings = sum(1 for t in flat_trajectories if t.get("embedding"))
    thermal_high = sum(1 for t in flat_trajectories if t.get("thermal_risk") == "HIGH")

    print(f"  Trajectories uploaded : {inserted_count}")
    print(f"  With embeddings       : {has_embeddings}")
    print(f"  Successful outcomes   : {total_success}")
    print(f"  Crash outcomes        : {total_crashes}")
    print(f"  High thermal risk     : {thermal_high}")
    print(f"  Episodes uploaded     : {len(result.inserted_ids)}")
    print()
    print("  [OK] MongoDB is now ready for Atlas Vector Search!")
    print("  Next step: python setup_vector_index.py")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    upload_data()
