"""
MongoDB Atlas Vector Search Index Setup
========================================
Run this ONCE to create the vector search index in your Atlas cluster.

This script:
1. Connects to your MongoDB Atlas cluster
2. Creates the "trajectory_vector_index" Atlas Vector Search index
3. Creates regular indexes for performance (outcome, thermal_risk)
4. Verifies the index was created

After running this script, your Atlas cluster can perform semantic similarity
search across all trajectory documents using cosine distance on 768-dim vectors.

Usage:
    python setup_vector_index.py

Note: Index creation takes 1-2 minutes to propagate across Atlas nodes.
      You'll see "ACTIVE" status in Atlas UI when it's ready.
"""

import os
import time
import json
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv
import certifi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "libratio"
COLLECTION_NAME = "trajectories"

# The vector search index definition.
# - field: "embedding" — where we store the 768-dim vector
# - similarity: "cosine" — measures angle between vectors (best for semantic text)
# - dimensions: 768 — must match Gemini's text-embedding-004 output
VECTOR_INDEX_DEFINITION = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",          # Document field name
            "numDimensions": 768,          # Gemini text-embedding-004 = 768 dims
            "similarity": "cosine",        # Cosine similarity (ideal for semantic text)
        }
    ]
}

# The standard Full-Text Search index definition.
# Uses dynamic mappings to index all text fields (like failure_reason and source_text)
# to support keyword searches, fuzzy match, and autocomplete.
TEXT_SEARCH_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": True
    }
}

# Regular indexes for fast filtering (non-vector queries)
REGULAR_INDEXES = [
    [("outcome", 1)],               # Filter by outcome (SUCCESS/CRASH)
    [("thermal_risk", 1)],          # Filter by thermal risk level
    [("memory_used_gb", -1)],       # Sort by memory usage
    [("episode_id", 1)],            # Lookup by episode
]


def setup_indexes():
    print("=" * 60)
    print("  MongoDB Atlas Search & Vector Search Index Setup")
    print("  Libratio Fleet — Trajectory Collection")
    print("=" * 60)

    if not MONGO_URI:
        print("[ERROR] MONGO_URI not set in .env")
        return False

    print("\n[1/4] Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
        client.admin.command("ping")
        print("  [OK] Connected successfully")
    except Exception as e:
        print(f"  [ERROR] Connection failed: {e}")
        return False

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # ── Check if data exists ──
    count = collection.count_documents({})
    print(f"\n[2/4] Found {count} documents in '{COLLECTION_NAME}' collection")
    if count == 0:
        print("  [WARN] No documents found. Run upload_to_mongodb.py first!")
        print("     Vector Search requires documents with 'embedding' fields.")

    # ── Create regular indexes ──
    print("\n[3/4] Creating regular performance indexes...")
    for index_spec in REGULAR_INDEXES:
        try:
            collection.create_index(index_spec)
            field = index_spec[0][0]
            print(f"  [OK] Index on '{field}' created")
        except Exception as e:
            print(f"  [WARN] Index creation note: {e}")

    # ── Create Atlas Search Indexes ──
    print("\n[4/4] Creating Atlas Search indexes...")

    # 1. Create Vector Search Index
    print("\n  - Setting up 'trajectory_vector_index' (type: vectorSearch)...")
    try:
        existing_indexes = list(collection.list_search_indexes())
        index_names = [idx.get("name") for idx in existing_indexes]

        if "trajectory_vector_index" in index_names:
            print("    [INFO] Index 'trajectory_vector_index' already exists")
            for idx in existing_indexes:
                if idx.get("name") == "trajectory_vector_index":
                    print(f"       Status: {idx.get('status', 'UNKNOWN')}")
        else:
            search_index_model = SearchIndexModel(
                definition=VECTOR_INDEX_DEFINITION,
                name="trajectory_vector_index",
                type="vectorSearch",
            )
            collection.create_search_index(model=search_index_model)
            print("    [OK] Vector Search index creation initiated!")
    except Exception as e:
        print(f"    [WARN] Vector Search index creation failed: {e}")

    # 2. Create Full-Text Search Index
    print("\n  - Setting up 'trajectory_text_search_index' (type: search)...")
    try:
        existing_indexes = list(collection.list_search_indexes())
        index_names = [idx.get("name") for idx in existing_indexes]

        if "trajectory_text_search_index" in index_names:
            print("    [INFO] Index 'trajectory_text_search_index' already exists")
            for idx in existing_indexes:
                if idx.get("name") == "trajectory_text_search_index":
                    print(f"       Status: {idx.get('status', 'UNKNOWN')}")
        else:
            search_index_model = SearchIndexModel(
                definition=TEXT_SEARCH_INDEX_DEFINITION,
                name="trajectory_text_search_index",
                type="search",
            )
            collection.create_search_index(model=search_index_model)
            print("    [OK] Full-Text Search index creation initiated!")
    except Exception as e:
        print(f"    [WARN] Full-Text Search index creation failed: {e}")

    # Wait and check status
    print("\n  Waiting 10 seconds for indexes to start building...")
    time.sleep(10)
    try:
        indexes = list(collection.list_search_indexes())
        print("\n  Current Search Indexes Status:")
        for idx in indexes:
            name = idx.get("name")
            status = idx.get("status", "UNKNOWN")
            print(f"  - {name} ({idx.get('type', 'UNKNOWN')}): {status}")
    except Exception as e:
        print(f"  Could not list search indexes status: {e}")

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    client.close()
    return True


if __name__ == "__main__":
    setup_indexes()
