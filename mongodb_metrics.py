"""
MongoDB Time Series Metrics Logging — Libratio Fleet
===================================================
Uses MongoDB's native Time Series collections to store step-by-step performance
telemetry (VRAM, thermal status, reward, power draw) from multi-agent training runs.

Why Time Series Collections?
- Highly optimized storage for sequential measurements over time.
- Automatic compression of metrics (reduces disk footprint by up to 95%).
- Efficient querying by time windows, intervals, and aggregations.
- Demonstrates advanced MongoDB capabilities to hackathon judges.
"""

import os
from datetime import datetime, timezone
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "libratio"
METRICS_COLLECTION = "gpu_telemetry_metrics"


def get_mongo_client():
    if not MONGO_URI:
        return None
    try:
        return MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
    except Exception:
        return None


def create_metrics_timeseries_collection() -> bool:
    """
    Creates a native MongoDB Time Series collection if it doesn't already exist.
    """
    client = get_mongo_client()
    if not client:
        print("[WARN] MONGO_URI not configured. Skipping Time Series creation.")
        return False

    db = client[DB_NAME]

    # Check if the collection already exists
    if METRICS_COLLECTION in db.list_collection_names():
        print(f"  [INFO] Time Series collection '{METRICS_COLLECTION}' already exists.")
        client.close()
        return True

    try:
        # Create a native time-series collection in MongoDB
        # - timeField: "timestamp" (specifies the date field in each document)
        # - metaField: "metadata" (specifies the tags used to categorize the metrics)
        # - granularity: "seconds" (tells MongoDB to optimize for sub-minute updates)
        db.create_collection(
            METRICS_COLLECTION,
            timeseries={
                "timeField": "timestamp",
                "metaField": "metadata",
                "granularity": "seconds",
            }
        )
        print(f"  [OK] Created native MongoDB Time Series collection '{METRICS_COLLECTION}'!")
        client.close()
        return True
    except Exception as e:
        print(f"  [WARN] Failed to create native Time Series collection: {e}")
        print("     (Usually means the database user does not have 'createCollection' permission)")
        client.close()
        return False


def log_step_metrics(
    task_id: str,
    model_id: str,
    step: int,
    reward: float,
    memory_used_gb: float,
    thermal_risk: str,
    power_util: float,
) -> bool:
    """
    Logs a single measurement point to the Time Series collection.
    """
    client = get_mongo_client()
    if not client:
        return False

    try:
        db = client[DB_NAME]
        col = db[METRICS_COLLECTION]

        # Structure for Time Series: timeField (timestamp) + metaField (metadata) + measurements
        metric_doc = {
            "timestamp": datetime.now(timezone.utc),
            "metadata": {
                "task_id": task_id,
                "model_id": model_id,
            },
            "step": step,
            "reward": float(reward),
            "memory_used_gb": float(memory_used_gb),
            "thermal_risk": str(thermal_risk),
            "power_util": float(power_util),
        }

        col.insert_one(metric_doc)
        client.close()
        return True
    except Exception:
        # Fail silently to prevent telemetry logging from crashing the main inference run
        if client:
            client.close()
        return False


if __name__ == "__main__":
    print("Initializing Time Series collection...")
    create_metrics_timeseries_collection()
