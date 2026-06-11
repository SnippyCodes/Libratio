"""
MongoDB Change Streams Real-Time Monitor — Libratio Fleet
=========================================================
Runs an event-driven background listener that monitors the 'runs' collection in
MongoDB Atlas. When a new training run is logged, the script automatically:

1. Detects failures or low-performing runs (score < 0.60).
2. Queries the 'gpu_telemetry_metrics' Time Series collection to build a step profile.
3. Automatically diagnoses the root cause (e.g., OOM vs thermal throttling).
4. Logs a detailed post-incident report to the 'incident_reports' collection.

This illustrates the exact kind of real-time, event-driven MLOps SRE workflow
enabled by MongoDB Atlas.
"""

import os
import json
import time
from datetime import datetime, timezone
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "libratio"


def run_change_stream_listener():
    print("=" * 60)
    print("  MongoDB Atlas Change Stream Listener - Real-Time MLOps SRE")
    print("=" * 60)

    if not MONGO_URI:
        print("[ERROR] MONGO_URI not set in .env")
        return

    retry_delay = 5  # Start with 5 seconds delay
    max_retry_delay = 60

    print("  Watching 'runs' collection for new executions...")
    print("  Press Ctrl+C to stop.\n")

    # Watch pipeline filters for only 'insert' operations
    pipeline = [{"$match": {"operationType": "insert"}}]

    while True:
        client = None
        try:
            # Connect to MongoDB
            client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
            client.admin.command("ping")
            
            db = client[DB_NAME]
            runs_col = db["runs"]
            metrics_col = db["gpu_telemetry_metrics"]
            reports_col = db["incident_reports"]

            # Reset retry delay on successful connection
            retry_delay = 5

            with runs_col.watch(pipeline) as stream:
                for change in stream:
                    run_doc = change["fullDocument"]
                    task_id = run_doc.get("task_id")
                    model_name = run_doc.get("model_name")
                    score = run_doc.get("score", 0.0)
                    success = run_doc.get("success", False)
                    rewards = run_doc.get("rewards", [])

                    print(f"[*] NEW RUN DETECTED: Task={task_id} | Model={model_name} | Score={score:.3f} | Success={success}")

                    # Check if it was a failure or low-performing run
                    if not success or score < 0.60:
                        print("    [WARNING] Run did not meet performance threshold. Generating SRE incident report...")

                        # Query Time Series collection for this specific run's telemetry steps
                        metrics_cursor = metrics_col.find(
                            {
                                "metadata.task_id": task_id,
                                "metadata.model_id": model_name,
                            }
                        ).sort("timestamp", 1)

                        steps_history = list(metrics_cursor)
                        print(f"    Retrieved {len(steps_history)} metrics steps from Time Series collection.")

                        # Profile the steps
                        steps_data = []
                        max_memory_gb = 0.0
                        has_high_thermal = False

                        for step in steps_history:
                            steps_data.append({
                                "step": step.get("step"),
                                "reward": step.get("reward"),
                                "memory_used_gb": step.get("memory_used_gb"),
                                "thermal_risk": step.get("thermal_risk"),
                                "power_util": step.get("power_util"),
                            })
                            if step.get("memory_used_gb", 0.0) > max_memory_gb:
                                max_memory_gb = step.get("memory_used_gb")
                            if step.get("thermal_risk") == "HIGH":
                                has_high_thermal = True

                        # Run auto-diagnostics
                        probable_cause = "Unknown cluster stress"
                        recommendation = "Re-evaluate precision strategy; check baseline parameter counts."

                        if max_memory_gb > 240.0:
                            probable_cause = "VRAM Memory Overflow (Potential OOM)"
                            recommendation = "Attention layers should be cast to BF16 or FFN to FP8 to reduce memory footprint."
                        elif has_high_thermal:
                            probable_cause = "GPU Thermal Risk Alert (High heat dissipation)"
                            recommendation = "Limit FP8 usage on high-priority attention layers; check GPU fans and power constraints."
                        elif score < 0.20:
                            probable_cause = "Severe Optimization Degradation"
                            recommendation = "Verify the precision strategy isn't applying FP8 to embedding or output layers."

                        report = {
                            "incident_id": f"INC-{int(time.time())}",
                            "timestamp": datetime.now(timezone.utc),
                            "task_id": task_id,
                            "model_name": model_name,
                            "score": score,
                            "max_memory_gb": max_memory_gb,
                            "probable_cause": probable_cause,
                            "recommendation": recommendation,
                            "steps_profiled": len(steps_data),
                            "run_id": run_doc.get("_id"),
                        }

                        # Save report to MongoDB
                        res = reports_col.insert_one(report)
                        print(f"    [OK] SRE Incident Report created: {report['incident_id']} (Ref ID: {res.inserted_id})")
                        print(f"       Cause: {probable_cause}")
                        print(f"       Rec:   {recommendation}\n")
        except KeyboardInterrupt:
            print("\nStopping Change Stream listener.")
            break
        except Exception as e:
            print(f"\n[WARN] Connection lost or Change Stream error: {e}")
            print(f"       Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
        finally:
            if client:
                client.close()


if __name__ == "__main__":
    run_change_stream_listener()
