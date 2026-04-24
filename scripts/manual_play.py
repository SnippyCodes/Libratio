import httpx
import json

print("\n--- Starting Libratio Fleet (Manual Mode) ---")

try:
    # 1. Start the game (Task: Oversight)
    print("\n[1] Sending /reset command to the Fleet server...")
    response = httpx.post("http://localhost:7860/fleet/reset", json={"task_id": "fleet_oversight"}).json()
    
    print("\n--- OBSERVATION (What the agent sees) ---")
    print(json.dumps(response["observation"], indent=2))

    # 2. You choose what to do manually
    print("\n[2] Sending a manual action to continue monitoring...")
    action = {
        "action_type": "continue_monitoring",
        "analysis": "I am a human and I think this looks fine."
    }
    
    result = httpx.post("http://localhost:7860/fleet/step", json={"action": action}).json()
    
    print("\n--- REWARD (Score from the physics engine) ---")
    print(f"Score: {result['reward']['score']}")
    print(f"Feedback: {result['reward']['feedback']}")

except httpx.ConnectError:
    print("\n❌ ERROR: Could not connect to http://localhost:7860/")
    print("Did you forget to start the server? Run this in another terminal:")
    print("  .venv\\Scripts\\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 7860")
