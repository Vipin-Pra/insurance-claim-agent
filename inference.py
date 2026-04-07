import os
import json
import time
import requests
from openai import OpenAI

# Required environment variables for OpenEnv evaluation
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
ENV_API_KEY = os.getenv("ENV_API_KEY")
SEED = int(os.getenv("INFERENCE_SEED", "42"))

client = OpenAI(
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy"),
    base_url=API_BASE_URL
)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an AI Insurance Adjuster Agent. Your goal is to process an insurance claim by interacting with the environment.
You can take one of the following actions. Return ONLY a valid JSON object matching this schema:
{
  "action_type": "SearchPolicy" | "RequestDocument" | "AnalyzeFraud" | "ApproveClaim" | "RejectClaim",
  "policy_id": "...", // Required if SearchPolicy. Find this in the state.
  "document_type": "...", // Required if RequestDocument. Usually 'Police Report' or 'Purchase Receipt'.
  "claim_id": "...", // Required if AnalyzeFraud. Find this in the state.
  "reason": "..." // Required if ApproveClaim or RejectClaim. Provide justification.
}
Rules:
1. Always SearchPolicy first to verify it's Active.
2. RequestDocument for any missing required documents if they aren't provided.
3. If documents contradict, or specifically for 'hard' tasks, use AnalyzeFraud.
4. If AnalyzeFraud returns a risk, you MUST RejectClaim. Otherwise, if all is good, ApproveClaim.
"""

def solve_task(task_name):
    print(f"[START] {task_name}")
    headers = {}
    if ENV_API_KEY:
        headers["X-API-Key"] = ENV_API_KEY

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, headers=headers)
        res.raise_for_status()
        obs = res.json()
        session_id = res.headers.get("X-Session-ID")
        if session_id:
            headers["X-Session-ID"] = session_id
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        print(f"[END] {task_name}")
        return 0.0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Initial State:\n{json.dumps(obs, indent=2)}"}
    ]
    
    score = 0.0
    for step in range(10):
        print(f"[STEP] {step}")
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={ "type": "json_object" },
                temperature=0,
                seed=SEED,
            )
            action_text = response.choices[0].message.content
            action = json.loads(action_text)
            print(f"Action: {json.dumps(action)}")
            
            messages.append({"role": "assistant", "content": action_text})
            
            res = requests.post(f"{ENV_URL}/step", json=action, headers=headers)
            res.raise_for_status()
            obs = res.json()

            new_session_id = res.headers.get("X-Session-ID")
            if new_session_id:
                headers["X-Session-ID"] = new_session_id
            
            print(f"Observation: {obs['message']}")
            messages.append({"role": "user", "content": f"Observation:\n{json.dumps(obs, indent=2)}"})
            
            score = obs.get("reward", score)
            if obs.get("done", False):
                print(f"[END] {task_name} | Score: {score}")
                break
                
        except Exception as e:
            print(f"Error during agent loop: {e}")
            print(f"[END] {task_name} | Score: {score}")
            break
            
        time.sleep(0.2)
        
    return score

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = solve_task(task)
        
    print("\n=== FINAL SCORES ===")
    for k, v in scores.items():
        print(f"{k}: {v:.2f}")
