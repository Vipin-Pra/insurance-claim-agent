import os
import sys

from fastapi.testclient import TestClient

from main import app
from app.tasks import TASKS, grade_task


def _read_dotenv_value(key: str) -> str:
    if not os.path.exists('.env'):
        return ""

    try:
        with open('.env', 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except OSError:
        return ""

    return ""


def _get_config_value(key: str, default: str = "") -> str:
    env_value = os.getenv(key)
    if env_value is not None and env_value != "":
        return env_value
    dotenv_value = _read_dotenv_value(key)
    return dotenv_value if dotenv_value != "" else default


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def validate_openenv_yaml() -> None:
    with open('openenv.yaml', 'r') as f:
        content = f.read()
        required = ['action_space', 'observation_space', 'tasks', 'entrypoint']
        missing = [k for k in required if k not in content]
        if missing:
            print(f"[ERROR] openenv.yaml missing required keys: {missing}")
            sys.exit(1)


def validate_tasks_and_graders() -> None:
    if len(TASKS) < 3:
        print("[ERROR] Must define at least 3 tasks.")
        sys.exit(1)

    for task_name, task_data in TASKS.items():
        state = {
            "documents_received": task_data["documents_provided"],
            "policy_verified": False,
            "fraud_analyzed": False,
            "missing_documents_requested": False,
            "last_action_type": None,
        }
        score = grade_task(task_name, task_data, state, [], done=False)
        if not (0.0 < score < 1.0):
            print(f"[ERROR] Grader score out of range for task '{task_name}': {score}")
            sys.exit(1)


def validate_api_contract() -> None:
    client = TestClient(app)
    require_api_key = _is_true(_get_config_value("REQUIRE_API_KEY", "false"))
    api_key = _get_config_value("API_KEY", "")
    auth_headers = {}

    if require_api_key:
        if not api_key:
            print("[ERROR] REQUIRE_API_KEY=true but API_KEY is missing in env/.env")
            sys.exit(1)
        auth_headers = {"X-API-Key": api_key}

    for task_name in ["easy", "medium", "hard"]:
        reset_resp = client.post("/reset", json={"task": task_name}, headers=auth_headers)
        if reset_resp.status_code != 200:
            print(f"[ERROR] /reset failed for task '{task_name}' with status {reset_resp.status_code}")
            sys.exit(1)

        session_id = reset_resp.headers.get("X-Session-ID")
        headers = dict(auth_headers)
        if session_id:
            headers["X-Session-ID"] = session_id
        payload = reset_resp.json()

        for key in ["message", "data", "reward", "done", "info"]:
            if key not in payload:
                print(f"[ERROR] /reset response missing key '{key}' for task '{task_name}'")
                sys.exit(1)

        step_resp = client.post(
            "/step",
            json={"action_type": "SearchPolicy", "policy_id": TASKS[task_name]["policy_id"]},
            headers=headers,
        )
        if step_resp.status_code != 200:
            print(f"[ERROR] /step failed for task '{task_name}' with status {step_resp.status_code}")
            sys.exit(1)

        step_payload = step_resp.json()
        for key in ["message", "data", "reward", "done", "info"]:
            if key not in step_payload:
                print(f"[ERROR] /step response missing key '{key}' for task '{task_name}'")
                sys.exit(1)

        if not (0.0 <= float(step_payload.get("reward", 0.0)) <= 1.0):
            print(f"[ERROR] /step reward out of range for task '{task_name}'")
            sys.exit(1)

        state_resp = client.get("/state", headers=headers)
        if state_resp.status_code != 200:
            print(f"[ERROR] /state failed for task '{task_name}' with status {state_resp.status_code}")
            sys.exit(1)


def main():
    print("Running Pre-submission Validator...")
    
    files_to_check = ['openenv.yaml', 'Dockerfile', 'inference.py', 'app/models.py', 'app/environment.py', 'main.py']
    for f in files_to_check:
        if not os.path.exists(f):
            print(f"[MISSING] Required file: {f}")
            sys.exit(1)
            
    print("[SUCCESS] All required files present.")
    
    validate_openenv_yaml()
    print("[SUCCESS] openenv.yaml has required keys.")

    validate_tasks_and_graders()
    print("[SUCCESS] Task graders are present and return scores in [0.0, 1.0].")

    validate_api_contract()
    print("[SUCCESS] API contract checks passed for /reset, /step, /state.")
            
    print("\nAll checks passed. You are ready to build the docker container and deploy to Hugging Face Spaces!")

if __name__ == "__main__":
    main()
