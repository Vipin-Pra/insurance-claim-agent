import os
import sys

def main():
    print("Running Pre-submission Validator...")
    
    files_to_check = ['openenv.yaml', 'Dockerfile', 'inference.py', 'app/models.py', 'app/environment.py', 'main.py']
    for f in files_to_check:
        if not os.path.exists(f):
            print(f"[MISSING] Required file: {f}")
            sys.exit(1)
            
    print("[SUCCESS] All required files present.")
    
    with open('openenv.yaml', 'r') as f:
        content = f.read()
        if 'action_space' not in content or 'observation_space' not in content or 'tasks' not in content:
            print("[ERROR] openenv.yaml missing required keys (action_space, observation_space, tasks).")
            sys.exit(1)
            
    print("[SUCCESS] openenv.yaml is valid.")
    print("\nAll checks passed. You are ready to build the docker container and deploy to Hugging Face Spaces!")

if __name__ == "__main__":
    main()
