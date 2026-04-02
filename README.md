---
title: InsuranceClaimVerification
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Insurance Claim Verification OpenEnv Environment

A complete, real-world OpenEnv environment simulating an AI agent acting as an insurance adjuster. The agent must verify facts, request missing documents, flag suspicious activities, and approve/reject claims appropriately.

## Tasks
* **Easy**: Straightforward verification. All required documents present, policy active, no fraud.
* **Medium**: Documents are missing. Agent must use the `RequestDocument` action before approving.
* **Hard**: Potential fraud present. Discrepancies between claim and police/fire reports must be analyzed. Agent must use `AnalyzeFraud` and then `RejectClaim`.

## Usage (LLM Evaluation Baseline)

For the OpenEnv benchmark evaluation:
```bash
# 1. Start the environment server (FastAPI)
pip install -r requirements.txt
python main.py

# 2. Run the baseline evaluation loop
export HF_TOKEN="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

## Usage (Stable-Baselines3 Reinforcement Learning)

The environment includes a Gymnasium wrapper which flattens the complex string-based semantic variables into discrete multidimensional math spaces for RL algorithms (PPO) to learn from!
```bash
pip install -r requirements.txt
python train_rl.py
```

Training now saves a reusable model at `models/ppo_insurance.zip`.

To evaluate the saved model across multiple episodes and tasks:
```bash
python evaluate_rl.py
```

Optional arguments:
```bash
# Evaluate more episodes
python evaluate_rl.py --episodes 90

# Evaluate a different saved model path
python evaluate_rl.py --model-path models/ppo_insurance

# Print state transitions during evaluation
python evaluate_rl.py --render
```

## Validation
```bash
python validator.py
```
