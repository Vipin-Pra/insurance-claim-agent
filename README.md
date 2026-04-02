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

## API Hardening Notes (Production)

The service now supports production-oriented controls:

- API key auth via header `X-API-Key` (optional by default).
- Per-session isolation via header `X-Session-ID`.
- Health endpoints: `GET /health` and `GET /ready`.
- Request tracing via `X-Request-ID` response header.
- Configurable session backend: in-memory (`memory`) or Redis (`redis`) with TTL.
- Rate limiting with `429` responses (per IP and per session).

Enable API key auth:
```bash
export REQUIRE_API_KEY=true
export API_KEY="change-me"
python main.py
```

Enable Redis session backend with TTL:
```bash
export SESSION_BACKEND=redis
export REDIS_URL="redis://localhost:6379/0"
export SESSION_TTL_SECONDS=3600
python main.py
```

Configure rate limiting:
```bash
export RATE_LIMIT_ENABLED=true
export RATE_LIMIT_WINDOW_SECONDS=60
export RATE_LIMIT_IP_MAX_REQUESTS=120
export RATE_LIMIT_SESSION_MAX_REQUESTS=180
python main.py
```

When limits are exceeded, the API returns `429` with a `Retry-After` header.

Example request flow with session header:
```bash
# Reset and get a fresh session (or provide your own X-Session-ID)
curl -X POST http://localhost:7860/reset \
	-H "Content-Type: application/json" \
	-H "X-API-Key: change-me" \
	-d '{"task":"easy"}'

# Use the same X-Session-ID value for step/state requests
curl -X POST http://localhost:7860/step \
	-H "Content-Type: application/json" \
	-H "X-API-Key: change-me" \
	-H "X-Session-ID: <session-id>" \
	-d '{"action_type":"SearchPolicy","policy_id":"POL-1001"}'
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

## Google Colab (Train + Evaluate)

Run this in a single Colab cell:
```bash
!git clone https://github.com/Vipin-Pra/insurance-claim-agent.git
%cd insurance-claim-agent
!pip install -r requirements.txt
!python train_rl.py
!python evaluate_rl.py --episodes 90
```

## Validation
```bash
python validator.py
```
