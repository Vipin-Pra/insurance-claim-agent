from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from app.models import ActionSchema, ObservationSchema
from app.environment import InsuranceEnvironment

app = FastAPI(title="Insurance Claim Verification OpenEnv")

# Instantiate single environment.
# Note: For multi-user we would need session IDs, but OpenEnv benchmark evaluates isolated containers.
env = InsuranceEnvironment()

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"

@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/reset", response_model=ObservationSchema)
def reset_env(req: ResetRequest = None):
    task_name = req.task if req else "easy"
    try:
        obs = env.reset(task_name)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=ObservationSchema)
def step_env(action: ActionSchema):
    obs = env.step(action)
    return obs

@app.get("/state", response_model=ObservationSchema)
def get_state():
    return env.state()

if __name__ == "__main__":
    import uvicorn
    # Typically environments run on port 7860 for HuggingFace Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
