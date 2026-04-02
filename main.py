import os
import uuid
from threading import Lock
from typing import Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.models import ActionSchema, ObservationSchema
from app.environment import InsuranceEnvironment

app = FastAPI(title="Insurance Claim Verification OpenEnv")

# Session-backed environments for basic multi-user isolation.
SESSIONS: Dict[str, InsuranceEnvironment] = {}
SESSIONS_LOCK = Lock()
DEFAULT_SESSION_ID = "default"


def _env_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


REQUIRE_API_KEY = _env_bool("REQUIRE_API_KEY", "false")
API_KEY = os.environ.get("API_KEY", "")


class APIError(BaseModel):
    code: str
    message: str
    request_id: str


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(code="http_error", message=detail, request_id=request_id).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content=APIError(
            code="internal_error",
            message="An internal error occurred.",
            request_id=request_id,
        ).model_dump(),
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if not REQUIRE_API_KEY:
        return
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY is required.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def get_or_create_env(session_id: Optional[str]) -> tuple[str, InsuranceEnvironment]:
    sid = session_id or DEFAULT_SESSION_ID
    with SESSIONS_LOCK:
        env = SESSIONS.get(sid)
        if env is None:
            env = InsuranceEnvironment()
            SESSIONS[sid] = env
    return sid, env


@app.get("/")
def read_root():
    return {
        "status": "running",
        "require_api_key": REQUIRE_API_KEY,
        "active_sessions": len(SESSIONS),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready", "active_sessions": len(SESSIONS)}


@app.post("/reset", response_model=ObservationSchema)
def reset_env(
    response: Response,
    req: ResetRequest = None,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    _auth: None = Depends(require_api_key),
):
    task_name = req.task if req else "easy"
    try:
        sid = x_session_id or str(uuid.uuid4())
        _, env = get_or_create_env(sid)
        obs = env.reset(task_name)
        response.headers["X-Session-ID"] = sid
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=ObservationSchema)
def step_env(
    action: ActionSchema,
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    _auth: None = Depends(require_api_key),
):
    sid, env = get_or_create_env(x_session_id)
    obs = env.step(action)
    response.headers["X-Session-ID"] = sid
    return obs


@app.get("/state", response_model=ObservationSchema)
def get_state(
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    _auth: None = Depends(require_api_key),
):
    sid, env = get_or_create_env(x_session_id)
    response.headers["X-Session-ID"] = sid
    return env.state()


if __name__ == "__main__":
    import uvicorn
    # Typically environments run on port 7860 for HuggingFace Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
