import os
import json
import time
import uuid
import importlib
from threading import Lock
from typing import Dict, Optional, Tuple

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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def _env_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


REQUIRE_API_KEY = False
API_KEY = os.environ.get("API_KEY", "")
SESSION_BACKEND = os.environ.get("SESSION_BACKEND", "memory").strip().lower()
SESSION_TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", 3600)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", "true")
RATE_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
RATE_LIMIT_IP_MAX_REQUESTS = _env_int("RATE_LIMIT_IP_MAX_REQUESTS", 120)
RATE_LIMIT_SESSION_MAX_REQUESTS = _env_int("RATE_LIMIT_SESSION_MAX_REQUESTS", 180)

_REDIS_CLIENT = None
_IP_RATE_STATE: Dict[str, Tuple[int, int]] = {}
_SESSION_RATE_STATE: Dict[str, Tuple[int, int]] = {}
_RATE_LOCK = Lock()


def _redis_key(session_id: str) -> str:
    return f"insurance:session:{session_id}"


def _get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    try:
        redis_module = importlib.import_module("redis")
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Redis backend selected but 'redis' package is not installed.",
        ) from exc

    _REDIS_CLIENT = redis_module.Redis.from_url(REDIS_URL, decode_responses=True)
    return _REDIS_CLIENT


def _save_env_redis(session_id: str, env: InsuranceEnvironment) -> None:
    client = _get_redis_client()
    payload = json.dumps(env.to_snapshot())
    client.setex(_redis_key(session_id), SESSION_TTL_SECONDS, payload)


def _load_env_redis(session_id: str) -> Optional[InsuranceEnvironment]:
    client = _get_redis_client()
    payload = client.get(_redis_key(session_id))
    if payload is None:
        return None

    snapshot = json.loads(payload)
    env = InsuranceEnvironment.from_snapshot(snapshot)
    client.expire(_redis_key(session_id), SESSION_TTL_SECONDS)
    return env


def _count_sessions_redis() -> int:
    client = _get_redis_client()
    return sum(1 for _ in client.scan_iter(match="insurance:session:*") )


class APIError(BaseModel):
    code: str
    message: str
    request_id: str


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


def _current_window_start() -> int:
    now = int(time.time())
    window = max(1, RATE_LIMIT_WINDOW_SECONDS)
    return now - (now % window)


def _check_bucket(state: Dict[str, Tuple[int, int]], key: str, max_requests: int) -> Tuple[bool, int]:
    if max_requests <= 0:
        return True, 0

    now = int(time.time())
    window_start = _current_window_start()
    count, stored_window = state.get(key, (0, window_start))
    if stored_window != window_start:
        count = 0
        stored_window = window_start

    count += 1
    state[key] = (count, stored_window)
    allowed = count <= max_requests
    retry_after = max(1, (stored_window + max(1, RATE_LIMIT_WINDOW_SECONDS)) - now)
    return allowed, retry_after


def _prune_rate_state() -> None:
    current_window = _current_window_start()
    stale_windows = {current_window - max(1, RATE_LIMIT_WINDOW_SECONDS), current_window}
    for key in list(_IP_RATE_STATE.keys()):
        if _IP_RATE_STATE[key][1] not in stale_windows:
            del _IP_RATE_STATE[key]
    for key in list(_SESSION_RATE_STATE.keys()):
        if _SESSION_RATE_STATE[key][1] not in stale_windows:
            del _SESSION_RATE_STATE[key]


def _rate_limit_error(request_id: str, retry_after: int, scope: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
        content=APIError(
            code="rate_limited",
            message=f"Rate limit exceeded for {scope}. Retry later.",
            request_id=request_id,
        ).model_dump(),
    )


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    if RATE_LIMIT_ENABLED:
        client_host = request.client.host if request.client else "unknown"
        session_id = request.headers.get("X-Session-ID", DEFAULT_SESSION_ID)
        with _RATE_LOCK:
            _prune_rate_state()
            ip_allowed, ip_retry_after = _check_bucket(_IP_RATE_STATE, client_host, RATE_LIMIT_IP_MAX_REQUESTS)
            if not ip_allowed:
                return _rate_limit_error(request_id, ip_retry_after, "ip")

            sess_allowed, sess_retry_after = _check_bucket(
                _SESSION_RATE_STATE, session_id, RATE_LIMIT_SESSION_MAX_REQUESTS
            )
            if not sess_allowed:
                return _rate_limit_error(request_id, sess_retry_after, "session")

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

    if SESSION_BACKEND == "redis":
        try:
            env = _load_env_redis(sid)
            if env is None:
                env = InsuranceEnvironment()
                _save_env_redis(sid, env)
            return sid, env
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Redis session backend unavailable: {exc}") from exc

    with SESSIONS_LOCK:
        env = SESSIONS.get(sid)
        if env is None:
            env = InsuranceEnvironment()
            SESSIONS[sid] = env
    return sid, env


def persist_env(session_id: str, env: InsuranceEnvironment) -> None:
    if SESSION_BACKEND == "redis":
        try:
            _save_env_redis(session_id, env)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to persist session state: {exc}") from exc
    else:
        with SESSIONS_LOCK:
            SESSIONS[session_id] = env


def count_active_sessions() -> int:
    if SESSION_BACKEND == "redis":
        try:
            return _count_sessions_redis()
        except Exception:
            return -1
    return len(SESSIONS)


@app.get("/")
def read_root():
    return {
        "status": "running",
        "require_api_key": REQUIRE_API_KEY,
        "session_backend": SESSION_BACKEND,
        "active_sessions": count_active_sessions(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if SESSION_BACKEND == "redis":
        try:
            _get_redis_client().ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Redis not ready: {exc}") from exc

    return {
        "status": "ready",
        "session_backend": SESSION_BACKEND,
        "active_sessions": count_active_sessions(),
    }


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
        persist_env(sid, env)
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
    persist_env(sid, env)
    response.headers["X-Session-ID"] = sid
    return obs


@app.get("/state", response_model=ObservationSchema)
def get_state(
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    _auth: None = Depends(require_api_key),
):
    sid, env = get_or_create_env(x_session_id)
    persist_env(sid, env)
    response.headers["X-Session-ID"] = sid
    return env.state()


if __name__ == "__main__":
    import uvicorn
    # Typically environments run on port 7860 for HuggingFace Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
