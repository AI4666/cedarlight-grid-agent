"""FastAPI app. Run:  uvicorn agent.api:app --app-dir src --host 0.0.0.0 --port 8000

Auth: mutating endpoints need header X-API-Key == $CEDARLIGHT_API_KEY (fail closed: if the variable is unset they are
disabled). /resume additionally needs the admin token in $CEDARLIGHT_ADMIN_TOKEN. /health is open."""
from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .service import AgentService

app = FastAPI(title="CedarLight Grid Agent", version="0.1.0")
_service: AgentService | None = None


def service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("CEDARLIGHT_API_KEY")
    if not expected:
        raise HTTPException(503, "API key not configured; mutating endpoints are disabled")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "invalid or missing API key")


class StepIn(BaseModel):
    percepts: list[dict[str, Any]] = Field(max_length=500)


class ApproveIn(BaseModel):
    approver: str = Field(min_length=1, max_length=64)


class KillIn(BaseModel):
    actor: str = Field(min_length=1, max_length=64)
    reason: str = Field(default="manual", max_length=200)


class ResumeIn(BaseModel):
    actor: str = Field(min_length=1, max_length=64)
    admin_token: str


@app.get("/health")
def health():
    return service().health()


@app.get("/scenarios")
def scenarios():
    return service().scenarios()


@app.get("/state")
def state():
    return service().state()


@app.get("/audit")
def audit(limit: int = 200, stage: str | None = None):
    return service().audit(limit, stage)


@app.post("/step", dependencies=[Depends(require_key)])
def step(body: StepIn):
    return service().step(body.percepts)


@app.post("/scenario/{name}", dependencies=[Depends(require_key)])
def scenario(name: str):
    if name not in service().scenarios():
        raise HTTPException(404, "unknown scenario")
    return service().run_scenario(name)


@app.post("/approve/{action_id}", dependencies=[Depends(require_key)])
def approve(action_id: str, body: ApproveIn):
    return service().approve(action_id, body.approver)


@app.post("/reject/{action_id}", dependencies=[Depends(require_key)])
def reject(action_id: str, body: ApproveIn):
    return service().reject(action_id, body.approver)


@app.post("/kill", dependencies=[Depends(require_key)])
def kill(body: KillIn):
    return service().kill(body.actor, body.reason)


@app.post("/resume", dependencies=[Depends(require_key)])
def resume(body: ResumeIn):
    r = service().resume(body.admin_token, body.actor)
    if not r["resumed"]:
        raise HTTPException(403, "admin token rejected")
    return r
