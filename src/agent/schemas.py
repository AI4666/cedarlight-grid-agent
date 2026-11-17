"""Strict Pydantic v2 schemas for the CedarLight agent (src/agent/schemas.py).

All models forbid extra fields so malformed or injected data fails validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=True)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Enums ----------
class ValidationStatus(str, Enum):
    VALID = "valid"
    SUSPECT = "suspect"
    REJECTED = "rejected"


class Entailment(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ActionTier(int, Enum):
    READ_ONLY = 0
    LOW_IMPACT_WRITE = 1
    APPROVAL_REQUIRED = 2
    FORBIDDEN = 3


class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class InferenceMethod(str, Enum):
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    RESOLUTION = "resolution"
    CSP = "csp"
    SEARCH = "search"
    BAYES_NET = "bayes_net"
    HMM = "hmm"
    KALMAN = "kalman"
    ML_MODEL = "ml_model"
    RL_POLICY = "rl_policy"


# ---------- 1. PERCEPTS ----------
class Percept(Strict):
    percept_id: str = Field(default_factory=lambda: _id("pcp"))
    source: Literal["meter", "weather", "asset_condition", "maintenance_window",
                    "crew", "operator", "tool_result"]
    entity_id: str = Field(min_length=1, description="e.g. feeder F03, transformer T07")
    timestamp: datetime
    values: dict[str, float | int | str | bool]
    quality: float = Field(ge=0.0, le=1.0, description="sensor/data quality 0..1")
    validation_status: ValidationStatus = ValidationStatus.VALID
    validation_notes: list[str] = Field(default_factory=list)


# ---------- 2. PROPOSITIONS ----------
class Proposition(Strict):
    prop_id: str = Field(default_factory=lambda: _id("prp"))
    predicate: str = Field(min_length=1, description="e.g. Overloaded, HeatAdvisory")
    args: list[str] = Field(default_factory=list)
    truth: bool = True
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    features: dict[str, float] = Field(default_factory=dict, description="model-ready features")
    derived_from: list[str] = Field(min_length=1, description="percept_ids")
    contradicts: list[str] = Field(default_factory=list, description="prop_ids in conflict")


# ---------- 3. KNOWLEDGE BASE ----------
class KnowledgeFact(Strict):
    fact_id: str = Field(default_factory=lambda: _id("fct"))
    predicate: str
    args: list[str] = Field(default_factory=list)
    truth: bool = True
    provenance: str = Field(min_length=1, description="source doc, percept, or model version")
    valid_from: datetime = Field(default_factory=_now)
    valid_to: datetime | None = None


class KnowledgeRule(Strict):
    rule_id: str = Field(default_factory=lambda: _id("rul"))
    name: str
    form: Literal["horn", "cnf", "fol"]
    body: list[str] = Field(description="antecedent literals (Horn) or clauses")
    head: str | None = Field(default=None, description="consequent literal for Horn rules")
    hard_constraint: bool = False
    provenance: str
    version: str = "1"


# ---------- 4. INFERENCE ----------
class InferenceStep(Strict):
    step: int = Field(ge=1)
    method: InferenceMethod
    inputs: list[str] = Field(description="fact/prop/percept ids")
    rule_or_model: str
    output: str
    note: str = Field(default="", max_length=300, description="concise rationale, not chain-of-thought")


class InferenceTrace(Strict):
    trace_id: str = Field(default_factory=lambda: _id("trc"))
    conclusion: str
    steps: list[InferenceStep] = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    model_versions: dict[str, str] = Field(default_factory=dict)


# ---------- 5. ENTAILMENT ----------
class EntailmentResult(Strict):
    query: str
    status: Entailment
    supporting_ids: list[str] = Field(default_factory=list)
    conflicting_ids: list[str] = Field(default_factory=list)
    model_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="informational only; never upgrades status to ENTAILED")
    rationale: str = Field(max_length=500)


# ---------- 6. ACTION ----------
class ActionProposal(Strict):
    action_id: str = Field(default_factory=lambda: _id("act"))
    idempotency_key: str
    tool: str = Field(description="must be on the allow-list")
    arguments: dict[str, Any] = Field(default_factory=dict)
    tier: ActionTier
    utility: float | None = None
    risk: float = Field(ge=0.0, le=1.0, default=0.0)
    entailment: list[EntailmentResult] = Field(default_factory=list)
    approval: ApprovalState = ApprovalState.NOT_REQUIRED
    justification: str = Field(max_length=500)


class ToolResult(Strict):
    action_id: str
    tool: str
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = Field(ge=0.0, default=0.0)
    timestamp: datetime = Field(default_factory=_now)


class AuditEvent(Strict):
    event_id: str = Field(default_factory=lambda: _id("aud"))
    timestamp: datetime = Field(default_factory=_now)
    stage: Literal["percepts", "propositions", "knowledge_base", "inference",
                   "entailment", "action", "policy", "kill_switch"]
    actor: str = Field(description="module or human user id")
    summary: str = Field(max_length=500)
    refs: list[str] = Field(default_factory=list, description="ids of related objects")
    run_id: str
    config_hash: str
    seed: int


# ---------- orchestration results (what the UI/API exposes: evidence records, never hidden reasoning) ----------
class PolicyDecision(Strict):
    action_id: str
    tool: str
    allowed: bool
    requires_approval: bool = False
    blocked: bool = False
    escalate: bool = False
    duplicate: bool = False
    reasons: list[str] = Field(default_factory=list)


class ConflictRecord(Strict):
    conflict_id: str = Field(default_factory=lambda: _id("cfl"))
    kind: Literal["observation", "protected_static"]
    atom: str
    fact_ids: list[str] = Field(default_factory=list)
    note: str = Field(default="", max_length=300)


class CycleResult(Strict):
    run_id: str
    cycle: int
    iterations: int
    status: Literal["completed", "halted_kill_switch", "max_iterations", "no_input"]
    percepts: list[Percept] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    propositions: list[Proposition] = Field(default_factory=list)
    kb_updates: list[str] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    traces: list[InferenceTrace] = Field(default_factory=list)
    entailments: list[EntailmentResult] = Field(default_factory=list)
    proposals: list[ActionProposal] = Field(default_factory=list)
    decisions: list[PolicyDecision] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list)
    audit_ids: list[str] = Field(default_factory=list)
