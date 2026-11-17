"""STAGE 1 - PERCEPTS. Capture raw observations with source, timestamp, quality and validation status.

All external input is untrusted: it is type-checked, range-checked, size-limited and scanned for prompt-injection
patterns BEFORE it can become knowledge. Nothing here executes or interprets text as instructions."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from .schemas import Percept, ValidationStatus

SOURCES = {"meter", "weather", "asset_condition", "maintenance_window", "crew", "operator", "tool_result"}

INJECTION = re.compile(
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|safety)?\s*(rules|instructions|policy|constraints)"
    r"|disregard|system\s*prompt|you\s+are\s+now|override|jailbreak|developer\s+mode|bypass"
    r"|\b(call|execute|run|invoke)\s+[a-z_]{3,}|switch_breaker|shed_load|dispatch_crew|rm\s+-rf|drop\s+table",
    re.I)


class PerceptRejected(ValueError):
    """Raised for input that cannot even be represented as a Percept (malformed structure)."""


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise PerceptRejected(f"bad timestamp: {value!r}") from e
    else:
        raise PerceptRejected("timestamp missing or wrong type")
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def perceive(raw: Any, cfg: dict, now: datetime) -> Percept:
    if not isinstance(raw, dict):
        raise PerceptRejected("percept must be an object")
    missing = [k for k in ("source", "entity_id", "timestamp", "values") if k not in raw]
    if missing:
        raise PerceptRejected(f"missing fields: {missing}")
    if raw["source"] not in SOURCES:
        raise PerceptRejected(f"unknown source {str(raw['source'])[:30]!r}")
    if not isinstance(raw["entity_id"], str) or not (0 < len(raw["entity_id"]) <= 64) or not re.fullmatch(r"[A-Za-z0-9_\-]+", raw["entity_id"]):
        raise PerceptRejected("entity_id must be 1-64 chars of [A-Za-z0-9_-]")
    if not isinstance(raw["values"], dict):
        raise PerceptRejected("values must be an object")

    th, pc = cfg["thresholds"], cfg["perception"]
    ts = _parse_ts(raw["timestamp"])
    notes: list[str] = []
    status = ValidationStatus.VALID

    values: dict[str, float | int | str | bool] = {}
    for k, v in raw["values"].items():
        if not isinstance(k, str) or len(k) > 40:
            notes.append("dropped_bad_key")
            continue
        if isinstance(v, bool) or isinstance(v, str):
            values[k] = v if isinstance(v, bool) else v[: pc["max_text_len"]]
        elif isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                notes.append(f"{k}:non_finite_dropped")
            else:
                values[k] = v
        else:
            notes.append(f"{k}:unsupported_type_dropped")

    q = raw.get("quality")
    if q is None:
        quality, notes = 0.5, notes + ["quality_missing_default_0.5"]
    elif isinstance(q, bool) or not isinstance(q, (int, float)) or not (0.0 <= float(q) <= 1.0):
        raise PerceptRejected("quality must be a number in [0, 1]")
    else:
        quality = float(q)

    # --- validation (order = severity) ---
    for k, (lo, hi) in pc["ranges"].items():
        if k in values and isinstance(values[k], (int, float)) and not isinstance(values[k], bool) and not (lo <= values[k] <= hi):
            notes.append(f"{k}:out_of_range[{lo},{hi}]")
            status = ValidationStatus.REJECTED
    if not values:
        notes.append("no_usable_values")
        status = ValidationStatus.REJECTED

    if status != ValidationStatus.REJECTED:
        if ts < now - timedelta(hours=th["max_percept_age_hours"]):
            notes.append("stale"); status = ValidationStatus.SUSPECT
        if ts > now + timedelta(minutes=th["future_skew_minutes"]):
            notes.append("future_timestamp"); status = ValidationStatus.SUSPECT
        if quality < th["min_percept_quality"]:
            notes.append("low_quality"); status = ValidationStatus.SUSPECT
        if any(isinstance(v, str) and INJECTION.search(v) for v in values.values()):
            notes.append("possible_prompt_injection"); status = ValidationStatus.SUSPECT

    return Percept(source=raw["source"], entity_id=raw["entity_id"], timestamp=ts, values=values, quality=quality,
                   validation_status=status, validation_notes=notes[:10])


def ingest(raw_list: Any, cfg: dict, now: datetime) -> tuple[list[Percept], list[dict[str, Any]]]:
    """Validate a batch. Malformed items are returned as rejection records (never silently dropped, never raised)."""
    if not isinstance(raw_list, list):
        return [], [{"index": -1, "reason": "batch must be a list"}]
    accepted, rejected = [], []
    for i, raw in enumerate(raw_list[: cfg["perception"]["max_batch"]]):
        try:
            accepted.append(perceive(raw, cfg, now))
        except PerceptRejected as e:
            rejected.append({"index": i, "reason": str(e)[:200]})
    if len(raw_list) > cfg["perception"]["max_batch"]:
        rejected.append({"index": cfg["perception"]["max_batch"], "reason": "batch truncated"})
    return accepted, rejected
