"""Append-only audit trail (JSON Lines) with secret redaction. Every stage of the reasoning cycle emits events."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from .config import secret_values
from .schemas import AuditEvent

_KEY_PATTERNS = [re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
                 re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*\S+")]


def redact(text: str) -> str:
    for v in secret_values():
        text = text.replace(v, "[REDACTED]")
    for pat in _KEY_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class AuditLog:
    def __init__(self, run_id: str, config_hash: str, seed: int, path: Path | None = None):
        self.run_id, self.config_hash, self.seed, self.path = run_id, config_hash, seed, path
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, actor: str, summary: str, refs: list[str] | None = None) -> AuditEvent:
        ev = AuditEvent(stage=stage, actor=actor, summary=redact(summary)[:500], refs=list(refs or [])[:50],
                        run_id=self.run_id, config_hash=self.config_hash, seed=self.seed)
        with self._lock:
            self.events.append(ev)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(json.loads(ev.model_dump_json())) + "\n")
        return ev

    def by_stage(self, stage: str) -> list[AuditEvent]:
        return [e for e in self.events if e.stage == stage]
