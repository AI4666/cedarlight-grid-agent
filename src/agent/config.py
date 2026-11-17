"""Configuration: non-secret parameters come from YAML; secrets ONLY from environment variables."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "agent.yaml"

# Names of environment variables that hold secrets. Their VALUES must never be logged, hashed or returned.
SECRET_ENV_VARS = ("CEDARLIGHT_API_KEY", "CEDARLIGHT_LLM_API_KEY", "CEDARLIGHT_ADMIN_TOKEN")


def load_config(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():     # e.g. {"run.seed": 7, "kill_switch.file_flag": "..."}
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    assert_no_secrets(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable fingerprint of the non-secret configuration, recorded in every audit event."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def assert_no_secrets(cfg: dict[str, Any]) -> None:
    """Fail closed if a secret value from the environment shows up inside the config."""
    blob = json.dumps(cfg, default=str)
    for name in SECRET_ENV_VARS:
        val = os.environ.get(name)
        if val and len(val) >= 8 and val in blob:
            raise ValueError(f"secret from {name} found in configuration; secrets belong in the environment only")


def secret_values() -> list[str]:
    return [v for n in SECRET_ENV_VARS if (v := os.environ.get(n)) and len(v) >= 4]


def admin_token_ok(candidate: str | None) -> bool:
    """Constant-time check of the admin token. If no token is configured, admin actions are DISABLED."""
    expected = os.environ.get("CEDARLIGHT_ADMIN_TOKEN")
    if not expected or not candidate:
        return False
    return hmac.compare_digest(candidate.encode(), expected.encode())


def resolve(cfg: dict[str, Any], rel: str) -> Path:
    """Portable path resolution relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
