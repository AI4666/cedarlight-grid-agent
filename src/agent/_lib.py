"""Locates the reasoning libraries built in earlier phases (03_logic, 04_uncertainty, 02_simulation) via
pathlib, so the package works from any working directory. Nothing here reads secrets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

for _d in ("03_logic", "04_uncertainty", "02_simulation"):
    _p = str(ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bayes_net  # noqa: E402,F401
import decision  # noqa: E402,F401
import logic_engine  # noqa: E402,F401
import sim_algos  # noqa: E402,F401
import temporal  # noqa: E402,F401
