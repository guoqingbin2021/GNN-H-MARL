"""Content identity for the numerical implementation behind an experiment.

Configuration and dataset contracts alone do not detect a corrected simulator
or learner. Keep the core numerical sources in the identity, while edits to
documentation, tests and launch scripts do not invalidate trained policies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parent
NUMERICAL_MODULES = (
    "contracts.py",
    "config.py",
    "data.py",
    "workflows.py",
    "env.py",
    "physics.py",
    "queueing.py",
    "terrain.py",
    "model.py",
    "variants.py",
    "ppo.py",
    "baselines.py",
    "official_baselines.py",
    "sota_baselines.py",
    "evaluate.py",
    "aggregate.py",
    "ablation.py",
    "communication_eval.py",
)


def implementation_manifest() -> dict[str, Any]:
    """Read the current core sources without reusing a stale process cache.

    Trainers must capture this at initialization and store that captured value
    in every checkpoint; a save-time digest could falsely certify code edited
    while an older in-memory implementation was still training.
    """
    sources = {
        name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
        for name in NUMERICAL_MODULES
    }
    body = {"schema": 1, "numerical_sources": sources}
    fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**body, "fingerprint": fingerprint}


def implementation_fingerprint() -> str:
    return str(implementation_manifest()["fingerprint"])
