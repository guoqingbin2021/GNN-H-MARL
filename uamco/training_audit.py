"""Evidence-only diagnostics. Never used in reward, gradients or action selection."""
from __future__ import annotations

import numpy as np

from .evaluate import _hypervolume, _semantic_score


def _audit_status(episodes: int, checks: dict[str, bool]) -> str:
    """Label evidence strength without turning diagnostics into convergence claims."""
    if episodes < 30:
        return "SMOKE_ONLY"
    return (
        "DESCRIPTIVE_CHECKS_PASS"
        if checks and all(checks.values())
        else "DESCRIPTIVE_CHECKS_FAIL"
    )


def audit_history(history, reference=(16.0, 1000.0)):
    n = len(history)
    window = max(1, min(10, n // 3))
    changes = {}
    for channel in range(5):
        key = f"ch{channel}"
        first = [x["channels"][key] for x in history[:window] if key in x.get("channels", {})]
        last = [x["channels"][key] for x in history[-window:] if key in x.get("channels", {})]
        if not first or not last:
            continue
        changes[key] = {metric: float(np.mean([r[metric] for r in last]) - np.mean([r[metric] for r in first]))
                        for metric in ["effective_delay", "energy_per_created_j", "uav_ratio", "reward"]}
    hvs = [_hypervolume(list(r["channels"].values()), reference) for r in history if len(r.get("channels", {})) == 5]
    tail = [r["reward"] for r in history[-window:]]
    final = list(history[-1].get("channels", {}).values()) if history else []
    final = [{**row, "remote_ratio": row.get("remote_ratio", row["rsu_ratio"] + row["uav_ratio"])} for row in final]
    checks = {
        "ch0_ch1_training_delay_decreases": all(changes.get(f"ch{i}", {}).get("effective_delay", 1.) < 0 for i in [0, 1]),
        "ch3_ch4_training_energy_decreases": all(changes.get(f"ch{i}", {}).get("energy_per_created_j", 1.) < 0 for i in [3, 4]),
        "training_hypervolume_advances": bool(hvs and np.mean(hvs[-window:]) > np.mean(hvs[:window])),
    }
    return {"status": _audit_status(n, checks),
            "episodes": n, "window": window, "changes": changes, "checks": checks,
            "reward_tail_std": float(np.std(tail)) if tail else None,
            "reward_tail_slope": float(np.dot(np.arange(window) - (window-1)/2, np.asarray(tail)-np.mean(tail)) /
                                       max(np.sum((np.arange(window)-(window-1)/2)**2), 1.)) if tail else None,
            "training_hypervolume": hvs,
            "final_channel_semantics": _semantic_score(final) if len(final) == 5 else {},
            "note": "SMOKE_ONLY means execution-scale evidence, not convergence. DESCRIPTIVE_CHECKS_PASS/FAIL summarizes only the listed training-history checks. Claim validation additionally requires matched held-out evaluation, independent paired seeds and failure/censoring checks; a stable reward alone is not convergence proof."}
