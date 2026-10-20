# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .statistics import mean_ci95


IDENTITY_KEYS = {
    "method",
    "channel",
    "preference",
    "load_multiplier",
}

FRONTIER_SECTIONS = (
    "channels",
    "baseline_channels",
    "learned_baseline_channels",
)

POINT_SECTIONS = (
    "comparison_points",
    "baselines",
    "learned_baselines",
)


def _validate_seed_payloads(payloads: list[dict[str, Any]]) -> list[int]:
    if not payloads:
        raise ValueError("No seed results to aggregate")
    seeds = [int(payload["seed"]) for payload in payloads]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Aggregation requires independent, unique seeds")
    for key in ("workflow_fingerprint", "workflow_source_fingerprint", "implementation_fingerprint"):
        # All-None legacy unit fixtures are accepted; mixing legacy/new or
        # genuinely different identities is never silently pooled.
        if len({payload.get(key) for payload in payloads}) != 1:
            raise ValueError(f"Seed results have mixed {key} identities")
    evaluation_ids = {
        (payload.get("workflow_evaluation") or {}).get("fingerprint")
        for payload in payloads
    }
    if len(evaluation_ids) != 1:
        raise ValueError("Seed results have mixed workflow evaluation dataset identities")
    return seeds


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in rows[0].items()
        if key in IDENTITY_KEYS
    }
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in IDENTITY_KEYS and isinstance(value, (int, float))
        }
    )
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = mean_ci95(values)
    return summary


def _group(
    payloads: list[dict[str, Any]],
    section: str,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for payload in payloads:
        for row in payload.get(section, []):
            identity = tuple(row[key] for key in keys)
            grouped.setdefault(identity, []).append(row)
    return [_summarize(grouped[key]) for key in sorted(grouped, key=str)]


def _metric_value(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if isinstance(value, dict) and "mean" in value:
        value = value["mean"]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Pareto/HV metric {key} must be finite, got {result}")
    return result


def _method_rows(payload: dict[str, Any], method: str) -> list[dict[str, Any]]:
    for section in FRONTIER_SECTIONS:
        rows = [
            row
            for row in payload.get(section, [])
            if row.get("method") == method
        ]
        if rows:
            return rows
    rows: list[dict[str, Any]] = []
    for section in POINT_SECTIONS:
        rows.extend(
            row
            for row in payload.get(section, [])
            if row.get("method") == method
        )
    return rows


def _pareto_front_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            _metric_value(item, "effective_delay"),
            _metric_value(item, "energy_per_created_j"),
        ),
    ):
        duplicate = any(
            np.isclose(
                _metric_value(row, "effective_delay"),
                _metric_value(other, "effective_delay"),
                atol=1e-8,
                rtol=1e-7,
            )
            and np.isclose(
                _metric_value(row, "energy_per_created_j"),
                _metric_value(other, "energy_per_created_j"),
                atol=1e-8,
                rtol=1e-7,
            )
            for other in unique
        )
        if not duplicate:
            unique.append(row)
    front: list[dict[str, Any]] = []
    for candidate in unique:
        dominated = any(
            _metric_value(other, "effective_delay")
            <= _metric_value(candidate, "effective_delay")
            and _metric_value(other, "energy_per_created_j")
            <= _metric_value(candidate, "energy_per_created_j")
            and (
                _metric_value(other, "effective_delay")
                < _metric_value(candidate, "effective_delay")
                or _metric_value(other, "energy_per_created_j")
                < _metric_value(candidate, "energy_per_created_j")
            )
            for other in unique
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: _metric_value(item, "effective_delay"))


def _hypervolume_rows(
    rows: list[dict[str, Any]],
    reference: tuple[float, float],
) -> tuple[float, int]:
    """Clipped minimization HV and global (not reference-clipped) front size.

    A point on or outside either reference coordinate contributes no area.
    Validate all supplied objectives, including excluded points: silently
    dropping NaN/Inf would turn a failed measurement into a valid score.
    This data-only implementation is also used by checkpoint evaluation.
    """
    if len(reference) != 2:
        raise ValueError("HV reference must contain exactly [delay, energy]")
    reference = (float(reference[0]), float(reference[1]))
    if any(not np.isfinite(value) or value <= 0.0 for value in reference):
        raise ValueError("HV reference values must be finite and positive")
    front = _pareto_front_rows(rows)
    contributing_front = _pareto_front_rows([
        row for row in rows
        if _metric_value(row, "effective_delay") < reference[0]
        and _metric_value(row, "energy_per_created_j") < reference[1]
    ])
    hv = 0.0
    for index, point in enumerate(contributing_front):
        next_delay = (
            _metric_value(contributing_front[index + 1], "effective_delay")
            if index + 1 < len(contributing_front)
            else reference[0]
        )
        width = max(next_delay - _metric_value(point, "effective_delay"), 0.0)
        height = max(reference[1] - _metric_value(point, "energy_per_created_j"), 0.0)
        hv += width * height
    if not np.isfinite(hv):
        raise ValueError("HV area must be finite; objective/reference magnitudes overflowed")
    return float(hv), len(front)


def aggregate_evaluations(
    result_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in result_paths
    ]
    seeds = _validate_seed_payloads(payloads)
    references = [
        tuple(float(value) for value in payload["hypervolume"]["reference"])
        for payload in payloads
    ]
    common_reference = references[0]
    if all(np.allclose(reference, common_reference) for reference in references):
        reference_payload: list[float] | list[list[float]] = [
            float(value) for value in common_reference
        ]
    else:
        reference_payload = [
            [float(value) for value in reference] for reference in references
        ]
    terrain_rows = []
    for payload in payloads:
        for terrain_mode, row in payload["terrain_ablation"].items():
            terrain_rows.append({"method": terrain_mode, **row})
    hypervolume = np.asarray(
        [
            _hypervolume_rows(_method_rows(payload, "uamco"), reference)[0]
            for payload, reference in zip(payloads, references)
        ],
        dtype=np.float64,
    )

    method_samples: dict[str, list[tuple[float, int]]] = {}
    for payload in payloads:
        reference = tuple(payload["hypervolume"]["reference"])
        methods = {
            row["method"]
            for section in FRONTIER_SECTIONS + POINT_SECTIONS
            for row in payload.get(section, [])
        }
        for method in methods:
            rows = _method_rows(payload, method)
            if not rows:
                continue
            hv_value, front_size = _hypervolume_rows(rows, reference)
            method_samples.setdefault(method, []).append((hv_value, front_size))

    method_hypervolume = {}
    for method, samples in method_samples.items():
        hv_summary = mean_ci95(item[0] for item in samples)
        front_summary = mean_ci95(item[1] for item in samples)
        method_hypervolume[method] = {
            **hv_summary,
            "front_size_mean": front_summary["mean"],
            "front_size_std": front_summary["std"],
            "front_size_n": front_summary["n"],
            "front_size_ci95": front_summary["ci95"],
        }
    method_hypervolume_ranking = [
        {"method": method, **stats}
        for method, stats in sorted(
            method_hypervolume.items(),
            key=lambda item: item[1]["mean"],
            reverse=True,
        )
    ]

    paired_comparisons: dict[str, Any] = {}
    for baseline in ("official_mappo", "official_happo"):
        hv_difference: list[float] = []
        delay_difference: list[float] = []
        energy_difference: list[float] = []
        for payload in payloads:
            reference = tuple(payload["hypervolume"]["reference"])
            ours_rows = _method_rows(payload, "uamco")
            baseline_rows = _method_rows(payload, baseline)
            if not ours_rows or not baseline_rows:
                continue
            ours_hv, _ = _hypervolume_rows(ours_rows, reference)
            baseline_hv, _ = _hypervolume_rows(baseline_rows, reference)
            hv_difference.append(ours_hv - baseline_hv)
            delay_difference.append(
                min(_metric_value(row, "effective_delay") for row in ours_rows)
                - min(
                    _metric_value(row, "effective_delay")
                    for row in baseline_rows
                )
            )
            energy_difference.append(
                min(_metric_value(row, "energy_per_created_j") for row in ours_rows)
                - min(
                    _metric_value(row, "energy_per_created_j")
                    for row in baseline_rows
                )
            )
        if hv_difference:
            paired_comparisons[baseline] = {
                "uamco_minus_baseline_hypervolume": mean_ci95(hv_difference),
                "uamco_minus_baseline_minimum_delay": mean_ci95(delay_difference),
                "uamco_minus_baseline_minimum_energy": mean_ci95(energy_difference),
            }

    aggregate = {
        "seeds": seeds,
        "workflow_source_fingerprint": payloads[0].get("workflow_source_fingerprint"),
        "workflow_fingerprint": payloads[0].get("workflow_fingerprint"),
        "implementation_fingerprint": payloads[0].get("implementation_fingerprint"),
        "workflow_evaluation_fingerprint": (payloads[0].get("workflow_evaluation") or {}).get("fingerprint"),
        "reference": reference_payload,
        "channels": _group(payloads, "channels", ("channel",)),
        "greedy_channels": _group(
            payloads,
            "greedy_channels",
            ("channel",),
        ),
        "comparison_points": _group(
            payloads,
            "comparison_points",
            ("method",),
        ),
        "baseline_channels": _group(
            payloads,
            "baseline_channels",
            ("method", "channel"),
        ),
        "baselines": _group(payloads, "baselines", ("method",)),
        "learned_baseline_channels": _group(
            payloads,
            "learned_baseline_channels",
            ("method", "channel"),
        ),
        "learned_baselines": _group(
            payloads,
            "learned_baselines",
            ("method",),
        ),
        "stress": _group(
            payloads,
            "stress",
            ("method", "load_multiplier"),
        ),
        "terrain_ablation": _group(
            [{"terrain": terrain_rows}],
            "terrain",
            ("method",),
        ),
        "semantic_score": {
            key: {
                "mean": float(
                    np.mean([payload["semantic_score"][key] for payload in payloads])
                ),
                "std": float(
                    np.std(
                        [payload["semantic_score"][key] for payload in payloads],
                        ddof=1,
                    )
                )
                if len(payloads) > 1
                else 0.0,
            }
            for key in payloads[0]["semantic_score"]
        },
        "hypervolume": mean_ci95(hypervolume),
        "method_hypervolume": method_hypervolume,
        "method_hypervolume_ranking": method_hypervolume_ranking,
        "paired_comparisons": paired_comparisons,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
