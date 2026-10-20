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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate import _summarize, _validate_seed_payloads
from .config import Config
from .contracts import implementation_fingerprint
from .evaluate import (
    _hypervolume,
    _hypervolume_reference,
    _pareto_front,
    _semantic_score,
    evaluate_learned,
)
from .ppo import load_policy, resolve_device, set_seed
from .statistics import holm_adjust, paired_signed_rank
from .variants import ABLATION_CONTRACTS, ABLATION_VARIANTS, get_variant
from .workflows import workflow_fingerprint


def evaluate_ablation_seed(
    config: Config,
    checkpoints: dict[str, str | Path],
    output_path: str | Path,
    seed: int,
    device_arg: str = "auto",
) -> Path:
    implementation = implementation_fingerprint()
    set_seed(seed)
    device = resolve_device(device_arg)
    episodes = int(config.evaluation.episodes)
    anchors = [list(map(float, anchor)) for anchor in config.reward.anchors]
    variants = []
    all_channels = []
    for method in ABLATION_VARIANTS:
        model = load_policy(config, checkpoints[method], device)
        if model.kind != method:
            raise ValueError(f"Expected {method} checkpoint, got {model.kind}")
        channels = []
        for channel, anchor in enumerate(anchors):
            metrics = evaluate_learned(
                config,
                model,
                anchor,
                episodes,
                seed + 100,
                device,
                deterministic=bool(config.evaluation.deterministic),
            )
            row = {
                "method": method,
                "label": get_variant(method).label,
                "intervention": ABLATION_CONTRACTS[method],
                "channel": channel,
                "preference": anchor,
                **metrics,
            }
            channels.append(row)
            all_channels.append(row)
        variants.append(
            {
                "method": method,
                "label": get_variant(method).label,
                "channels": channels,
                "semantic_score": _semantic_score(channels),
            }
        )

    reference = _hypervolume_reference(config, all_channels)
    for variant in variants:
        channels = variant["channels"]
        variant["summary"] = {
            "delay_endpoint": channels[0]["effective_delay"],
            "energy_endpoint": channels[-1]["energy_per_created_j"],
            "mean_completion_rate": float(
                np.mean([row["completion_rate"] for row in channels])
            ),
            "mean_constraint_violation_rate": float(
                np.mean([row["constraint_violation_rate"] for row in channels])
            ),
            "mean_invalid_action_rate": float(
                np.mean([row["invalid_action_rate"] for row in channels])
            ),
            "delay_channel_uav_ratio": channels[0]["uav_ratio"],
            "semantic_score": variant["semantic_score"]["overall"],
            "pareto_points": len(_pareto_front(channels)),
            "hypervolume": _hypervolume(channels, reference),
        }
        for key in (
            "upload_interruptions", "return_retries", "return_deferrals",
            "delivered_results", "uav_flight_energy_j", "timeout_rate",
        ):
            if all(key in row for row in channels):
                variant["summary"][f"mean_{key}"] = float(
                    np.mean([row[key] for row in channels])
                )

    payload = {
        "seed": int(seed),
        "workflow_fingerprint": workflow_fingerprint(config),
        "implementation_fingerprint": implementation,
        "episodes": episodes,
        "reference": reference,
        "evaluation_physics": "full_dependency_transfers_and_delivery",
        "interventions": ABLATION_CONTRACTS,
        "variants": variants,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def aggregate_ablation_seeds(
    result_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in result_paths
    ]
    seeds = _validate_seed_payloads(payloads)
    reference = payloads[0].get("reference")
    if any(not np.allclose(item.get("reference"), reference) for item in payloads):
        raise ValueError("Ablation hypervolume requires a common fixed reference")
    aggregate_variants = []
    for method in ABLATION_VARIANTS:
        rows = [
            next(
                variant
                for variant in payload["variants"]
                if variant["method"] == method
            )
            for payload in payloads
        ]
        channel_summaries = []
        for channel in range(len(rows[0]["channels"])):
            channel_summaries.append(
                _summarize([row["channels"][channel] for row in rows])
            )
        aggregate_variants.append(
            {
                "method": method,
                "label": get_variant(method).label,
                "intervention": ABLATION_CONTRACTS[method],
                "summary": _summarize([row["summary"] for row in rows]),
                "channels": channel_summaries,
            }
        )
    paired = {}
    for method in ABLATION_VARIANTS:
        if method == "uamco":
            continue
        differences = []
        for item in payloads:
            summaries = {
                row["method"]: row["summary"] for row in item["variants"]
            }
            differences.append(
                summaries["uamco"]["hypervolume"] - summaries[method]["hypervolume"]
            )
        statistics = paired_signed_rank(differences)
        paired[method] = {
            **statistics,
            "uamco_minus_ablation_hypervolume": statistics["difference_summary"],
            "seed_differences": [{"seed": seed, "full_minus_ablation_hypervolume": float(delta)}
                                 for seed, delta in zip(seeds, differences)],
            "full_model_wins": statistics["wins"],
            "interpretation": "Positive differences favor the full model; retain negative and null effects.",
        }
    corrected = holm_adjust([item["wilcoxon"]["p_uncorrected"] for item in paired.values()])
    for item, pvalue in zip(paired.values(), corrected):
        item["wilcoxon"]["p_holm"] = pvalue
        item["wilcoxon"]["holm_family_size"] = len(paired)
        item["wilcoxon"]["significant_holm_0_05"] = (
            bool(pvalue < 0.05) if len(seeds) >= 5 and pvalue is not None else None
        )
    payload = {
        "seeds": seeds,
        "reference": reference,
        "workflow_fingerprint": payloads[0].get("workflow_fingerprint"),
        "implementation_fingerprint": payloads[0].get("implementation_fingerprint"),
        "evaluation_physics": "full_dependency_transfers_and_delivery",
        "interventions": ABLATION_CONTRACTS,
        "variants": aggregate_variants,
        "paired_comparisons": paired,
        "statistical_plan": {
            "primary_outcome": "fixed-reference hypervolume",
            "unit": "independent training seed, paired across methods",
            "test": "two-sided Wilcoxon signed-rank; exact for distinct nonzero ranks up to 50 seeds, otherwise SciPy auto",
            "multiplicity": "Holm correction across the four prespecified full-model versus ablation contrasts",
            "minimum_seeds_for_significance_label": 5,
            "caution": "Smoke results do not establish significance or convergence. Inspect signed-rank symmetry assumptions and effect magnitude; a null result is not equivalence.",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def plot_ablation_aggregate(
    aggregate_path: str | Path,
    figure_dir: str | Path,
) -> None:
    payload = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    variants = payload["variants"]
    labels = [item["label"] for item in variants]
    directory = Path(figure_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("hypervolume", "Hypervolume"),
        ("semantic_score", "Semantic score"),
        ("mean_completion_rate", "Mean completion rate"),
        ("mean_invalid_action_rate", "Invalid action rate"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(variants))
    for axis, (key, ylabel) in zip(axes.flat, metrics):
        means = [item["summary"][key]["mean"] for item in variants]
        stds = [item["summary"][key]["std"] for item in variants]
        colors = ["#0072B2"] + ["#999999"] * (len(variants) - 1)
        axis.bar(x, means, yerr=stds, color=colors, capsize=3)
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "ablation_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / "ablation_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    channel_metrics = (
        ("effective_delay", "Effective delay"),
        ("energy_per_created_j", "Energy / task (J)"),
        ("remote_ratio", "Remote action ratio"),
    )
    for variant in variants:
        channels = variant["channels"]
        channel_x = np.arange(len(channels))
        for axis, (key, ylabel) in zip(axes, channel_metrics):
            axis.errorbar(
                channel_x,
                [row[key]["mean"] for row in channels],
                yerr=[row[key]["std"] for row in channels],
                marker="o",
                linewidth=1.2,
                capsize=2,
                label=variant["label"],
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Preference channel")
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(
        directory / "ablation_channel_semantics.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        directory / "ablation_channel_semantics.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
