# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    dag_encoder: str
    use_preference: bool
    use_action_mask: bool
    hierarchical_uav: bool
    delivery_aware: bool
    category: str
    contact_feature: bool | None = None
    predictive_admission: bool | None = None


VARIANTS = {
    "ablation_completion_only": VariantSpec(
        "ablation_completion_only", "Compute-only training model", "gat",
        True, True, True, True, "ablation",
    ),
    "uamco": VariantSpec(
        "uamco",
        "UAMCO-DAG",
        "gat",
        True,
        True,
        True,
        True,
        "proposed",
    ),
    "ablation_no_dag": VariantSpec(
        "ablation_no_dag",
        "w/o DAG edges",
        "none",
        True,
        True,
        True,
        True,
        "ablation",
    ),
    "ablation_no_hierarchy": VariantSpec(
        "ablation_no_hierarchy",
        "Synchronous UAV control",
        "gat",
        True,
        True,
        False,
        True,
        "ablation",
    ),
    "ablation_no_preference": VariantSpec(
        "ablation_no_preference",
        "w/o Preference",
        "gat",
        False,
        True,
        True,
        True,
        "ablation",
    ),
    "ablation_no_mask": VariantSpec(
        "ablation_no_mask",
        "w/o Feasibility Mask",
        "gat",
        True,
        False,
        True,
        True,
        "ablation",
    ),
    "ablation_no_delivery": VariantSpec(
        "ablation_no_delivery",
        "w/o Contact Prediction",
        "gat",
        True,
        True,
        True,
        False,
        "ablation",
    ),
    "ablation_no_contact_feature": VariantSpec(
        "ablation_no_contact_feature", "w/o Contact Feature", "gat",
        True, True, True, True, "ablation",
        contact_feature=False, predictive_admission=True,
    ),
    "ablation_no_contact_mask": VariantSpec(
        "ablation_no_contact_mask", "w/o Predictive Admission", "gat",
        True, True, True, True, "ablation",
        contact_feature=True, predictive_admission=False,
    ),
    "ablation_no_dag_no_delivery": VariantSpec(
        "ablation_no_dag_no_delivery", "w/o DAG Edges and Contact Prediction", "none",
        True, True, True, False, "ablation",
        contact_feature=False, predictive_admission=False,
    ),
}

COMPARISON_VARIANTS = ("uamco",)
ABLATION_VARIANTS = (
    "uamco",
    "ablation_no_dag",
    "ablation_no_hierarchy",
    "ablation_no_delivery",
    "ablation_completion_only",
)

# New diagnostic matrix; keep the historical suite definition unchanged.
DIAGNOSTIC_VARIANTS = (
    "uamco", "ablation_no_dag", "ablation_no_hierarchy",
    "ablation_no_delivery", "ablation_no_contact_feature",
    "ablation_no_contact_mask", "ablation_no_dag_no_delivery",
)

# These are intervention descriptions, not expected performance rankings.
# Every variant is evaluated with the same actual dependency/return physics.
ABLATION_CONTRACTS = {
    "uamco": "Full bidirectional DAG encoder and macro-timescale UAV controller.",
    "ablation_no_dag": (
        "Remove dependency edges only. Keep identical node features, node masks, "
        "selected-node context, encoder parameter shapes and readout."
    ),
    "ablation_no_hierarchy": (
        "Make UAV decisions every simulator step instead of every macro interval; "
        "keep the same actor architecture and physical elapsed-time accounting. "
        "This tests temporal abstraction, not removal of all hierarchical architecture."
    ),
    "ablation_no_delivery": (
        "Hide predicted contact duration and use current-connectivity masks; "
        "retain measured geometry/rate features and actual delivery physics."
    ),
    "ablation_completion_only": (
        "Train with compute-completion dependencies and sink-only return, then "
        "evaluate with full dependency transfers and delivery. This is a "
        "training-physics ablation, not an architecture ablation."
    ),
    "ablation_no_contact_feature": (
        "Hide predicted contact duration only; retain predictive admission and "
        "all actual dependency transfers and delivery physics."
    ),
    "ablation_no_contact_mask": (
        "Use current-connectivity admission only; retain predicted contact "
        "duration features and all actual dependency transfers and delivery physics."
    ),
    "ablation_no_dag_no_delivery": (
        "Joint intervention: remove DAG edges, hide predicted contact duration "
        "and use current-connectivity admission; retain identical node features, "
        "parameter shapes, macro control and actual delivery physics."
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant: {name}") from exc


def model_kind_choices() -> tuple[str, ...]:
    return tuple(VARIANTS)


def macro_interval(name: str, configured_interval: int) -> int:
    return int(configured_interval) if get_variant(name).hierarchical_uav else 1
