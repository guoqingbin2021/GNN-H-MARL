# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import torch

from .env import VectorOffloadEnv


def heuristic_dispatch(
    name: str,
    env: VectorOffloadEnv,
    preference: torch.Tensor,
) -> torch.Tensor:
    device = env.device
    if name == "random_feasible":
        region_mask = env.observe()["demand_region_mask"].float()
        weights = torch.cat(
            [region_mask, torch.ones((env.b, 1), device=device)],
            dim=1,
        )
        return torch.multinomial(weights, env.u, replacement=True)
    if name == "greedy_delay":
        obs = env.observe()
        region = obs["demand_regions"]
        score = (
            region[..., 2]
            + region[..., 3]
            + region[..., 4]
            + region[..., 5]
            + region[..., 6]
            + region[..., 7]
        ).masked_fill(~obs["demand_region_mask"], -1e9)
        target_count = min(env.u, env.region_n)
        targets = score.topk(target_count, dim=1).indices
        if target_count < env.u:
            padding = torch.full(
                (env.b, env.u - target_count),
                env.region_n,
                device=device,
                dtype=torch.long,
            )
            return torch.cat([targets, padding], dim=1)
        return targets
    return torch.full(
        (env.b, env.u),
        env.region_n,
        dtype=torch.long,
        device=device,
    )


def heuristic_vehicle_actions(
    name: str,
    env: VectorOffloadEnv,
    preference: torch.Tensor,
) -> torch.Tensor:
    device = env.device
    if name == "local_only":
        return torch.zeros((env.b, env.v), dtype=torch.long, device=device)
    if name == "nearest_rsu":
        _, feasible, _ = env.link_state()
        distance = torch.linalg.norm(
            env.vehicle_xy.unsqueeze(2)
            - env.rsu_xy.view(1, 1, env.r, 2),
            dim=-1,
        )
        distance = distance.masked_fill(~feasible[..., : env.r], 1e9)
        best_distance, best_rsu = distance.min(dim=-1)
        actions = torch.where(
            best_distance < 1e8,
            best_rsu + 1,
            torch.zeros_like(best_rsu),
        )
        return actions

    costs = env.estimate_action_costs()
    if name == "greedy_delay":
        score = costs[..., 0]
    elif name == "greedy_energy":
        score = costs[..., 1]
    elif name == "weighted_greedy":
        score = (
            preference[:, None, None, 0:1]
            * costs[..., 0:1]
            / float(env.cfg.reward.delay_reference)
            + preference[:, None, None, 1:2]
            * costs[..., 1:2]
            / float(env.cfg.reward.energy_reference_j)
        ).squeeze(-1)
    elif name == "random_feasible":
        weights = env.action_mask().float()
        return torch.multinomial(
            weights.reshape(-1, env.action_n),
            1,
        ).reshape(env.b, env.v)
    else:
        raise ValueError(f"Unknown heuristic baseline: {name}")

    return score.argmin(dim=-1)


def heuristic_actions(
    name: str,
    env: VectorOffloadEnv,
    preference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        heuristic_vehicle_actions(name, env, preference),
        heuristic_dispatch(name, env, preference),
    )
