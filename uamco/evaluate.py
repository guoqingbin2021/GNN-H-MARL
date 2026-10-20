# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .aggregate import _hypervolume_rows
from .baselines import heuristic_dispatch, heuristic_vehicle_actions
from .config import Config
from .env import VectorOffloadEnv
from .model import PreferencePolicy
from .official_baselines import evaluate_official_baseline
from .ppo import load_policy, resolve_device, set_seed
from .variants import macro_interval
from .workflows import load_workflow_bank, workflow_source_fingerprint
from .contracts import implementation_fingerprint


def _preference_batch(anchor: list[float], episodes: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(anchor, dtype=torch.float32, device=device).view(1, 2).expand(episodes, -1)


@torch.no_grad()
def evaluate_learned(
    config: Config,
    model: PreferencePolicy,
    anchor: list[float],
    episodes: int,
    seed: int,
    device: torch.device,
    load_multiplier: float = 1.0,
    terrain_enabled: bool = True,
    deterministic: bool | None = None,
    obs_filter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    scenario: Any | None = None,
) -> dict[str, float]:
    set_seed(seed)
    env = VectorOffloadEnv(
        config,
        episodes,
        device,
        seed=seed,
        load_multiplier=load_multiplier,
        terrain_enabled=terrain_enabled,
        workflow_split="eval",
    )
    obs = env.reset(_preference_batch(anchor, episodes, device))
    if scenario is not None:
        scenario.after_reset(env)
        obs = env.observe()
    if obs_filter is not None:
        obs = obs_filter(obs)
    greedy = (
        bool(config.evaluation.deterministic)
        if deterministic is None
        else bool(deterministic)
    )
    interval = macro_interval(model.kind, int(config.system.macro_steps))
    inference_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for step in range(int(config.system.episode_steps)):
        macro = step % interval == 0
        if macro:
            synchronize()
            started = time.perf_counter()
            uav_output = model.act_uavs(obs, deterministic=greedy)
            synchronize()
            inference_seconds += time.perf_counter() - started
            env.set_uav_dispatch(uav_output.dispatch)
            obs = env.observe()
            if obs_filter is not None:
                obs = obs_filter(obs)
        synchronize()
        started = time.perf_counter()
        vehicle_output = model.act_vehicles(obs, deterministic=greedy)
        synchronize()
        inference_seconds += time.perf_counter() - started
        obs, _, _, _ = env.step(vehicle_output.actions)
        if scenario is not None:
            scenario.after_step(env)
            obs = env.observe()
        if obs_filter is not None:
            obs = obs_filter(obs)
    metrics = env.summary_metrics()
    metrics["policy_seconds_per_batched_step"] = inference_seconds / int(config.system.episode_steps)
    metrics["peak_cuda_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    try:
        import psutil
        memory = psutil.Process().memory_info()
        metrics["process_rss_bytes"] = float(memory.rss)
        metrics["process_peak_working_set_bytes"] = float(getattr(memory, "peak_wset", memory.rss))
    except ImportError:
        pass
    return metrics


@torch.no_grad()
def evaluate_heuristic(
    config: Config,
    name: str,
    anchor: list[float],
    episodes: int,
    seed: int,
    device: torch.device,
    load_multiplier: float = 1.0,
    terrain_enabled: bool = True,
    obs_filter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    scenario: Any | None = None,
) -> dict[str, float]:
    set_seed(seed)
    env = VectorOffloadEnv(
        config,
        episodes,
        device,
        seed=seed,
        load_multiplier=load_multiplier,
        terrain_enabled=terrain_enabled,
        workflow_split="eval",
    )
    preference = _preference_batch(anchor, episodes, device)
    obs = env.reset(preference)
    if scenario is not None:
        scenario.after_reset(env)
        obs = env.observe()
    if obs_filter is not None:
        obs_filter(obs)
    for step in range(int(config.system.episode_steps)):
        macro = step % int(config.system.macro_steps) == 0
        if macro:
            env.set_uav_dispatch(heuristic_dispatch(name, env, preference))
        actions = heuristic_vehicle_actions(name, env, preference)
        env.step(actions)
        if scenario is not None:
            scenario.after_step(env)
    return env.summary_metrics()


def _semantic_score(channels: list[dict[str, Any]]) -> dict[str, float]:
    delay = np.asarray([item["effective_delay"] for item in channels])
    energy = np.asarray([item["energy_per_created_j"] for item in channels])
    remote = np.asarray([item["remote_ratio"] for item in channels])
    uav = np.asarray([item["uav_ratio"] for item in channels])

    def ordered(values: np.ndarray, increasing: bool) -> float:
        diffs = np.diff(values)
        scale = max(float(np.max(np.abs(values))), 1.0)
        tolerance = 1e-3 * scale
        return float(np.mean(diffs > tolerance if increasing else diffs < -tolerance))

    def trend(values: np.ndarray) -> float:
        if np.allclose(values, values[0]):
            return 0.0
        # Five-point Pearson correlation written explicitly.  This avoids a
        # platform-specific LAPACK crash observed in np.corrcoef on Windows
        # while preserving the exact statistic used in result files.
        index = np.arange(len(values), dtype=np.float64)
        centered_index = index - index.mean()
        centered_values = values.astype(np.float64) - values.mean()
        denominator = np.sqrt(
            np.dot(centered_index, centered_index)
            * np.dot(centered_values, centered_values)
        )
        if denominator <= 0.0:
            return 0.0
        return float(np.dot(centered_index, centered_values) / denominator)

    delay_order = ordered(delay, increasing=True)
    energy_order = ordered(energy, increasing=False)
    uav_order = ordered(uav, increasing=False)
    return {
        "delay_order_fraction": delay_order,
        "energy_order_fraction": energy_order,
        "uav_order_fraction": uav_order,
        "remote_order_fraction": ordered(remote, increasing=False),
        "delay_trend_correlation": trend(delay),
        "energy_trend_correlation": trend(energy),
        "uav_trend_correlation": trend(uav),
        "delay_endpoint_delta": float(delay[-1] - delay[0]),
        "energy_endpoint_reduction_fraction": float(
            (energy[0] - energy[-1]) / max(abs(float(energy[0])), 1e-6)
        ),
        "uav_endpoint_reduction": float(uav[0] - uav[-1]),
        "overall": float(np.mean([delay_order, energy_order, uav_order])),
    }


def _pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for point in sorted(
        points,
        key=lambda item: (
            float(item["effective_delay"]),
            float(item["energy_per_created_j"]),
        ),
    ):
        duplicate = any(
            np.isclose(
                float(point["effective_delay"]),
                float(other["effective_delay"]),
                atol=1e-8,
                rtol=1e-7,
            )
            and np.isclose(
                float(point["energy_per_created_j"]),
                float(other["energy_per_created_j"]),
                atol=1e-8,
                rtol=1e-7,
            )
            for other in unique
        )
        if not duplicate:
            unique.append(point)
    front = []
    for candidate in unique:
        dominated = any(
            other["effective_delay"] <= candidate["effective_delay"]
            and other["energy_per_created_j"] <= candidate["energy_per_created_j"]
            and (
                other["effective_delay"] < candidate["effective_delay"]
                or other["energy_per_created_j"] < candidate["energy_per_created_j"]
            )
            for other in unique
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["effective_delay"])


def _hypervolume(points: list[dict[str, Any]], reference: tuple[float, float]) -> float:
    return _hypervolume_rows(points, reference)[0]


def _hypervolume_reference(
    config: Config,
    points: list[dict[str, Any]],
) -> tuple[float, float]:
    configured = config.evaluation.get("hypervolume_reference")
    if configured is not None:
        if len(configured) != 2:
            raise ValueError("evaluation.hypervolume_reference must contain [delay, energy]")
        reference = (float(configured[0]), float(configured[1]))
        if reference[0] <= 0.0 or reference[1] <= 0.0:
            raise ValueError("evaluation.hypervolume_reference values must be positive")
        return reference
    return (
        max(item["effective_delay"] for item in points) * 1.1,
        max(item["energy_per_created_j"] for item in points) * 1.1,
    )


def run_evaluation(
    config: Config,
    checkpoint: str | Path,
    output_path: str | Path,
    seed: int,
    device_arg: str = "auto",
    official_mappo_checkpoint: str | Path | None = None,
    official_happo_checkpoint: str | Path | None = None,
    official_cavdn_checkpoint: str | Path | None = None,
    official_droo_checkpoint: str | Path | None = None,
) -> Path:
    set_seed(seed)
    device = resolve_device(device_arg)
    episodes = int(config.evaluation.episodes)
    anchors = [list(map(float, anchor)) for anchor in config.reward.anchors]
    ours = load_policy(config, checkpoint, device)
    channels = []
    for channel, anchor in enumerate(anchors):
        metrics = evaluate_learned(
            config,
            ours,
            anchor,
            episodes,
            seed + 100,
            device,
            deterministic=bool(config.evaluation.deterministic),
        )
        channels.append({"method": "uamco", "channel": channel, "preference": anchor, **metrics})
        print(
            f"[eval] ch{channel} delay={metrics['effective_delay']:.3f} "
            f"energy={metrics['energy_per_created_j']:.1f}J "
            f"completion={metrics['completion_rate']:.3f} remote={metrics['remote_ratio']:.3f}"
        )

    greedy_channels = []
    if bool(config.evaluation.get("report_greedy", True)):
        for channel, anchor in enumerate(anchors):
            metrics = evaluate_learned(
                config,
                ours,
                anchor,
                episodes,
                seed + 100,
                device,
                deterministic=True,
            )
            greedy_channels.append(
                {
                    "method": "uamco_greedy",
                    "channel": channel,
                    "preference": anchor,
                    **metrics,
                }
            )

    balanced = [0.5, 0.5]
    balanced_channel = min(
        channels,
        key=lambda item: float(
            np.square(np.asarray(item["preference"]) - np.asarray(balanced)).sum()
        ),
    )
    comparison_point = {
        **balanced_channel,
        "method": "uamco_balanced",
    }

    baseline_channels = []
    for name in config.baselines.methods:
        if name in {
            "official_mappo",
            "official_happo",
            "official_cavdn",
            "official_droo",
        }:
            continue
        for channel, anchor in enumerate(anchors):
            metrics = evaluate_heuristic(
                config,
                name,
                anchor,
                episodes,
                seed + 100,
                device,
            )
            baseline_channels.append(
                {
                    "method": name,
                    "channel": channel,
                    "preference": anchor,
                    **metrics,
                }
            )
    baselines = [
        {
            **min(
                (
                    item
                    for item in baseline_channels
                    if item["method"] == name
                ),
                key=lambda item: float(
                    np.square(
                        np.asarray(item["preference"]) - np.asarray(balanced)
                    ).sum()
                ),
            ),
        }
        for name in sorted({item["method"] for item in baseline_channels})
    ]

    official_checkpoints = (
        ("official_mappo", official_mappo_checkpoint),
        ("official_happo", official_happo_checkpoint),
        ("official_cavdn", official_cavdn_checkpoint),
        ("official_droo", official_droo_checkpoint),
    )
    learned_baseline_channels = []
    for method_name, baseline_path in official_checkpoints:
        if baseline_path and Path(baseline_path).exists():
            for channel, anchor in enumerate(anchors):
                metrics = evaluate_official_baseline(
                    config,
                    baseline_path,
                    episodes=episodes,
                    seed=seed + 100,
                    device=device,
                    deterministic=bool(config.evaluation.deterministic),
                    preference=anchor,
                )
                learned_baseline_channels.append(
                    {
                        "method": method_name,
                        "channel": channel,
                        "preference": anchor,
                        **metrics,
                    }
                )
    learned_baselines = [
        {
            **min(
                (
                    item
                    for item in learned_baseline_channels
                    if item["method"] == method_name
                ),
                key=lambda item: float(
                    np.square(
                        np.asarray(item["preference"]) - np.asarray(balanced)
                    ).sum()
                ),
            ),
        }
        for method_name in sorted(
            {item["method"] for item in learned_baseline_channels}
        )
    ]

    stress = []
    delay_anchor = anchors[0]
    for multiplier in config.evaluation.stress_load_multipliers:
        multiplier = float(multiplier)
        ours_metrics = evaluate_learned(
            config,
            ours,
            delay_anchor,
            episodes,
            seed + 300,
            device,
            load_multiplier=multiplier,
        )
        local_metrics = evaluate_heuristic(
            config,
            "local_only",
            delay_anchor,
            episodes,
            seed + 300,
            device,
            load_multiplier=multiplier,
        )
        stress.extend(
            [
                {"method": "uamco", "load_multiplier": multiplier, **ours_metrics},
                {"method": "local_only", "load_multiplier": multiplier, **local_metrics},
            ]
        )
        for method_name, baseline_path in official_checkpoints:
            if baseline_path and Path(baseline_path).exists():
                metrics = evaluate_official_baseline(
                    config,
                    baseline_path,
                    episodes=episodes,
                    seed=seed + 300,
                    device=device,
                    deterministic=bool(config.evaluation.deterministic),
                    load_multiplier=multiplier,
                    preference=delay_anchor,
                )
                stress.append(
                    {
                        "method": method_name,
                        "load_multiplier": multiplier,
                        **metrics,
                    }
                )

    terrain_ablation = {}
    if bool(config.evaluation.get("run_terrain", True)):
        terrain_ablation = {
            "real_terrain": evaluate_learned(
                config,
                ours,
                delay_anchor,
                episodes,
                seed + 500,
                device,
                terrain_enabled=True,
            ),
            "flat_terrain": evaluate_learned(
                config,
                ours,
                delay_anchor,
                episodes,
                seed + 500,
                device,
                terrain_enabled=False,
            ),
        }
    all_points = (
        channels
        + baseline_channels
        + learned_baseline_channels
    )
    reference = _hypervolume_reference(config, all_points)
    payload = {
        "environment_version": "pegasus_micro_motion_v1",
        "workflow_source_fingerprint": workflow_source_fingerprint(config),
        "implementation_fingerprint": implementation_fingerprint(),
        "workflow_evaluation": load_workflow_bank(config, split="eval").metadata,
        "objective_version": "team_undiscounted_causal_delivery_v4",
        "hypervolume_sensitivity": [
            {"reference": [reference[0] * scale, reference[1] * scale],
             "methods": {method: _hypervolume([p for p in all_points if p.get("method") == method],
                                               (reference[0] * scale, reference[1] * scale))
                         for method in sorted({p["method"] for p in all_points})}}
            for scale in [0.8, 1.0, 1.2]
        ],
        "seed": seed,
        "episodes": episodes,
        "channels": channels,
        "greedy_channels": greedy_channels,
        "semantic_score": _semantic_score(channels),
        "comparison_points": [comparison_point],
        "baseline_channels": baseline_channels,
        "baselines": baselines,
        "learned_baseline_channels": learned_baseline_channels,
        "learned_baselines": learned_baselines,
        "stress": stress,
        "terrain_ablation": terrain_ablation,
        "pareto_front": _pareto_front(all_points),
        "hypervolume": {
            "reference": reference,
            "uamco": _hypervolume(channels, reference),
            "learned_baselines": {
                method: _hypervolume(
                    [
                        item
                        for item in learned_baseline_channels
                        if item["method"] == method
                    ],
                    reference,
                )
                for method in {
                    item["method"] for item in learned_baseline_channels
                }
            },
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
