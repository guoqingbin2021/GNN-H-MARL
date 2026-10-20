# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import Config, resolve_path
from .data import TraceData, load_trace_data
from .physics import (
    compute_duration,
    compute_energy,
    dbm_per_hz_to_w_per_hz,
    free_space_path_loss_db,
    lookup_priority_weight,
    remaining_contact_time,
    rotary_wing_level_power,
    scalarized_delay_energy_reward,
    semantic_priority_class,
    shannon_rate,
    weighted_proportional_share,
)
from .terrain import TerrainMap
from .workflows import load_workflow_bank


def _tensor(value: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=dtype, device=device)


def _weighted_farthest_positions(data: TraceData, count: int) -> np.ndarray:
    flow = data.traffic.mean(axis=0)
    degree = data.adjacency.sum(axis=1)
    score = flow / max(float(flow.max()), 1e-6) + degree / max(float(degree.max()), 1.0)
    selected = [int(score.argmax())]
    while len(selected) < count:
        distances = np.linalg.norm(
            data.xy[:, None, :] - data.xy[np.asarray(selected)][None, :, :],
            axis=-1,
        ).min(axis=1)
        utility = distances * (0.5 + score)
        utility[np.asarray(selected)] = -1.0
        selected.append(int(utility.argmax()))
    return np.asarray(selected, dtype=np.int64)


@dataclass
class EpisodeMetrics:
    reward: torch.Tensor
    delay_cost: torch.Tensor
    energy_j: torch.Tensor
    completed: torch.Tensor
    timed_out: torch.Tensor
    dropped: torch.Tensor
    created: torch.Tensor
    completion_latency: torch.Tensor
    local_actions: torch.Tensor
    rsu_actions: torch.Tensor
    uav_actions: torch.Tensor
    uav_flight_energy_j: torch.Tensor
    invalid_actions: torch.Tensor
    action_decisions: torch.Tensor
    invalid_uav_dispatches: torch.Tensor
    uav_dispatch_decisions: torch.Tensor
    upload_interruptions: torch.Tensor
    buffered_results: torch.Tensor
    return_attempts: torch.Tensor
    return_retries: torch.Tensor
    return_deferrals: torch.Tensor
    delivered_results: torch.Tensor
    terminal_unfinished: torch.Tensor

    def mean_dict(self, mask: torch.Tensor | None = None) -> dict[str, float]:
        def values(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if mask is None else tensor[mask]

        reward = values(self.reward)
        delay_cost = values(self.delay_cost)
        energy_j = values(self.energy_j)
        completed_values = values(self.completed)
        timed_out = values(self.timed_out)
        dropped = values(self.dropped)
        created = values(self.created)
        completion_latency = values(self.completion_latency)
        local_actions = values(self.local_actions)
        rsu_actions = values(self.rsu_actions)
        uav_actions = values(self.uav_actions)
        uav_flight_energy_j = values(self.uav_flight_energy_j)
        invalid_actions = values(self.invalid_actions)
        action_decisions = values(self.action_decisions)
        invalid_uav_dispatches = values(self.invalid_uav_dispatches)
        uav_dispatch_decisions = values(self.uav_dispatch_decisions)
        upload_interruptions = values(self.upload_interruptions)
        buffered_results = values(self.buffered_results)
        return_attempts = values(self.return_attempts)
        return_retries = values(self.return_retries)
        return_deferrals = values(self.return_deferrals)
        delivered_results = values(self.delivered_results)
        terminal_unfinished = values(self.terminal_unfinished)
        completed = completed_values.sum().clamp_min(1.0)
        total_actions = (local_actions + rsu_actions + uav_actions).sum().clamp_min(1.0)
        total_invalid = invalid_actions.sum() + invalid_uav_dispatches.sum()
        total_decisions = (
            action_decisions.sum() + uav_dispatch_decisions.sum()
        ).clamp_min(1.0)
        return {
            "reward": float(reward.mean().item()),
            "delay_cost": float(delay_cost.mean().item()),
            "energy_j": float(energy_j.mean().item()),
            "completed": float(completed_values.mean().item()),
            "timed_out": float(timed_out.mean().item()),
            "dropped": float(dropped.mean().item()),
            "created": float(created.mean().item()),
            "terminal_unfinished": float(terminal_unfinished.mean().item()),
            "terminal_unfinished_rate": float((terminal_unfinished.sum() / created.sum().clamp_min(1.0)).item()),
            "completion_rate": float((completed_values.sum() / created.sum().clamp_min(1.0)).item()),
            "timeout_rate": float(((timed_out + dropped).sum() / created.sum().clamp_min(1.0)).item()),
            "effective_delay": float((delay_cost.sum() / created.sum().clamp_min(1.0)).item()),
            "energy_per_created_j": float((energy_j.sum() / created.sum().clamp_min(1.0)).item()),
            "mean_completion_latency": float((completion_latency.sum() / completed).item()),
            "local_ratio": float((local_actions.sum() / total_actions).item()),
            "rsu_ratio": float((rsu_actions.sum() / total_actions).item()),
            "uav_ratio": float((uav_actions.sum() / total_actions).item()),
            "uav_flight_energy_j": float(uav_flight_energy_j.mean().item()),
            "invalid_action_rate": float(
                (total_invalid / total_decisions).item()
            ),
            "vehicle_invalid_action_rate": float(
                (invalid_actions.sum() / action_decisions.sum().clamp_min(1.0)).item()
            ),
            "uav_invalid_dispatch_rate": float(
                (
                    invalid_uav_dispatches.sum()
                    / uav_dispatch_decisions.sum().clamp_min(1.0)
                ).item()
            ),
            "upload_interruptions": float(upload_interruptions.mean().item()),
            "buffered_results": float(buffered_results.mean().item()),
            "return_attempts": float(return_attempts.mean().item()),
            "return_retries": float(return_retries.mean().item()),
            "return_deferrals": float(return_deferrals.mean().item()),
            "delivered_results": float(delivered_results.mean().item()),
        }


class VectorOffloadEnv:
    """GPU-vectorized, event-timed DAG offloading environment."""

    def __init__(
        self,
        config: Config,
        num_envs: int,
        device: torch.device,
        seed: int = 1,
        load_multiplier: float | torch.Tensor = 1.0,
        terrain_enabled: bool = True,
        workflow_split: str | None = None,
    ):
        self.cfg = config
        self.device = device
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        load_tensor = torch.as_tensor(
            load_multiplier,
            dtype=torch.float32,
            device=device,
        )
        if load_tensor.ndim == 0:
            self.load_multiplier = load_tensor.expand(self.num_envs).clone()
        else:
            self.load_multiplier = load_tensor.reshape(self.num_envs).clone()
        self.terrain_enabled = bool(terrain_enabled)
        self.communication_mode = str(
            config.system.get("communication_mode", "intermittent")
        ).lower()
        if self.communication_mode not in {"complete", "intermittent"}:
            raise ValueError(
                "system.communication_mode must be 'complete' or 'intermittent'"
            )
        self.contact_prediction_error = float(
            config.system.get("contact_prediction_error", 0.0)
        )
        if not 0.0 <= self.contact_prediction_error <= 1.0:
            raise ValueError("system.contact_prediction_error must be in [0, 1]")
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.task_generator = torch.Generator(device=device)
        self.task_generator.manual_seed(seed + 1_000_003)
        self.prediction_generator = torch.Generator(device=device)
        self.prediction_generator.manual_seed(seed + 2_000_003)
        self.completion_only_dependencies = bool(config.system.get("completion_only_dependencies", False))
        self.data = load_trace_data(config)
        self.xy = _tensor(self.data.xy, device)
        self.map_xy_min = self.xy.min(dim=0).values
        self.map_xy_max = self.xy.max(dim=0).values
        self.latlon = _tensor(self.data.latlon, device)
        self.neighbors = _tensor(self.data.neighbors, device, torch.long)
        self.neighbor_mask = _tensor(self.data.neighbor_mask, device, torch.bool)
        self.traffic = _tensor(self.data.traffic_unit, device)
        self.uci_scale = _tensor(self.data.uci_scale, device)
        self.xy_to_latlon = _tensor(self.data.xy_to_latlon, device)
        self.terrain = TerrainMap.from_json(
            resolve_path(config, config.data.terrain_srtm),
            device,
        )
        self.workflow_bank = load_workflow_bank(config, split=workflow_split)
        self.templates = _tensor(self.workflow_bank.adjacency, device, torch.bool)
        self.workflow_node_mask = _tensor(self.workflow_bank.node_mask, device, torch.bool)
        self.workflow_cycles = _tensor(self.workflow_bank.cycles_g, device) * 1e9
        self.workflow_inputs = _tensor(self.workflow_bank.input_mb, device)
        self.workflow_outputs = _tensor(self.workflow_bank.output_mb, device)
        self.workflow_edges = _tensor(self.workflow_bank.edge_data_mb, device)
        family_members = [np.flatnonzero(self.workflow_bank.family_ids == family)
                          for family in np.unique(self.workflow_bank.family_ids)]
        family_table = np.zeros((len(family_members), max(map(len, family_members))), dtype=np.int64)
        for family, members in enumerate(family_members):
            family_table[family, :len(members)] = members
        self.workflow_family_table = _tensor(family_table, device, torch.long)
        self.workflow_family_counts = _tensor([len(members) for members in family_members], device, torch.long)

        self.b = self.num_envs
        self.v = int(config.system.num_vehicles)
        self.r = int(config.system.num_rsus)
        self.u = int(config.system.num_uavs)
        self.region_rows = int(config.system.get("demand_region_rows", 5))
        self.region_cols = int(config.system.get("demand_region_cols", 5))
        self.region_n = self.region_rows * self.region_cols
        self.s = int(config.system.num_subtasks)
        self.q = int(config.system.task_queue_size)
        self.remote_n = self.r + self.u
        self.action_n = 1 + self.remote_n
        self.dt = float(config.system.dt_seconds)
        self.horizon = int(config.system.episode_steps)
        self.deadline = float(config.system.deadline_seconds)
        x_edges = torch.linspace(
            self.map_xy_min[0],
            self.map_xy_max[0],
            self.region_cols + 1,
            device=device,
        )
        y_edges = torch.linspace(
            self.map_xy_min[1],
            self.map_xy_max[1],
            self.region_rows + 1,
            device=device,
        )
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        region_y, region_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
        self.region_centers = torch.stack(
            [region_x.flatten(), region_y.flatten()],
            dim=-1,
        )

        # Jointly spread fixed RSU sites and UAV launch sites over the training
        # trace.  Co-locating every UAV base with an RSU duplicated existing
        # coverage and forced a long transit before UAVs could serve blind
        # zones.  This placement is fixed before RL and uses no episode state,
        # preference, reward, or held-out trace information.
        infrastructure_indices = _weighted_farthest_positions(
            self.data,
            self.r + self.u,
        )
        rsu_indices = infrastructure_indices[: self.r]
        self.rsu_xy = self.xy[rsu_indices]
        self.rsu_latlon = self.latlon[rsu_indices]
        base_indices = infrastructure_indices[self.r :]
        self.uav_base_xy = self.xy[base_indices]
        self.uav_base_latlon = self.latlon[base_indices]
        self.cpu_hz = _tensor(
            [config.compute.vehicle_cpu_ghz] * self.v
            + [config.compute.rsu_cpu_ghz] * self.r
            + [config.compute.uav_cpu_ghz] * self.u,
            device,
        ) * 1e9
        self.priority_weights = _tensor(config.reward.priority_weights, device)
        self.noise_psd_w_hz = dbm_per_hz_to_w_per_hz(
            float(config.radio.noise_psd_dbm_hz)
            + float(config.radio.noise_figure_db)
        )
        self.reset()

    def summary_metrics(self, mask: torch.Tensor | None = None) -> dict[str, float]:
        """Return the single fairness-evaluation metric definition used by every method."""
        def selected(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if mask is None else tensor[mask]

        metrics = self.metrics.mean_dict(mask)
        metrics["remaining_tasks"] = float(
            selected(self.queue_count.sum(dim=1).float()).mean().item()
        )
        metrics["mean_aoi_seconds"] = float(
            selected(self.aoi.mean(dim=1)).mean().item()
        )
        metrics["deadline_violation_rate"] = metrics["timeout_rate"]
        metrics["action_feasibility_violation_rate"] = metrics[
            "invalid_action_rate"
        ]
        metrics["resource_violation_rate"] = 0.0
        # The three categories use different denominators and therefore must
        # not be averaged as if they were one event probability.  Retain the
        # legacy aggregate key as the worst category rate; reports and gates
        # also expose/check every category separately.
        metrics["constraint_violation_rate"] = float(
            max(
                metrics["deadline_violation_rate"],
                metrics["action_feasibility_violation_rate"],
                metrics["resource_violation_rate"],
            )
        )
        metrics["service_continuity"] = 1.0 - metrics["timeout_rate"]
        metrics["remote_ratio"] = metrics["rsu_ratio"] + metrics["uav_ratio"]
        metrics.update(self.region_metrics(mask))
        return metrics

    def region_metrics(
        self,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        def selected(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if mask is None else tensor[mask]

        decisions = selected(self.uav_region_decisions).sum().clamp_min(1.0)
        active = selected(self.uav_region_active_decisions).sum().clamp_min(1.0)
        new_missions = selected(self.uav_new_missions).sum().clamp_min(1.0)
        return {
            "hotspot_workload_coverage": float((selected(self.hotspot_covered_work).sum()
                / selected(self.hotspot_total_work).sum().clamp_min(1e-9)).item()),
            "uav_region_active_rate": float(
                (selected(self.uav_region_active_decisions).sum() / decisions).item()
            ),
            "uav_region_switch_rate": float(
                (selected(self.uav_region_switches).sum() / active).item()
            ),
            "uav_selected_region_demand_ratio": float(
                (
                    selected(self.uav_selected_region_demand_ratio).sum()
                    / active
                ).item()
            ),
            "uav_joint_overlap_rate": float(
                (
                    selected(self.uav_joint_overlap_assignments).sum()
                    / active
                ).item()
            ),
            "uav_joint_distinct_region_ratio": float(
                (
                    selected(self.uav_joint_distinct_regions).sum()
                    / active
                ).item()
            ),
            "uav_region_stay_rate": float(
                (selected(self.uav_region_stays).sum() / active).item()
            ),
            "uav_stationary_rate": float(
                (selected(self.uav_stationary_dispatches).sum() / active).item()
            ),
            "uav_new_mission_rate": float(
                (selected(self.uav_new_missions).sum() / active).item()
            ),
            "uav_target_reachable_rate": float(
                (selected(self.uav_target_reachable).sum() / active).item()
            ),
            "uav_target_coverage_ready_rate": float(
                (selected(self.uav_target_coverage_ready).sum() / active).item()
            ),
            "uav_move_distance_m_per_active_dispatch": float(
                (selected(self.uav_move_distance_m).sum() / active).item()
            ),
            "uav_selected_distance_to_comparable_ratio": float(
                (
                    selected(self.uav_selected_distance_ratio).sum()
                    / new_missions
                ).item()
            ),
        }

    def reset(
        self,
        preference: torch.Tensor | None = None,
        load_multiplier: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if load_multiplier is not None:
            load_tensor = torch.as_tensor(
                load_multiplier,
                dtype=torch.float32,
                device=self.device,
            )
            if load_tensor.ndim == 0:
                self.load_multiplier = load_tensor.expand(self.b).clone()
            else:
                self.load_multiplier = load_tensor.reshape(self.b).clone()
        self.step_index = 0
        self.time = torch.zeros(self.b, device=self.device)
        self.traffic_index = torch.randint(
            0,
            max(len(self.traffic) - self.horizon - 1, 1),
            (self.b,),
            generator=self.generator,
            device=self.device,
        )
        self.uci_index = torch.randint(
            0,
            max(len(self.uci_scale) - self.horizon - 1, 1),
            (self.b,),
            generator=self.generator,
            device=self.device,
        )
        weights = self.traffic[self.traffic_index]
        self.sensor = torch.multinomial(weights.clamp_min(1e-3), self.v, replacement=True, generator=self.generator)
        self.next_sensor = self._sample_next_sensor(self.sensor, weights)
        self.progress = torch.rand((self.b, self.v), generator=self.generator, device=self.device)
        self.vehicle_xy, self.vehicle_latlon = self._vehicle_positions()
        self.vehicle_velocity_xy = self._vehicle_velocity()
        self.contact_prediction_noise = torch.empty(
            (self.b, self.v, self.remote_n),
            device=self.device,
        ).uniform_(-1.0, 1.0, generator=self.generator)
        self._sample_geometry_error()

        self.uav_xy = self.uav_base_xy.unsqueeze(0).expand(self.b, -1, -1).clone()
        self.uav_target_active = torch.zeros((self.b, self.u), dtype=torch.bool, device=self.device)
        self.uav_region_assignment = torch.full(
            (self.b, self.u),
            self.region_n,
            dtype=torch.long,
            device=self.device,
        )
        self.previous_uav_region_assignment = self.uav_region_assignment.clone()
        self.uav_mission_xy = self.uav_base_xy.unsqueeze(0).expand(
            self.b,
            -1,
            -1,
        ).clone()
        self.uav_available = torch.zeros_like(self.uav_target_active)
        self.pending_flight_energy = torch.zeros((self.b, self.u), device=self.device)
        self.uav_energy_used = torch.zeros((self.b, self.u), device=self.device)
        self.blind_zone_age = torch.zeros((self.b, self.v), device=self.device)
        self.contact_interruptions = torch.zeros((self.b, self.v), device=self.device)
        self.rsu_connected_prev = self._rsu_connected()
        self.uav_region_decisions = torch.zeros(self.b, device=self.device)
        self.uav_region_active_decisions = torch.zeros(self.b, device=self.device)
        self.uav_region_switches = torch.zeros(self.b, device=self.device)
        self.uav_selected_region_demand_ratio = torch.zeros(
            self.b,
            device=self.device,
        )
        self.uav_joint_overlap_assignments = torch.zeros(
            self.b,
            device=self.device,
        )
        self.uav_joint_distinct_regions = torch.zeros(
            self.b,
            device=self.device,
        )
        self.uav_region_stays = torch.zeros(self.b, device=self.device)
        self.uav_stationary_dispatches = torch.zeros(
            self.b,
            device=self.device,
        )
        self.uav_new_missions = torch.zeros(self.b, device=self.device)
        self.uav_target_reachable = torch.zeros(self.b, device=self.device)
        self.uav_target_coverage_ready = torch.zeros(self.b, device=self.device)
        self.uav_move_distance_m = torch.zeros(self.b, device=self.device)
        self.uav_selected_distance_ratio = torch.zeros(
            self.b,
            device=self.device,
        )

        shape_task = (self.b, self.v, self.q)
        shape_subtask = shape_task + (self.s,)
        self.queue_count = torch.zeros((self.b, self.v), dtype=torch.long, device=self.device)
        self.queue_head = torch.zeros_like(self.queue_count)
        self.task_template = torch.zeros(shape_task, dtype=torch.long, device=self.device)
        self.task_arrival = torch.zeros(shape_task, device=self.device)
        self.task_cycles = torch.zeros(shape_subtask, device=self.device)
        self.task_data_mb = torch.zeros(shape_subtask, device=self.device)
        self.task_output_mb = torch.zeros(shape_subtask, device=self.device)
        self.task_high_value = torch.zeros(shape_subtask, dtype=torch.bool, device=self.device)
        self.task_urgent = torch.zeros(shape_subtask, dtype=torch.bool, device=self.device)
        self.task_priority = torch.full(shape_subtask, 4, dtype=torch.long, device=self.device)
        self.sub_done = torch.zeros(shape_subtask, dtype=torch.bool, device=self.device)
        self.sub_running = torch.zeros_like(self.sub_done)
        self.sub_finish = torch.full(shape_subtask, torch.inf, device=self.device)
        self.sub_node = torch.full(shape_subtask, -1, dtype=torch.long, device=self.device)
        self.sub_remote_queued = torch.zeros_like(self.sub_done)
        self.sub_remote_computing = torch.zeros_like(self.sub_done)
        self.sub_return_pending = torch.zeros_like(self.sub_done)
        self.sub_returning = torch.zeros_like(self.sub_done)
        self.sub_service_ready = torch.full(shape_subtask, torch.inf, device=self.device)
        self.sub_service_duration = torch.full(shape_subtask, torch.inf, device=self.device)
        self.sub_is_sink = torch.zeros_like(self.sub_done)
        self.node_available = torch.zeros((self.b, self.v + self.remote_n), device=self.device)
        self.aoi = torch.zeros((self.b, self.v), device=self.device)
        self._enqueue(torch.ones((self.b, self.v), dtype=torch.bool, device=self.device))

        if preference is None:
            self.preference = torch.full((self.b, 2), 0.5, device=self.device)
        else:
            self.preference = preference.to(self.device).reshape(self.b, 2)

        zeros = torch.zeros(self.b, device=self.device)
        self.hotspot_covered_work = zeros.clone()
        self.hotspot_total_work = zeros.clone()
        self.metrics = EpisodeMetrics(
            reward=zeros.clone(),
            delay_cost=zeros.clone(),
            energy_j=zeros.clone(),
            completed=zeros.clone(),
            timed_out=zeros.clone(),
            dropped=zeros.clone(),
            created=torch.full_like(zeros, float(self.v)),
            completion_latency=zeros.clone(),
            local_actions=zeros.clone(),
            rsu_actions=zeros.clone(),
            uav_actions=zeros.clone(),
            uav_flight_energy_j=zeros.clone(),
            invalid_actions=zeros.clone(),
            action_decisions=zeros.clone(),
            invalid_uav_dispatches=zeros.clone(),
            uav_dispatch_decisions=zeros.clone(),
            upload_interruptions=zeros.clone(),
            buffered_results=zeros.clone(),
            return_attempts=zeros.clone(),
            return_retries=zeros.clone(),
            return_deferrals=zeros.clone(),
            delivered_results=zeros.clone(),
            terminal_unfinished=zeros.clone(),
        )
        return self.observe()

    def _sample_next_sensor(self, sensor: torch.Tensor, traffic_row: torch.Tensor) -> torch.Tensor:
        neighbor_ids = self.neighbors[sensor]
        valid = self.neighbor_mask[sensor]
        weights = traffic_row.gather(1, neighbor_ids.reshape(self.b, -1)).reshape_as(neighbor_ids)
        weights = torch.where(valid, weights.clamp_min(1e-3), torch.zeros_like(weights))
        choice = torch.multinomial(weights.reshape(-1, weights.shape[-1]), 1, generator=self.generator)
        return neighbor_ids.reshape(-1, neighbor_ids.shape[-1]).gather(1, choice).reshape(self.b, self.v)

    def _vehicle_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        start_xy = self.xy[self.sensor]
        end_xy = self.xy[self.next_sensor]
        start_ll = self.latlon[self.sensor]
        end_ll = self.latlon[self.next_sensor]
        fraction = self.progress.unsqueeze(-1)
        return start_xy * (1.0 - fraction) + end_xy * fraction, start_ll * (1.0 - fraction) + end_ll * fraction

    def _vehicle_velocity(self) -> torch.Tensor:
        row = self.traffic[
            (self.traffic_index + self.step_index).clamp_max(len(self.traffic) - 1)
        ]
        local_flow = row.gather(1, self.sensor)
        speed = 6.0 + 12.0 * (1.0 - local_flow)
        edge = self.xy[self.next_sensor] - self.xy[self.sensor]
        edge_length = torch.linalg.norm(edge, dim=-1).clamp_min(10.0)
        if self.terrain_enabled:
            start_elevation = self.terrain.sample(self.latlon[self.sensor])
            end_elevation = self.terrain.sample(self.latlon[self.next_sensor])
            grade = (end_elevation - start_elevation).abs() / edge_length
            speed = speed * torch.exp(-8.0 * grade).clamp_min(0.55)
        return edge / edge_length.unsqueeze(-1) * speed.unsqueeze(-1)

    def _xy_to_ll(self, xy: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((*xy.shape[:-1], 1), dtype=xy.dtype, device=xy.device)
        return torch.cat([xy, ones], dim=-1) @ self.xy_to_latlon

    def vehicle_region_index(self) -> torch.Tensor:
        span = (self.map_xy_max - self.map_xy_min).clamp_min(1.0)
        normalized = ((self.vehicle_xy - self.map_xy_min) / span).clamp(
            0.0,
            1.0 - 1e-6,
        )
        col = (normalized[..., 0] * self.region_cols).long()
        row = (normalized[..., 1] * self.region_rows).long()
        return row * self.region_cols + col

    def demand_region_centers(self) -> torch.Tensor:
        region_index = self.vehicle_region_index()
        membership = torch.nn.functional.one_hot(
            region_index,
            num_classes=self.region_n,
        ).float()
        remaining_cycles_g = (
            self.task_cycles
            * (~self.sub_done).float()
        ).sum(dim=(2, 3)) / 1e9
        demand_weight = remaining_cycles_g.clamp_min(1e-3)
        weighted_membership = membership * demand_weight.unsqueeze(-1)
        total_weight = weighted_membership.sum(dim=1)
        weighted_xy = torch.einsum(
            "bvr,bvd->brd",
            weighted_membership,
            self.vehicle_xy,
        )
        fallback = self.region_centers.unsqueeze(0).expand(self.b, -1, -1)
        return torch.where(
            (total_weight > 1e-6).unsqueeze(-1),
            weighted_xy / total_weight.clamp_min(1e-6).unsqueeze(-1),
            fallback,
        )

    def uav_region_service_points(
        self,
        region_centers: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if region_centers is None:
            region_centers = self.demand_region_centers()
        region_index = self.vehicle_region_index()
        membership = torch.nn.functional.one_hot(
            region_index,
            num_classes=self.region_n,
        ).bool()
        vehicle_distance = torch.linalg.norm(
            self.vehicle_xy.unsqueeze(2) - region_centers.unsqueeze(1),
            dim=-1,
        )
        cluster_radius = vehicle_distance.masked_fill(
            ~membership,
            0.0,
        ).max(dim=1).values
        service_margin = (
            float(self.cfg.radio.uav_range_m) - cluster_radius
        ).clamp_min(0.0)
        center_delta = region_centers.unsqueeze(1) - self.uav_xy.unsqueeze(2)
        center_distance = torch.linalg.norm(center_delta, dim=-1)
        service_distance = (
            center_distance - service_margin.unsqueeze(1)
        ).clamp_min(0.0)
        service_points = self.uav_xy.unsqueeze(2) + center_delta * (
            service_distance / center_distance.clamp_min(1e-6)
        ).unsqueeze(-1)
        if hasattr(self, "uav_mission_xy"):
            current_assignment = (
                self.uav_region_assignment.unsqueeze(-1)
                == torch.arange(
                    self.region_n,
                    device=self.device,
                ).view(1, 1, -1)
            )
            service_points = torch.where(
                current_assignment.unsqueeze(-1),
                self.uav_mission_xy.unsqueeze(2),
                service_points,
            )
        return service_points, cluster_radius

    def demand_region_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        region_index = self.vehicle_region_index()
        one_hot = torch.nn.functional.one_hot(
            region_index,
            num_classes=self.region_n,
        ).float()
        count = one_hot.sum(dim=1)
        row = self.traffic[
            (self.traffic_index + self.step_index).clamp_max(len(self.traffic) - 1)
        ]
        flow = row.gather(1, self.sensor)
        age = (
            self.time.view(self.b, 1) - self._head_tensor(self.task_arrival)
        ).clamp(0.0, self.deadline)
        remaining_cycles_g = (
            self.task_cycles
            * (~self.sub_done).float()
        ).sum(dim=(2, 3)) / 1e9
        rsu_connected = self._rsu_connected().float()
        uav_covered = (
            torch.linalg.norm(
                self.vehicle_xy.unsqueeze(2) - self.uav_xy.unsqueeze(1),
                dim=-1,
            )
            <= float(self.cfg.radio.uav_range_m)
        ).any(dim=2).float()

        def region_sum(values: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bvr,bv->br", one_hot, values)

        denominator = count.clamp_min(1.0)
        map_scale = max(float(np.abs(self.data.xy).max()), 1.0)
        centers = self.demand_region_centers() / map_scale
        features = torch.cat(
            [
                centers,
                (count / max(float(self.v), 1.0)).unsqueeze(-1),
                (region_sum(flow) / max(float(self.v), 1.0)).unsqueeze(-1),
                (
                    region_sum(self.queue_count.float() / max(float(self.q), 1.0))
                    / max(float(self.v), 1.0)
                ).unsqueeze(-1),
                (region_sum(age / self.deadline) / denominator).unsqueeze(-1),
                (
                    region_sum(remaining_cycles_g)
                    / max(
                        float(
                            self.v
                            * self.q
                            * self.s
                            * self.cfg.task.max_cycles_g
                        ),
                        1.0,
                    )
                ).unsqueeze(-1),
                (region_sum(1.0 - rsu_connected) / denominator).unsqueeze(-1),
                (region_sum(uav_covered) / denominator).unsqueeze(-1),
            ],
            dim=-1,
        )
        return features, count > 0

    def uav_region_physics(
        self,
        region_centers: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if region_centers is None:
            region_centers = self.demand_region_centers()
        if region_mask is None:
            _, region_mask = self.demand_region_state()
        service_points, cluster_radius = self.uav_region_service_points(
            region_centers
        )
        delta = service_points - self.uav_xy.unsqueeze(2)
        distance = torch.linalg.norm(delta, dim=-1)
        # Physical travel is one micro-slot, irrespective of policy frequency.
        macro_duration = self.dt
        max_move = float(self.cfg.uav.max_speed_mps) * macro_duration
        deadline_reachable = (
            distance
            <= float(self.cfg.uav.max_speed_mps)
            * float(self.cfg.system.deadline_seconds)
        )
        move_distance = distance.clamp_max(max_move)
        speed = move_distance / max(macro_duration, 1e-6)
        level_power = rotary_wing_level_power(
            speed,
            profile_power_w=float(self.cfg.uav.profile_power_w),
            induced_power_w=float(self.cfg.uav.induced_power_w),
            tip_speed_mps=float(self.cfg.uav.tip_speed_mps),
            mean_rotor_velocity_mps=float(self.cfg.uav.mean_rotor_velocity_mps),
            fuselage_drag_ratio=float(self.cfg.uav.fuselage_drag_ratio),
            air_density_kg_m3=float(self.cfg.uav.air_density_kg_m3),
            rotor_solidity=float(self.cfg.uav.rotor_solidity),
            rotor_area_m2=float(self.cfg.uav.rotor_area_m2),
        )
        flight_energy = (
            float(self.cfg.uav.base_power_w) + level_power
        ) * macro_duration
        battery_limit = (
            float(self.cfg.uav.battery_capacity_j)
            * float(self.cfg.uav.battery_usage_ratio)
        )
        map_diagonal = torch.linalg.norm(
            self.map_xy_max - self.map_xy_min
        ).clamp_min(1.0)
        current_assignment = (
            self.uav_region_assignment.unsqueeze(-1)
            == torch.arange(self.region_n, device=self.device).view(1, 1, -1)
        )
        return torch.stack(
            [
                cluster_radius.unsqueeze(1).expand(-1, self.u, -1)
                / max(float(self.cfg.radio.uav_range_m), 1.0),
                distance / max(max_move, 1.0),
                move_distance / map_diagonal,
                (distance <= max_move).float(),
                deadline_reachable.float(),
                flight_energy / max(battery_limit, 1.0),
                current_assignment.float(),
                (
                    distance <= float(self.cfg.uav.arrival_radius_m)
                ).float(),
                region_mask.unsqueeze(1).expand(-1, self.u, -1).float(),
            ],
            dim=-1,
        )

    def _advance_vehicles(self) -> None:
        row = self.traffic[(self.traffic_index + self.step_index).clamp_max(len(self.traffic) - 1)]
        velocity = self._vehicle_velocity()
        speed = torch.linalg.norm(velocity, dim=-1)
        edge_length = torch.linalg.norm(
            self.xy[self.next_sensor] - self.xy[self.sensor],
            dim=-1,
        ).clamp_min(10.0)
        self.progress += speed * self.dt / edge_length
        reached = self.progress >= 1.0
        if reached.any():
            self.progress = torch.where(reached, self.progress - 1.0, self.progress)
            self.sensor = torch.where(reached, self.next_sensor, self.sensor)
            sampled = self._sample_next_sensor(self.sensor, row)
            self.next_sensor = torch.where(reached, sampled, self.next_sensor)
        self.vehicle_xy, self.vehicle_latlon = self._vehicle_positions()
        self.vehicle_velocity_xy = self._vehicle_velocity()

    def _rsu_connected(self) -> torch.Tensor:
        _, feasible, _, _ = self._link_physics()
        return feasible[..., : self.r].any(dim=-1)

    def _update_contact_history(self) -> None:
        if self.communication_mode == "complete":
            self.blind_zone_age.zero_()
            self.contact_interruptions.zero_()
            self.rsu_connected_prev.fill_(True)
            return
        connected = self._rsu_connected()
        interrupted = self.rsu_connected_prev & ~connected
        self.blind_zone_age = torch.where(
            connected,
            torch.zeros_like(self.blind_zone_age),
            self.blind_zone_age + self.dt,
        )
        self.contact_interruptions = (
            0.9 * self.contact_interruptions + interrupted.float()
        )
        self.rsu_connected_prev = connected

    def _enqueue(self, arrival_mask: torch.Tensor) -> torch.Tensor:
        valid = arrival_mask & (self.queue_count < self.q)
        dropped = arrival_mask & ~valid
        if hasattr(self, "metrics"):
            self.metrics.dropped += dropped.sum(dim=1).float()
            self.metrics.created += arrival_mask.sum(dim=1).float()
        high_value = torch.rand(
            (self.b, self.v, self.s),
            generator=self.task_generator,
            device=self.device,
        ) < float(self.cfg.task.high_value_probability)
        urgent = torch.rand(
            (self.b, self.v, self.s),
            generator=self.task_generator,
            device=self.device,
        ) < float(self.cfg.task.urgent_probability)
        priority = semantic_priority_class(high_value, urgent)
        families = torch.randint(
            0,
            len(self.workflow_family_counts),
            (self.b, self.v),
            generator=self.task_generator,
            device=self.device,
        )
        window = (torch.rand((self.b, self.v), generator=self.task_generator, device=self.device)
                  * self.workflow_family_counts[families]).long()
        templates = self.workflow_family_table[families, window]
        cycles = self.workflow_cycles[templates]
        data = self.workflow_inputs[templates]
        if not valid.any():
            return dropped
        tail = (self.queue_head + self.queue_count) % self.q
        idx = valid.nonzero(as_tuple=False)
        b_idx, v_idx = idx[:, 0], idx[:, 1]
        q_idx = tail[b_idx, v_idx]
        self.task_template[b_idx, v_idx, q_idx] = templates[b_idx, v_idx]
        self.task_arrival[b_idx, v_idx, q_idx] = self.time[b_idx]
        self.task_cycles[b_idx, v_idx, q_idx] = cycles[b_idx, v_idx]
        self.task_data_mb[b_idx, v_idx, q_idx] = data[b_idx, v_idx]
        self.task_output_mb[b_idx, v_idx, q_idx] = self.workflow_outputs[templates[b_idx, v_idx]]
        self.task_high_value[b_idx, v_idx, q_idx] = high_value[b_idx, v_idx]
        self.task_urgent[b_idx, v_idx, q_idx] = urgent[b_idx, v_idx]
        self.task_priority[b_idx, v_idx, q_idx] = priority[b_idx, v_idx]
        # Padding nodes never enter readiness, workload, or completion counts.
        self.sub_done[b_idx, v_idx, q_idx] = ~self.workflow_node_mask[templates[b_idx, v_idx]]
        self.sub_running[b_idx, v_idx, q_idx] = False
        self.sub_finish[b_idx, v_idx, q_idx] = torch.inf
        self.sub_node[b_idx, v_idx, q_idx] = -1
        self.sub_remote_queued[b_idx, v_idx, q_idx] = False
        self.sub_remote_computing[b_idx, v_idx, q_idx] = False
        self.sub_return_pending[b_idx, v_idx, q_idx] = False
        self.sub_returning[b_idx, v_idx, q_idx] = False
        self.sub_service_ready[b_idx, v_idx, q_idx] = torch.inf
        self.sub_service_duration[b_idx, v_idx, q_idx] = torch.inf
        self.sub_is_sink[b_idx, v_idx, q_idx] = False
        self.queue_count[valid] += 1
        return dropped

    def _head_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        extra_dims = tensor.ndim - 3
        index = self.queue_head.view(self.b, self.v, 1, *([1] * extra_dims))
        index = index.expand(self.b, self.v, 1, *tensor.shape[3:])
        return tensor.gather(2, index).squeeze(2)

    def _ready_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        template_id = self._head_tensor(self.task_template)
        done = self._head_tensor(self.sub_done)
        running = self._head_tensor(self.sub_running)
        adjacency = self.templates[template_id]
        predecessors_done = (done.unsqueeze(-1) | ~adjacency).all(dim=-2)
        active = self.queue_count > 0
        ready = predecessors_done & ~done & ~running & active.unsqueeze(-1)
        # A common deterministic first-ready queue exposes a destination action
        # without silently providing a hand-written critical-path scheduler.
        selected = ready.long().argmax(dim=-1)
        has_ready = ready.any(dim=-1)
        return ready, selected, has_ready, adjacency

    def _resource_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rsu_xy = self.rsu_xy.view(1, 1, self.r, 2).expand(self.b, self.v, -1, -1)
        rsu_ll = self.rsu_latlon.view(1, 1, self.r, 2).expand(self.b, self.v, -1, -1)
        uav_xy = self.uav_xy.unsqueeze(1).expand(-1, self.v, -1, -1)
        uav_ll = self._xy_to_ll(self.uav_xy).unsqueeze(1).expand(-1, self.v, -1, -1)
        resource_xy = torch.cat([rsu_xy, uav_xy], dim=2)
        resource_ll = torch.cat([rsu_ll, uav_ll], dim=2)
        heights = _tensor(
            [self.cfg.radio.rsu_height_m] * self.r + [self.cfg.radio.uav_height_m] * self.u,
            self.device,
        )
        return resource_xy, resource_ll, heights

    def _link_physics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        resource_xy, resource_ll, heights = self._resource_geometry()
        distance_2d = torch.linalg.norm(self.vehicle_xy.unsqueeze(2) - resource_xy, dim=-1)
        if self.terrain_enabled:
            vehicle_ground = self.terrain.sample(self.vehicle_latlon).unsqueeze(-1)
            resource_ground = self.terrain.sample(resource_ll)
        else:
            vehicle_ground = torch.zeros((self.b, self.v, 1), device=self.device)
            resource_ground = torch.zeros_like(distance_2d)
        vehicle_altitude = vehicle_ground + float(self.cfg.radio.vehicle_height_m)
        resource_altitude = resource_ground + heights.view(1, 1, -1)
        distance_3d = torch.sqrt(
            distance_2d.square()
            + (resource_altitude - vehicle_altitude).square()
        ).clamp_min(1.0)
        if self.terrain_enabled:
            start_ll = self.vehicle_latlon.unsqueeze(2).expand_as(resource_ll)
            start_height = float(self.cfg.radio.vehicle_height_m)
            blocked = []
            for node in range(self.remote_n):
                blocked.append(
                    self.terrain.blocked(
                        start_ll[:, :, node],
                        resource_ll[:, :, node],
                        start_height,
                        float(heights[node].item()),
                    )
                )
            blocked_tensor = torch.stack(blocked, dim=-1)
        else:
            blocked_tensor = torch.zeros_like(distance_2d, dtype=torch.bool)

        fspl_db = free_space_path_loss_db(
            distance_3d,
            float(self.cfg.radio.carrier_ghz),
        )
        rsu_loss = (
            fspl_db[..., : self.r]
            + float(self.cfg.radio.rayleigh_loss_db)
            + blocked_tensor[..., : self.r].float()
            * float(self.cfg.radio.nlos_extra_loss_db)
        )
        elevation_deg = torch.rad2deg(
            torch.asin(
                (
                    (
                        resource_altitude[..., self.r :] - vehicle_altitude
                    ).clamp_min(0.0)
                    / distance_3d[..., self.r :]
                ).clamp(0.0, 1.0)
            )
        )
        los_probability = 1.0 / (
            1.0
            + float(self.cfg.radio.los_a)
            * torch.exp(
                -float(self.cfg.radio.los_b)
                * (elevation_deg - float(self.cfg.radio.los_a))
            )
        )
        los_probability = torch.where(
            blocked_tensor[..., self.r :],
            torch.zeros_like(los_probability),
            los_probability,
        )
        uav_extra_loss = (
            los_probability * float(self.cfg.radio.los_extra_loss_db)
            + (1.0 - los_probability)
            * float(self.cfg.radio.a2g_nlos_extra_loss_db)
        )
        uav_loss = fspl_db[..., self.r :] + uav_extra_loss
        path_loss_db = torch.cat([rsu_loss, uav_loss], dim=-1)
        channel_gain = torch.pow(10.0, -path_loss_db / 10.0)
        unit_share = torch.ones_like(channel_gain)
        rates = shannon_rate(
            float(self.cfg.radio.bandwidth_mhz) * 1e6,
            unit_share,
            float(self.cfg.radio.tx_power_w),
            channel_gain,
            self.noise_psd_w_hz,
        )
        uav_radius = (
            float(self.cfg.radio.uav_height_m)
            * math.tan(math.radians(float(self.cfg.radio.beamwidth_deg) / 2.0))
        )
        range_limit = _tensor(
            [self.cfg.radio.rsu_range_m] * self.r
            + [min(float(self.cfg.radio.uav_range_m), uav_radius)] * self.u,
            self.device,
        )
        available = torch.cat(
            [
                torch.ones((self.b, self.r), dtype=torch.bool, device=self.device),
                self.uav_available,
            ],
            dim=1,
        )
        feasible = (
            (distance_2d <= range_limit.view(1, 1, -1))
            & (rates >= 1e6)
            & available.unsqueeze(1)
        )
        if self.communication_mode == "complete":
            feasible = (
                (rates >= 1e6)
                & available.unsqueeze(1)
            )
        return channel_gain, feasible, blocked_tensor, range_limit

    def link_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gain, feasible, blocked, _ = self._link_physics()
        rates = shannon_rate(
            float(self.cfg.radio.bandwidth_mhz) * 1e6,
            torch.ones_like(gain),
            float(self.cfg.radio.tx_power_w),
            gain,
            self.noise_psd_w_hz,
        )
        return rates, feasible, blocked

    def _sample_geometry_error(self) -> None:
        # Independent stream: changing observation noise never changes arrivals/mobility.
        shape = (self.b, self.v, 2)
        self.position_error = torch.randn(shape, generator=self.prediction_generator, device=self.device)
        self.velocity_error = torch.randn(shape, generator=self.prediction_generator, device=self.device)

    def _upload_payload_mb(self, selected: torch.Tensor) -> torch.Tensor:
        inputs = self._head_tensor(self.task_data_mb)
        payload = inputs.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        if not self.completion_only_dependencies:
            edges = self.workflow_edges[self._head_tensor(self.task_template)]
            payload = payload + edges.gather(
                3, selected[..., None, None].expand(-1, -1, self.s, 1)
            ).squeeze(-1).sum(-1)
        return payload

    def _contact_horizon(
        self,
        resource_xy: torch.Tensor,
        range_limit: torch.Tensor,
        *,
        predicted: bool = True,
    ) -> torch.Tensor:
        if self.communication_mode == "complete":
            return torch.full(
                resource_xy.shape[:-1],
                torch.inf,
                dtype=resource_xy.dtype,
                device=self.device,
            )
        position = self.vehicle_xy
        velocity = self.vehicle_velocity_xy
        if predicted:
            position = position + self.position_error * float(self.cfg.system.get("position_prediction_std_m", 0.0))
            velocity = velocity + self.velocity_error * float(self.cfg.system.get("velocity_prediction_std_mps", 0.0))
        true_horizon = remaining_contact_time(
            position,
            resource_xy,
            velocity,
            range_limit.view(1, 1, -1),
            resource_motion_segments=self._resource_contact_motion(),
        )
        if not predicted or self.contact_prediction_error <= 0.0:
            return true_horizon
        multiplier = 1.0 + (
            self.contact_prediction_error * self.contact_prediction_noise
        )
        return torch.where(
            torch.isfinite(true_horizon),
            (true_horizon * multiplier).clamp_min(0.0),
            true_horizon,
        )

    def action_mask(
        self,
        feasible: torch.Tensor | None = None,
        has_ready: torch.Tensor | None = None,
        *,
        delivery_aware: bool = True,
    ) -> torch.Tensor:
        if feasible is None:
            _, feasible, _ = self.link_state()
        if has_ready is None:
            _, _, has_ready, _ = self._ready_state()
        _, selected, _, _ = self._ready_state()
        selected_data = self._upload_payload_mb(selected)
        rates, _, _ = self.link_state()
        upload = selected_data.unsqueeze(-1) * 8e6 / rates.clamp_min(1.0)
        resource_xy, _, _ = self._resource_geometry()
        _, _, _, range_limit = self._link_physics()
        contact = self._contact_horizon(resource_xy, range_limit)
        contact_feasible = upload <= contact
        remote_feasible = feasible & contact_feasible if delivery_aware else feasible
        mask = torch.cat(
            [
                torch.ones((self.b, self.v, 1), dtype=torch.bool, device=self.device),
                remote_feasible,
            ],
            dim=-1,
        )
        idle_mask = torch.zeros_like(mask)
        idle_mask[..., 0] = True
        return torch.where(has_ready.unsqueeze(-1), mask, idle_mask)

    def observe(self) -> dict[str, torch.Tensor]:
        ready, selected, has_ready, adjacency = self._ready_state()
        cycles = self._head_tensor(self.task_cycles) / 1e9
        data = self._head_tensor(self.task_data_mb)
        done = self._head_tensor(self.sub_done)
        running = self._head_tensor(self.sub_running)
        finish = self._head_tensor(self.sub_finish)
        remaining = (finish - self.time.view(self.b, 1, 1)).clamp(0.0, self.deadline)
        template_id = self._head_tensor(self.task_template)
        priority_class = self._head_tensor(self.task_priority)
        priority = 1.0 - (priority_class.float() - 1.0) / 3.0
        dag_nodes = torch.stack(
            [
                cycles / float(self.cfg.task.max_cycles_g),
                data / float(self.cfg.task.max_input_mb),
                done.float(),
                running.float(),
                ready.float(),
                priority,
                remaining / self.deadline,
                self._head_tensor(self.task_output_mb) / float(self.cfg.task.max_input_mb),
                self.workflow_edges[template_id].sum(dim=-2) / float(self.cfg.task.max_input_mb),
            ],
            dim=-1,
        )
        row = self.traffic[(self.traffic_index + self.step_index).clamp_max(len(self.traffic) - 1)]
        flow = row.gather(1, self.sensor)
        terrain_height = self.terrain.sample(self.vehicle_latlon)
        map_scale = max(float(np.abs(self.data.xy).max()), 1.0)
        age = (
            self.time.view(self.b, 1) - self._head_tensor(self.task_arrival)
        ).clamp_min(0.0)
        local_wait = (
            self.node_available[:, : self.v] - self.time.unsqueeze(1)
        ).clamp(0.0, self.deadline)
        velocity_scale = 20.0
        vehicle = torch.stack(
            [
                self.vehicle_xy[..., 0] / map_scale,
                self.vehicle_xy[..., 1] / map_scale,
                flow,
                terrain_height / 300.0,
                self.queue_count.float() / self.q,
                age / self.deadline,
                local_wait / self.deadline,
                cycles.gather(2, selected.unsqueeze(-1)).squeeze(-1)
                / float(self.cfg.task.max_cycles_g),
                data.gather(2, selected.unsqueeze(-1)).squeeze(-1)
                / float(self.cfg.task.max_input_mb),
                self.vehicle_velocity_xy[..., 0] / velocity_scale,
                self.vehicle_velocity_xy[..., 1] / velocity_scale,
                self.blind_zone_age.clamp_max(self.deadline) / self.deadline,
                self.contact_interruptions.clamp_max(5.0) / 5.0,
            ],
            dim=-1,
        )
        resource_xy = torch.cat(
            [
                self.rsu_xy.unsqueeze(0).expand(self.b, -1, -1),
                self.uav_xy,
            ],
            dim=1,
        )
        remote_wait = (
            self.node_available[:, self.v :] - self.time.unsqueeze(1)
        ).clamp(0.0, self.deadline)
        resource = torch.zeros((self.b, self.remote_n, 8), device=self.device)
        resource[..., 0:2] = resource_xy / map_scale
        resource[:, : self.r, 2] = 1.0
        resource[:, self.r :, 3] = 1.0
        resource[..., 4] = self.cpu_hz[self.v :] / 30e9
        resource[..., 5] = remote_wait / self.deadline
        resource[:, : self.r, 6] = 1.0
        resource[:, self.r :, 6] = self.uav_available.float()
        resource[:, : self.r, 7] = 1.0
        battery_limit = (
            float(self.cfg.uav.battery_capacity_j)
            * float(self.cfg.uav.battery_usage_ratio)
        )
        resource[:, self.r :, 7] = (
            1.0 - self.uav_energy_used / max(battery_limit, 1.0)
        ).clamp(0.0, 1.0)
        rates, feasible, blocked = self.link_state()
        relative_xy = (
            resource_xy.unsqueeze(1) - self.vehicle_xy.unsqueeze(2)
        ) / map_scale
        distance_2d = torch.linalg.norm(relative_xy, dim=-1, keepdim=True)
        normalized_rate = (
            torch.log1p(rates / 1e6) / math.log1p(500.0)
        ).unsqueeze(-1)
        _, _, _, range_limit = self._link_physics()
        contact = self._contact_horizon(
            resource_xy.unsqueeze(1).expand(-1, self.v, -1, -1),
            range_limit,
        )
        normalized_contact = (
            contact.clamp_max(self.deadline) / self.deadline
        ).unsqueeze(-1)
        assigned = torch.zeros(
            (self.b, self.v, self.remote_n),
            dtype=relative_xy.dtype,
            device=self.device,
        )
        if self.u > 0:
            vehicle_ids = torch.arange(
                self.v,
                dtype=torch.long,
                device=self.device,
            ).view(1, self.v, 1)
            vehicle_regions = self.vehicle_region_index().unsqueeze(-1)
            assigned[:, :, self.r :] = (
                (self.uav_region_assignment.unsqueeze(1) == vehicle_regions)
                & self.uav_available.unsqueeze(1)
            ).float()
        pair = torch.cat(
            [
                relative_xy,
                distance_2d,
                normalized_rate,
                feasible.float().unsqueeze(-1),
                blocked.float().unsqueeze(-1),
                normalized_contact,
                assigned.unsqueeze(-1),
            ],
            dim=-1,
        )
        demand_regions, demand_region_mask = self.demand_region_state()
        uav_region_pair = self.uav_region_physics(
            demand_regions[..., :2] * map_scale,
            demand_region_mask,
        )
        return {
            "vehicle": vehicle,
            "dag_nodes": dag_nodes,
            "dag_adjacency": adjacency.float(),
            "dag_selected": selected,
            "dag_node_mask": self.workflow_node_mask[template_id] & (self.queue_count > 0).unsqueeze(-1),
            "resource": resource,
            "pair": pair,
            "demand_regions": demand_regions,
            "demand_region_mask": demand_region_mask,
            "uav_region_pair": uav_region_pair,
            "uav_action_mask": self.uav_dispatch_mask(
                demand_region_mask,
                uav_region_pair,
            ),
            "action_mask": self.action_mask(
                feasible,
                has_ready,
                delivery_aware=True,
            ),
            "connectivity_action_mask": self.action_mask(
                feasible,
                has_ready,
                delivery_aware=False,
            ),
            "preference": self.preference,
        }

    def uav_dispatch_mask(
        self,
        region_mask: torch.Tensor | None = None,
        physical_pair: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if region_mask is None:
            _, region_mask = self.demand_region_state()
        if physical_pair is None:
            physical_pair = self.uav_region_physics(
                self.demand_region_centers(),
                region_mask,
            )
        current_region = physical_pair[..., 6].bool()
        service_ready = physical_pair[..., 7].bool()
        locked_region = current_region & ~service_ready
        mission_locked = locked_region.any(dim=-1)
        # The upper action selects a *current demand region*, rather than an
        # arbitrary empty grid cell.  Workload magnitude, hotspot identity and
        # deadline reachability remain observations learned by the policy; the
        # mask only removes cells with no current service object.  A mission
        # that becomes empty while the UAV is en route remains available so a
        # physical flight cannot be preempted by a changing aggregation bin.
        battery_limit = float(self.cfg.uav.battery_capacity_j) * float(self.cfg.uav.battery_usage_ratio)
        service_regions = region_mask.unsqueeze(1) | current_region
        feasible_regions = service_regions & (
            self.uav_energy_used[..., None]
            + physical_pair[..., 5] * battery_limit
            <= battery_limit
        )
        locked_feasible = locked_region & feasible_regions
        can_continue = locked_feasible.any(dim=-1)
        enforce_lock = mission_locked & can_continue
        feasible_regions = torch.where(
            enforce_lock.unsqueeze(-1),
            locked_feasible,
            feasible_regions,
        )
        # A dispatched UAV is a non-preemptive physical actuator while it is
        # en route.  If continuing becomes battery-infeasible, returning to
        # base remains the only valid fail-safe action.
        base_allowed = ~enforce_lock
        return torch.cat(
            [feasible_regions, base_allowed.unsqueeze(-1)],
            dim=-1,
        )

    def set_uav_dispatch(self, dispatch: torch.Tensor) -> None:
        dispatch = dispatch.long().clamp(0, self.region_n)
        dispatch_mask = self.uav_dispatch_mask()
        requested_valid = dispatch_mask.gather(
            2,
            dispatch.unsqueeze(-1),
        ).squeeze(-1)
        self.metrics.invalid_uav_dispatches += (
            ~requested_valid
        ).sum(dim=1).float()
        self.metrics.uav_dispatch_decisions += float(self.u)
        fallback = torch.where(
            self.previous_uav_region_assignment < self.region_n,
            self.previous_uav_region_assignment,
            torch.full_like(dispatch, self.region_n),
        )
        dispatch = torch.where(requested_valid, dispatch, fallback)
        active = dispatch < self.region_n
        target_index = dispatch.clamp_max(self.region_n - 1)
        dynamic_centers = self.demand_region_centers()
        service_points, _ = self.uav_region_service_points(dynamic_centers)
        candidate_target = service_points.gather(
            2,
            target_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
        ).squeeze(2)
        previous_active = self.previous_uav_region_assignment < self.region_n
        same_mission = (
            active
            & previous_active
            & (target_index == self.previous_uav_region_assignment)
        )
        target = torch.where(
            same_mission.unsqueeze(-1),
            self.uav_mission_xy,
            candidate_target,
        )
        base = self.uav_base_xy.unsqueeze(0).expand(self.b, -1, -1)
        target_xy = torch.where(active.unsqueeze(-1), target, base)
        delta = target_xy - self.uav_xy
        distance = torch.linalg.norm(delta, dim=-1)
        macro_duration = self.dt
        max_move = float(self.cfg.uav.max_speed_mps) * macro_duration
        move = delta * (
            max_move / distance.clamp_min(max_move)
        ).clamp_max(1.0).unsqueeze(-1)
        next_xy = torch.maximum(
            torch.minimum(self.uav_xy + move, self.map_xy_max),
            self.map_xy_min,
        )
        moved_distance = torch.linalg.norm(move, dim=-1)
        speed = moved_distance / macro_duration
        airborne = active | (
            torch.linalg.norm(next_xy - base, dim=-1)
            > float(self.cfg.uav.arrival_radius_m)
        )
        level_power = rotary_wing_level_power(
            speed,
            profile_power_w=float(self.cfg.uav.profile_power_w),
            induced_power_w=float(self.cfg.uav.induced_power_w),
            tip_speed_mps=float(self.cfg.uav.tip_speed_mps),
            mean_rotor_velocity_mps=float(self.cfg.uav.mean_rotor_velocity_mps),
            fuselage_drag_ratio=float(self.cfg.uav.fuselage_drag_ratio),
            air_density_kg_m3=float(self.cfg.uav.air_density_kg_m3),
            rotor_solidity=float(self.cfg.uav.rotor_solidity),
            rotor_area_m2=float(self.cfg.uav.rotor_area_m2),
        )
        proposed_energy = torch.where(
            airborne,
            (
                float(self.cfg.uav.base_power_w) + level_power
            ) * macro_duration,
            torch.zeros_like(level_power),
        )
        energy_limit = (
            float(self.cfg.uav.battery_capacity_j)
            * float(self.cfg.uav.battery_usage_ratio)
        )
        battery_feasible = self.uav_energy_used + proposed_energy <= energy_limit
        usable = active & battery_feasible
        region_features, _ = self.demand_region_state()
        region_demand = region_features[..., 2:7].sum(dim=-1)
        selected_demand = region_demand.gather(1, target_index)
        demand_ratio = selected_demand / region_demand.max(dim=1).values.clamp_min(
            1e-6
        ).unsqueeze(1)
        switched = (
            active
            & previous_active
            & (target_index != self.previous_uav_region_assignment)
        )
        self.uav_region_decisions += float(self.u)
        self.uav_region_active_decisions += active.float().sum(dim=1)
        self.uav_region_switches += switched.float().sum(dim=1)
        stayed = (
            active
            & previous_active
            & (target_index == self.previous_uav_region_assignment)
        )
        self.uav_region_stays += stayed.float().sum(dim=1)
        new_mission = active & (
            ~previous_active
            | (target_index != self.previous_uav_region_assignment)
        )
        self.uav_new_missions += new_mission.float().sum(dim=1)
        self.uav_target_reachable += (
            active & (distance <= max_move)
        ).float().sum(dim=1)
        remaining_to_target = torch.linalg.norm(
            target_xy - next_xy,
            dim=-1,
        )
        self.uav_target_coverage_ready += (
            active
            & (
                remaining_to_target
                <= float(self.cfg.uav.arrival_radius_m)
            )
        ).float().sum(dim=1)
        self.uav_stationary_dispatches += (
            active
            & (
                moved_distance
                <= float(self.cfg.uav.arrival_radius_m)
            )
        ).float().sum(dim=1)
        _, region_mask = self.demand_region_state()
        all_service_points, _ = self.uav_region_service_points(
            self.demand_region_centers()
        )
        all_distances = torch.linalg.norm(
            all_service_points - self.uav_xy.unsqueeze(2),
            dim=-1,
        )
        workload_scale = max(
            float(
                self.v
                * self.q
                * self.s
                * self.cfg.task.max_cycles_g
            ),
            1.0,
        )
        remaining_workload_g = region_features[..., 6] * workload_scale
        planned_capacity_g = torch.zeros_like(remaining_workload_g)
        for uav_id in range(self.u):
            residual_workload_g = (
                remaining_workload_g - planned_capacity_g
            ).clamp_min(0.0)
            selected_residual_g = residual_workload_g.gather(
                1,
                target_index[:, uav_id : uav_id + 1],
            ).squeeze(1)
            comparable = (
                region_mask
                & (residual_workload_g > 1e-6)
                & (
                    residual_workload_g
                    >= 0.8 * selected_residual_g.unsqueeze(-1)
                )
            )
            nearest_distance = all_distances[:, uav_id].masked_fill(
                ~comparable,
                torch.inf,
            ).min(dim=1).values
            selected_is_new = new_mission[:, uav_id]
            valid_nearest = torch.isfinite(nearest_distance)
            self.uav_selected_distance_ratio += torch.where(
                selected_is_new & valid_nearest,
                distance[:, uav_id]
                / nearest_distance.clamp_min(
                    float(self.cfg.uav.arrival_radius_m)
                ),
                0.0,
            )
            selected_one_hot = torch.nn.functional.one_hot(
                target_index[:, uav_id],
                num_classes=self.region_n,
            ).float()
            uav_capacity_g = (
                self.cpu_hz[self.v + self.r + uav_id]
                / 1e9
                * macro_duration
            )
            planned_capacity_g += (
                selected_one_hot
                * active[:, uav_id : uav_id + 1].float()
                * uav_capacity_g
            )
        self.uav_selected_region_demand_ratio += (
            demand_ratio * active.float()
        ).sum(dim=1)
        selected_counts = (
            torch.nn.functional.one_hot(
                target_index,
                num_classes=self.region_n,
            ).float()
            * active.unsqueeze(-1)
        ).sum(dim=1)
        distinct_regions = (selected_counts > 0).float().sum(dim=1)
        active_count = active.float().sum(dim=1)
        self.uav_joint_distinct_regions += distinct_regions
        self.uav_joint_overlap_assignments += (
            active_count - distinct_regions
        ).clamp_min(0.0)
        self.uav_region_assignment = torch.where(
            usable,
            target_index,
            torch.full_like(target_index, self.region_n),
        )
        self.uav_mission_xy = torch.where(
            usable.unsqueeze(-1),
            target,
            base,
        )
        self.previous_uav_region_assignment = self.uav_region_assignment.clone()
        # Dispatch selects an actuator mission only. Moving/charging here made
        # synchronous policies pay five seconds of flight every one-second tick.
        # The same micro-slot integrator below now advances every policy once.
        self.uav_target_active = active
        self.uav_available = usable

    def _uav_motion_plan(self) -> tuple[torch.Tensor, ...]:
        """Pure current-mission plan shared by contact checks and the integrator.

        Keep the existing per-micro-slot endpoint and average-speed energy
        accounting. Neither prediction nor repeated observations move a UAV
        or charge its battery.
        """
        base = self.uav_base_xy.unsqueeze(0).expand(self.b, -1, -1)
        active = self.uav_region_assignment < self.region_n
        target = torch.where(active.unsqueeze(-1), self.uav_mission_xy, base)
        delta = target - self.uav_xy
        distance = torch.linalg.norm(delta, dim=-1)
        max_move = float(self.cfg.uav.max_speed_mps) * self.dt
        move = delta * (max_move / distance.clamp_min(max_move)).clamp_max(1.0).unsqueeze(-1)
        next_xy = torch.maximum(torch.minimum(self.uav_xy + move, self.map_xy_max), self.map_xy_min)
        moved = torch.linalg.norm(next_xy - self.uav_xy, dim=-1)
        # Include the final flight back to base, even when it lands this tick.
        airborne = active | (torch.linalg.norm(self.uav_xy - base, dim=-1) > 1e-6) | (moved > 1e-6)
        power = rotary_wing_level_power(
            moved / self.dt,
            profile_power_w=float(self.cfg.uav.profile_power_w),
            induced_power_w=float(self.cfg.uav.induced_power_w),
            tip_speed_mps=float(self.cfg.uav.tip_speed_mps),
            mean_rotor_velocity_mps=float(self.cfg.uav.mean_rotor_velocity_mps),
            fuselage_drag_ratio=float(self.cfg.uav.fuselage_drag_ratio),
            air_density_kg_m3=float(self.cfg.uav.air_density_kg_m3),
            rotor_solidity=float(self.cfg.uav.rotor_solidity),
            rotor_area_m2=float(self.cfg.uav.rotor_area_m2),
        )
        proposed = torch.where(airborne, (float(self.cfg.uav.base_power_w) + power) * self.dt, 0.0)
        limit = float(self.cfg.uav.battery_capacity_j) * float(self.cfg.uav.battery_usage_ratio)
        feasible = self.uav_energy_used + proposed <= limit
        return target, next_xy, moved, active, proposed, feasible

    def _resource_contact_motion(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Current UAV mission as full micro-slots, final partial step, then rest.

        Mission/base targets are in the map. A sub-step displacement is spread
        over its micro-slot, matching the integrator's average-speed convention;
        it is not continued beyond the target. Future control changes, vehicle
        turns and future battery commitments are not oracle information.
        """
        target, next_xy, _, _, _, feasible = self._uav_motion_plan()
        bounded_target = torch.maximum(torch.minimum(target, self.map_xy_max), self.map_xy_min)
        delta = bounded_target - self.uav_xy
        distance = torch.linalg.norm(delta, dim=-1)
        speed = float(self.cfg.uav.max_speed_mps)
        max_move = speed * self.dt
        direction = delta / distance.clamp_min(1e-12).unsqueeze(-1)
        full_steps = torch.floor(distance / max(max_move, 1e-12))
        remainder = (distance - full_steps * max_move).clamp_min(0.0)
        full_velocity = direction * speed
        final_velocity = direction * (remainder / self.dt).unsqueeze(-1)
        full_duration = full_steps * self.dt
        final_duration = torch.where(remainder > 0.0, self.dt, 0.0)
        # Valid service points are already bounded. For an externally supplied
        # out-of-map endpoint, predict only the actual clipped step, then rest;
        # never invent movement through the map boundary.
        clipped_target = (target != bounded_target).any(dim=-1)
        full_velocity = torch.where(
            clipped_target.unsqueeze(-1), (next_xy - self.uav_xy) / self.dt, full_velocity
        )
        full_duration = torch.where(clipped_target, self.dt, full_duration)
        final_duration = torch.where(clipped_target, 0.0, final_duration)
        durations = (full_duration, final_duration)
        velocities = (full_velocity, final_velocity)
        rsu_velocity = torch.zeros((self.b, self.r, 2), device=self.device, dtype=self.uav_xy.dtype)
        rsu_duration = torch.zeros((self.b, self.r), device=self.device, dtype=self.uav_xy.dtype)
        return tuple(
            (
                torch.cat([rsu_velocity, torch.where(feasible.unsqueeze(-1), velocity, 0.0)], dim=1).unsqueeze(1),
                torch.cat([rsu_duration, torch.where(feasible, duration, 0.0)], dim=1).unsqueeze(1),
            )
            for velocity, duration in zip(velocities, durations)
        )

    def _advance_uav_motion(self) -> None:
        """Integrate one dt of travel/hover; independent of dispatch frequency."""
        _, next_xy, moved, active, proposed, feasible = self._uav_motion_plan()
        self.uav_xy = torch.where(feasible.unsqueeze(-1), next_xy, self.uav_xy)
        self.pending_flight_energy += torch.where(feasible, proposed, 0.0)
        self.uav_energy_used += torch.where(feasible, proposed, 0.0)
        self.uav_available = active & feasible
        self.uav_move_distance_m += (moved * feasible).sum(dim=1)

    def _consume_pending_flight_energy(self) -> torch.Tensor:
        energy = self.pending_flight_energy.sum(dim=1)
        self.pending_flight_energy.zero_()
        return energy

    def estimate_action_costs(self) -> torch.Tensor:
        gain, feasible, _, _ = self._link_physics()
        _, selected, has_ready, _ = self._ready_state()
        cycles = self._head_tensor(self.task_cycles).gather(
            2, selected.unsqueeze(-1)
        ).squeeze(-1)
        data_mb = self._upload_payload_mb(selected)
        now = self.time.unsqueeze(1)
        local_wait = (self.node_available[:, : self.v] - now).clamp_min(0.0)
        local_delay = local_wait + compute_duration(
            cycles,
            self.cpu_hz[: self.v].unsqueeze(0),
        )
        local_energy = compute_energy(
            cycles,
            float(self.cfg.compute.vehicle_energy_per_gcycle_j),
        )
        unit_rate = shannon_rate(
            float(self.cfg.radio.bandwidth_mhz) * 1e6,
            torch.ones_like(gain),
            float(self.cfg.radio.tx_power_w),
            gain,
            self.noise_psd_w_hz,
        )
        remote_wait = (
            self.node_available[:, self.v :].unsqueeze(1) - now.unsqueeze(-1)
        ).clamp_min(0.0)
        upload = data_mb.unsqueeze(-1) * 8e6 / unit_rate.clamp_min(1.0)
        output_mb = self._head_tensor(self.task_output_mb).gather(2, selected.unsqueeze(-1)).squeeze(-1)
        return_rate = unit_rate.clamp_min(1.0).clamp_max(float(self.cfg.radio.return_mbps) * 1e6)
        result_return = output_mb.unsqueeze(-1) * 8e6 / return_rate
        remote_duration = compute_duration(
            cycles.unsqueeze(-1),
            self.cpu_hz[self.v :].view(1, 1, -1),
        )
        remote_delay = upload + remote_wait + remote_duration + result_return
        energy_per_gcycle = _tensor(
            [self.cfg.compute.rsu_energy_per_gcycle_j] * self.r
            + [self.cfg.compute.uav_energy_per_gcycle_j] * self.u,
            self.device,
        )
        remote_energy = (
            float(self.cfg.radio.tx_power_w) * (upload + result_return)
            + compute_energy(
                cycles.unsqueeze(-1),
                energy_per_gcycle.view(1, 1, -1),
            )
        )
        delay = torch.cat([local_delay.unsqueeze(-1), remote_delay], dim=-1)
        energy = torch.cat([local_energy.unsqueeze(-1), remote_energy], dim=-1)
        feasible_full = torch.cat(
            [torch.ones((self.b, self.v, 1), dtype=torch.bool, device=self.device), feasible],
            dim=-1,
        )
        delay = torch.where(feasible_full & has_ready.unsqueeze(-1), delay, torch.full_like(delay, 1e6))
        energy = torch.where(feasible_full & has_ready.unsqueeze(-1), energy, torch.full_like(energy, 1e6))
        return torch.stack([delay, energy], dim=-1)

    def _execution_finishes(
        self,
        schedule: torch.Tensor,
        node_index: torch.Tensor,
        ready_time: torch.Tensor,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        """Reserve local serial processors; remote jobs enter explicit queues."""
        finish = torch.full_like(ready_time, torch.inf)
        local = schedule & (node_index < self.v)
        local_available = self.node_available[:, : self.v]
        local_start = torch.maximum(ready_time, local_available)
        local_finish = local_start + duration
        finish = torch.where(local, local_finish, finish)
        self.node_available[:, : self.v] = torch.where(
            local,
            local_finish,
            local_available,
        )

        return finish

    def _complete_local_and_returns(self, until: torch.Tensor) -> None:
        deadline = until.view(self.b, 1, 1, 1)
        local = (
            self.sub_running
            & (self.sub_node >= 0)
            & (self.sub_node < self.v)
            & (self.sub_finish <= deadline)
        )
        returned = self.sub_returning & (self.sub_finish <= deadline)
        finished = local | returned
        self.sub_done |= finished
        self.sub_running &= ~finished
        self.sub_returning &= ~returned
        self.metrics.delivered_results += returned.sum(dim=(1, 2, 3)).float()

    def _advance_remote_queues(self, slot_end: torch.Tensor) -> torch.Tensor:
        """Advance single-server non-preemptive priority queues causally."""
        jobs = self.v * self.q * self.s
        queued = self.sub_remote_queued.view(self.b, jobs)
        computing = self.sub_remote_computing.view(self.b, jobs)
        running = self.sub_running.view(self.b, jobs)
        done = self.sub_done.view(self.b, jobs)
        finish = self.sub_finish.view(self.b, jobs)
        node = self.sub_node.view(self.b, jobs)
        ready_time = self.sub_service_ready.view(self.b, jobs)
        duration = self.sub_service_duration.view(self.b, jobs)
        priority = self.task_priority.view(self.b, jobs)
        sink = self.sub_is_sink.view(self.b, jobs)
        return_pending = self.sub_return_pending.view(self.b, jobs)
        new_return_pending = torch.zeros_like(return_pending)
        epsilon = 1e-7

        for resource in range(self.remote_n):
            resource_node = self.v + resource
            # Calls recur as DAG successors are released inside one slot.
            # Never rewind a server that already processed work in this slot.
            cursor = torch.maximum(self.time, self.node_available[:, resource_node])
            resource_mask = node == resource_node
            for _ in range(jobs + 1):
                active = computing & resource_mask
                active_finish = finish.masked_fill(~active, torch.inf).min(dim=1).values
                completes = active.any(dim=1) & (active_finish <= slot_end + epsilon)
                completed_job = active & torch.isclose(
                    finish,
                    active_finish.unsqueeze(1),
                    atol=epsilon,
                    rtol=0.0,
                ) & completes.unsqueeze(1)
                if bool(completed_job.any()):
                    requires_delivery = sink if self.completion_only_dependencies else torch.ones_like(sink)
                    completed_sink = completed_job & requires_delivery
                    completed_internal = completed_job & ~requires_delivery
                    computing[completed_job] = False
                    queued[completed_job] = False
                    done[completed_internal] = True
                    running[completed_internal] = False
                    return_pending[completed_sink] = True
                    new_return_pending[completed_sink] = True
                    # Preserve the compute-completion event time.  A causal
                    # return can start at this instant inside the same slot.
                    finish[completed_sink] = active_finish.unsqueeze(1).expand_as(
                        finish
                    )[completed_sink]
                    self.metrics.buffered_results += completed_sink.sum(dim=1).float()
                    cursor = torch.where(completes, torch.maximum(cursor, active_finish), cursor)

                still_active = computing & resource_mask
                idle = ~still_active.any(dim=1)
                waiting = queued & resource_mask
                has_waiting = waiting.any(dim=1)
                earliest = ready_time.masked_fill(~waiting, torch.inf).min(dim=1).values
                arrived = waiting & (ready_time <= cursor.unsqueeze(1) + epsilon)
                has_arrived = arrived.any(dim=1)
                start = torch.where(has_arrived, cursor, earliest)
                can_start = idle & has_waiting & (start < slot_end - epsilon)
                eligible = waiting & (ready_time <= start.unsqueeze(1) + epsilon)
                eligible &= can_start.unsqueeze(1)

                min_priority = priority.masked_fill(
                    ~eligible,
                    torch.iinfo(priority.dtype).max,
                ).min(dim=1).values
                eligible &= priority == min_priority.unsqueeze(1)
                min_ready = ready_time.masked_fill(~eligible, torch.inf).min(dim=1).values
                eligible &= torch.isclose(
                    ready_time,
                    min_ready.unsqueeze(1),
                    atol=epsilon,
                    rtol=0.0,
                )
                selected = eligible.float().argmax(dim=1)
                batch_index = torch.nonzero(can_start, as_tuple=False).flatten()
                if len(batch_index) > 0:
                    job_index = selected[batch_index]
                    service_finish = start[batch_index] + duration[
                        batch_index,
                        job_index,
                    ]
                    queued[batch_index, job_index] = False
                    computing[batch_index, job_index] = True
                    finish[batch_index, job_index] = service_finish
                    self.node_available[batch_index, resource_node] = service_finish

                changed = completes | can_start
                if not bool(changed.any()):
                    break

            active = computing & resource_mask
            active_finish = finish.masked_fill(~active, torch.inf).min(dim=1).values
            self.node_available[:, resource_node] = torch.where(
                active.any(dim=1),
                active_finish,
                torch.maximum(cursor, self.time),
            )
        return new_return_pending.view_as(self.sub_return_pending)

    def _attempt_buffered_returns(
        self,
        pending_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attempt whole-packet intermediate or sink delivery from current state."""
        pending = self.sub_return_pending
        if pending_mask is not None:
            pending = pending & pending_mask
        vehicle_energy = torch.zeros((self.b, self.v), device=self.device)
        if not bool(pending.any()):
            return vehicle_energy

        rates, feasible, _ = self.link_state()
        resource_xy, _, _ = self._resource_geometry()
        _, _, _, range_limit = self._link_physics()
        predicted_contact = self._contact_horizon(
            resource_xy,
            range_limit,
            predicted=True,
        )
        true_contact = self._contact_horizon(
            resource_xy,
            range_limit,
            predicted=False,
        )
        resource_index = (self.sub_node - self.v).clamp(0, self.remote_n - 1)

        def select_resource(values: torch.Tensor) -> torch.Tensor:
            expanded = values.unsqueeze(2).unsqueeze(3).expand(
                self.b,
                self.v,
                self.q,
                self.s,
                self.remote_n,
            )
            return expanded.gather(
                -1,
                resource_index.unsqueeze(-1),
            ).squeeze(-1)

        selected_rate = select_resource(rates).clamp_min(1.0)
        selected_feasible = select_resource(feasible)
        # A newly computed result can return immediately at its continuous
        # completion time.  Geometry is quasi-static inside the micro-slot, so
        # the contact horizon observed at slot start is reduced by the elapsed
        # within-slot offset.
        slot_start = self.time.view(self.b, 1, 1, 1)
        return_ready_time = torch.where(
            torch.isfinite(self.sub_finish),
            self.sub_finish,
            slot_start,
        )
        attempt_time = torch.maximum(slot_start, return_ready_time)
        elapsed = (attempt_time - slot_start).clamp_min(0.0)
        selected_predicted_contact = (
            select_resource(predicted_contact) - elapsed
        ).clamp_min(0.0)
        selected_true_contact = (
            select_resource(true_contact) - elapsed
        ).clamp_min(0.0)
        return_rate = torch.minimum(
            selected_rate,
            torch.full_like(
                selected_rate,
                float(self.cfg.radio.return_mbps) * 1e6,
            ),
        )
        return_duration = (
            self.task_output_mb
            * 8e6
            / return_rate
        )
        attempt = (
            pending
            & selected_feasible
            & (return_duration <= selected_predicted_contact)
        )
        success = attempt & (return_duration <= selected_true_contact)
        failed = attempt & ~success
        deferred = pending & ~attempt
        transmit_time = torch.where(
            success,
            return_duration,
            torch.minimum(return_duration, selected_true_contact.clamp_min(0.0)),
        )
        transmit_energy = torch.where(
            attempt,
            float(self.cfg.radio.tx_power_w) * transmit_time,
            0.0,
        )
        vehicle_energy += transmit_energy.sum(dim=(2, 3))

        self.sub_return_pending &= ~success
        self.sub_returning |= success
        self.sub_finish = torch.where(
            success,
            attempt_time + return_duration,
            self.sub_finish,
        )
        self.metrics.return_attempts += attempt.sum(dim=(1, 2, 3)).float()
        self.metrics.return_retries += failed.sum(dim=(1, 2, 3)).float()
        self.metrics.return_deferrals += deferred.sum(dim=(1, 2, 3)).float()
        return vehicle_energy

    def _record_hotspot_coverage(self) -> None:
        """Diagnostic only: remaining work in busiest region physically in UAV range."""
        slots = torch.arange(self.q, device=self.device).view(1, 1, self.q)
        active = (slots - self.queue_head.unsqueeze(-1)) % self.q < self.queue_count.unsqueeze(-1)
        work = (self.task_cycles * (~self.sub_done) * active.unsqueeze(-1)).sum(dim=(-1, -2)) / 1e9
        regions = self.vehicle_region_index()
        totals = torch.zeros((self.b, self.region_n), device=self.device)
        totals.scatter_add_(1, regions, work)
        hottest = totals.argmax(dim=1)
        hotspot_work = work * (regions == hottest.unsqueeze(-1))
        _, feasible, _, _ = self._link_physics()
        covered = feasible[..., self.r:].any(dim=-1)
        self.hotspot_total_work += hotspot_work.sum(dim=1)
        self.hotspot_covered_work += (hotspot_work * covered).sum(dim=1)

    def step(
        self,
        actions: torch.Tensor,
        uav_dispatch: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if uav_dispatch is not None:
            self.set_uav_dispatch(uav_dispatch)
        self._record_hotspot_coverage()
        action_mask = self.action_mask(delivery_aware=False)
        actions = actions.long().clamp(0, self.action_n - 1)
        _, _, has_ready_before_action, _ = self._ready_state()
        selected_action_valid = action_mask.gather(
            -1,
            actions.unsqueeze(-1),
        ).squeeze(-1)
        self.metrics.invalid_actions += (
            has_ready_before_action & ~selected_action_valid
        ).sum(dim=1)
        self.metrics.action_decisions += has_ready_before_action.sum(dim=1)
        actions = torch.where(
            selected_action_valid,
            actions,
            torch.zeros_like(actions),
        )
        channel_gain, _, _, range_limit = self._link_physics()
        resource_xy, _, _ = self._resource_geometry()
        true_contact_time = self._contact_horizon(
            resource_xy,
            range_limit,
            predicted=False,
        )
        step_energy = torch.zeros(self.b, device=self.device)
        vehicle_energy = torch.zeros((self.b, self.v), device=self.device)
        buffered_return_energy = self._attempt_buffered_returns()
        vehicle_energy += buffered_return_energy
        step_energy += buffered_return_energy.sum(dim=1)
        slot_end = self.time + self.dt
        dispatch_blocked = torch.zeros(
            (self.b, self.v),
            dtype=torch.bool,
            device=self.device,
        )
        for _ in range(self.s + 1):
            self._complete_local_and_returns(slot_end)
            new_returns = self._advance_remote_queues(slot_end)
            if bool(new_returns.any()):
                immediate_return_energy = self._attempt_buffered_returns(
                    new_returns
                )
                vehicle_energy += immediate_return_energy
                step_energy += immediate_return_energy.sum(dim=1)
                self._complete_local_and_returns(slot_end)
            ready, selected, has_ready, adjacency = self._ready_state()
            schedule = has_ready & ~dispatch_blocked
            if not schedule.any():
                break

            q_index = self.queue_head.unsqueeze(-1)
            selected_index = selected.unsqueeze(-1)
            cycles = self._head_tensor(self.task_cycles).gather(
                2,
                selected_index,
            ).squeeze(-1)
            data_mb = self._upload_payload_mb(selected)
            predecessor_mask = adjacency.gather(
                3, selected.view(self.b, self.v, 1, 1).expand(-1, -1, self.s, 1)
            ).squeeze(-1)
            predecessor_finish = self._head_tensor(self.sub_finish)
            predecessor_ready = predecessor_finish.masked_fill(~predecessor_mask, -torch.inf).max(dim=-1).values
            upload_start = torch.maximum(self.time.unsqueeze(1), predecessor_ready)
            upload_offset = upload_start - self.time.unsqueeze(1)
            priority_class = self._head_tensor(self.task_priority).gather(
                2,
                selected_index,
            ).squeeze(-1)
            priority_weight = lookup_priority_weight(
                priority_class,
                self.priority_weights,
            )
            current_mask = self.action_mask(delivery_aware=False)
            current_valid = current_mask.gather(
                -1,
                actions.unsqueeze(-1),
            ).squeeze(-1)
            chosen = torch.where(current_valid, actions, torch.zeros_like(actions))
            remote = schedule & (chosen > 0)
            is_sink = ~adjacency.gather(
                2,
                selected.view(self.b, self.v, 1, 1).expand(-1, -1, 1, self.s),
            ).squeeze(2).any(dim=-1)
            vehicle_index = torch.arange(self.v, device=self.device).view(1, self.v)
            node_index = torch.where(
                chosen == 0,
                vehicle_index.expand(self.b, -1),
                self.v + chosen - 1,
            )

            remote_index = (chosen - 1).clamp_min(0)
            uav_compute_energy = compute_energy(
                cycles,
                float(self.cfg.compute.uav_energy_per_gcycle_j),
            )
            uav_request = remote & (remote_index >= self.r)
            if uav_request.any():
                uav_index = (remote_index - self.r).clamp(0, self.u - 1)
                grouped_uav_energy = torch.zeros(
                    (self.b, self.u),
                    dtype=cycles.dtype,
                    device=self.device,
                )
                grouped_uav_energy.scatter_add_(
                    1,
                    uav_index,
                    torch.where(uav_request, uav_compute_energy, 0.0),
                )
                battery_limit = (
                    float(self.cfg.uav.battery_capacity_j)
                    * float(self.cfg.uav.battery_usage_ratio)
                )
                remaining_uav_energy = (
                    battery_limit - self.uav_energy_used
                ).clamp_min(0.0)
                overloaded_uav = grouped_uav_energy > remaining_uav_energy
                rejected_uav = uav_request & overloaded_uav.gather(1, uav_index)
                chosen = torch.where(
                    rejected_uav,
                    torch.zeros_like(chosen),
                    chosen,
                )
                remote = schedule & (chosen > 0)
                node_index = torch.where(
                    chosen == 0,
                    vehicle_index.expand(self.b, -1),
                    self.v + chosen - 1,
                )
                remote_index = (chosen - 1).clamp_min(0)
            share = weighted_proportional_share(
                remote,
                remote_index,
                priority_weight,
                self.remote_n,
            )
            selected_gain = channel_gain.gather(
                2,
                remote_index.unsqueeze(-1),
            ).squeeze(-1)
            selected_rate = shannon_rate(
                float(self.cfg.radio.bandwidth_mhz) * 1e6,
                share,
                float(self.cfg.radio.tx_power_w),
                selected_gain,
                self.noise_psd_w_hz,
            )
            upload = torch.where(
                remote,
                data_mb * 8e6 / selected_rate.clamp_min(1.0),
                0.0,
            )
            selected_contact = true_contact_time.gather(
                2,
                remote_index.unsqueeze(-1),
            ).squeeze(-1)
            selected_contact = (selected_contact - upload_offset).clamp_min(0.0)
            joint_contact_failure = remote & (upload > selected_contact)
            if joint_contact_failure.any():
                interrupted_time = torch.minimum(
                    upload,
                    selected_contact.clamp_min(0.0),
                )
                interrupted_energy = torch.where(
                    joint_contact_failure,
                    float(self.cfg.radio.tx_power_w) * interrupted_time,
                    0.0,
                )
                vehicle_energy += interrupted_energy
                step_energy += interrupted_energy.sum(dim=1)
                self.metrics.upload_interruptions += joint_contact_failure.sum(
                    dim=1
                ).float()
                dispatch_blocked |= joint_contact_failure
                remote &= ~joint_contact_failure
                schedule &= ~joint_contact_failure
                if not bool(schedule.any()):
                    continue
                share = weighted_proportional_share(
                    remote,
                    remote_index,
                    priority_weight,
                    self.remote_n,
                )
                selected_gain = channel_gain.gather(
                    2,
                    remote_index.unsqueeze(-1),
                ).squeeze(-1)
                selected_rate = shannon_rate(
                    float(self.cfg.radio.bandwidth_mhz) * 1e6,
                    share,
                    float(self.cfg.radio.tx_power_w),
                    selected_gain,
                    self.noise_psd_w_hz,
                )
                upload = torch.where(
                    remote,
                    data_mb * 8e6 / selected_rate.clamp_min(1.0),
                    0.0,
                )

            predecessor_nodes = self._head_tensor(self.sub_node)
            different_predecessor = predecessor_mask & (predecessor_nodes != node_index.unsqueeze(-1))
            # Physical model: every remote output is delivered to its owning UGV
            # before readiness; successor uploads include all predecessor bytes.
            # The completion-only diagnostic intentionally ignores those bytes.
            # A successor's packet cannot start uploading before the producing
            # predecessors finish delivering. The upload then takes extra time.
            ready_time = upload_start + upload

            selected_capacity = self.cpu_hz[self.v + remote_index]
            allocated_frequency = torch.where(
                remote,
                selected_capacity,
                self.cpu_hz[node_index],
            )
            duration = compute_duration(
                cycles,
                allocated_frequency,
            )
            finish_exec = self._execution_finishes(
                schedule,
                node_index,
                ready_time,
                duration,
            )
            finish = torch.where(remote, torch.full_like(finish_exec, torch.inf), finish_exec)

            migration_edges = different_predecessor.sum(dim=-1).float()
            migration_energy = (
                migration_edges * float(self.cfg.uav.migration_energy_j)
            )

            b_idx, v_idx = schedule.nonzero(as_tuple=True)
            q_idx = q_index.expand(-1, -1, 1)[b_idx, v_idx, 0]
            s_idx = selected[b_idx, v_idx]
            self.sub_running[b_idx, v_idx, q_idx, s_idx] = True
            self.sub_finish[b_idx, v_idx, q_idx, s_idx] = finish[b_idx, v_idx]
            self.sub_node[b_idx, v_idx, q_idx, s_idx] = node_index[b_idx, v_idx]
            scheduled_remote = remote[b_idx, v_idx]
            if bool(scheduled_remote.any()):
                remote_b = b_idx[scheduled_remote]
                remote_v = v_idx[scheduled_remote]
                remote_q = q_idx[scheduled_remote]
                remote_s = s_idx[scheduled_remote]
                self.sub_remote_queued[
                    remote_b,
                    remote_v,
                    remote_q,
                    remote_s,
                ] = True
                self.sub_service_ready[
                    remote_b,
                    remote_v,
                    remote_q,
                    remote_s,
                ] = ready_time[remote_b, remote_v]
                self.sub_service_duration[
                    remote_b,
                    remote_v,
                    remote_q,
                    remote_s,
                ] = duration[remote_b, remote_v]
                self.sub_is_sink[
                    remote_b,
                    remote_v,
                    remote_q,
                    remote_s,
                ] = is_sink[remote_b, remote_v]

            task_energy_per_gcycle = torch.where(
                chosen == 0,
                torch.full_like(
                    cycles,
                    float(self.cfg.compute.vehicle_energy_per_gcycle_j),
                ),
                torch.where(
                    chosen <= self.r,
                    torch.full_like(
                        cycles,
                        float(self.cfg.compute.rsu_energy_per_gcycle_j),
                    ),
                    torch.full_like(
                        cycles,
                        float(self.cfg.compute.uav_energy_per_gcycle_j),
                    ),
                ),
            )
            task_compute_energy = compute_energy(
                cycles,
                task_energy_per_gcycle,
            )
            comm_energy = torch.where(
                remote,
                float(self.cfg.radio.tx_power_w) * upload,
                0.0,
            )
            task_energy = torch.where(
                schedule,
                task_compute_energy + comm_energy + migration_energy,
                0.0,
            )
            step_energy += task_energy.sum(dim=1)
            vehicle_energy += task_energy
            uav_selected = schedule & (chosen > self.r)
            if uav_selected.any():
                uav_index = (chosen - self.r - 1).clamp(0, self.u - 1)
                uav_energy = torch.zeros(
                    (self.b, self.u),
                    dtype=task_energy.dtype,
                    device=self.device,
                )
                uav_energy.scatter_add_(
                    1,
                    uav_index,
                    torch.where(uav_selected, task_compute_energy, 0.0),
                )
                self.uav_energy_used += uav_energy
            self.metrics.local_actions += (schedule & (chosen == 0)).sum(dim=1)
            self.metrics.rsu_actions += (
                schedule & (chosen > 0) & (chosen <= self.r)
            ).sum(dim=1)
            self.metrics.uav_actions += (
                schedule & (chosen > self.r)
            ).sum(dim=1)

        self._complete_local_and_returns(slot_end)
        new_returns = self._advance_remote_queues(slot_end)
        if bool(new_returns.any()):
            immediate_return_energy = self._attempt_buffered_returns(new_returns)
            vehicle_energy += immediate_return_energy
            step_energy += immediate_return_energy.sum(dim=1)
        self._complete_local_and_returns(slot_end)
        head_active = self.queue_count > 0
        head_arrival = self._head_tensor(self.task_arrival)
        valid_nodes = self.workflow_node_mask[self._head_tensor(self.task_template)]
        head_finish = self._head_tensor(self.sub_finish).masked_fill(~valid_nodes, -torch.inf).max(dim=-1).values
        all_nodes_done = (self._head_tensor(self.sub_done) | ~valid_nodes).all(dim=-1)
        completion_event = all_nodes_done & torch.isfinite(head_finish) & head_active
        # Integrate the actual head-job occupancy, preserving sub-second trace
        # runtimes. Tail jobs wait until the next control tick in this slotted
        # scheduler, even if the head finishes partway through the current tick.
        head_end = torch.minimum(
            torch.where(completion_event, head_finish, slot_end.unsqueeze(1)),
            head_arrival + self.deadline,
        )
        head_occupancy = (head_end - self.time.unsqueeze(1)).clamp(0.0, self.dt) * head_active
        delay_cost = (head_occupancy + (self.queue_count.float() - 1).clamp_min(0.0) * self.dt).sum(dim=1)
        self._advance_uav_motion()
        flight_energy = self._consume_pending_flight_energy()
        step_energy += flight_energy
        self.metrics.uav_flight_energy_j += flight_energy
        self.time = slot_end

        head_done = completion_event & (head_finish <= head_arrival + self.deadline)
        head_age = self.time.view(self.b, 1) - head_arrival
        timed_out = (head_age >= self.deadline) & (self.queue_count > 0) & ~head_done
        self.metrics.completed += head_done.sum(dim=1).float()
        self.metrics.timed_out += timed_out.sum(dim=1).float()
        completion_latency = torch.where(head_done, head_finish - head_arrival, 0.0)
        self.metrics.completion_latency += completion_latency.sum(dim=1)
        self.aoi = torch.where(
            head_done,
            completion_latency,
            self.aoi + self.dt,
        )
        terminal = head_done | timed_out
        terminal_index = torch.nonzero(terminal, as_tuple=False)
        if len(terminal_index) > 0:
            terminal_b = terminal_index[:, 0]
            terminal_v = terminal_index[:, 1]
            terminal_q = self.queue_head[terminal_b, terminal_v]
            self.sub_running[terminal_b, terminal_v, terminal_q] = False
            self.sub_remote_queued[terminal_b, terminal_v, terminal_q] = False
            self.sub_remote_computing[terminal_b, terminal_v, terminal_q] = False
            self.sub_return_pending[terminal_b, terminal_v, terminal_q] = False
            self.sub_returning[terminal_b, terminal_v, terminal_q] = False
        self.queue_head = torch.where(terminal, (self.queue_head + 1) % self.q, self.queue_head)
        self.queue_count = (self.queue_count - terminal.long()).clamp_min(0)

        self.step_index += 1
        self._advance_vehicles()
        self._sample_geometry_error()
        self.contact_prediction_noise.uniform_(
            -1.0,
            1.0,
            generator=self.generator,
        )
        self._update_contact_history()
        row = self.traffic[(self.traffic_index + self.step_index).clamp_max(len(self.traffic) - 1)]
        local_flow = row.gather(1, self.sensor)
        macro_scale = 0.75 + 0.5 * self.uci_scale[
            (self.uci_index + self.step_index).clamp_max(len(self.uci_scale) - 1)
        ].unsqueeze(1)
        arrival_rate = (
            float(self.cfg.system.base_arrival_rate)
            + float(self.cfg.system.hotspot_arrival_rate) * local_flow
        ) * macro_scale * self.load_multiplier.unsqueeze(1)
        arrivals = torch.rand(
            (self.b, self.v),
            generator=self.generator,
            device=self.device,
        ) < (arrival_rate * self.dt).clamp_max(0.95)
        # Reserve a deadline-long drain window.  Without it, tasks created in
        # the final slots are right-censored by construction and a completion
        # gate compares policies against an unattainable denominator.
        drain_seconds = float(
            self.cfg.system.get(
                "arrival_drain_seconds",
                self.cfg.system.deadline_seconds,
            )
        )
        arrival_steps = max(
            self.horizon - int(math.ceil(drain_seconds / self.dt)),
            0,
        )
        arrivals &= self.step_index <= arrival_steps
        dropped_mask = self._enqueue(arrivals)
        new_dropped = dropped_mask.sum(dim=1).float()

        vehicle_timeout_cost = timed_out.float() * float(self.cfg.system.timeout_penalty_seconds)
        if self.step_index >= self.horizon:
            # Do not make unresolved work free by ending the simulation. Keep
            # it explicitly separate from observed deadline misses (especially
            # in short smoke horizons), but charge the same fixed failure cost.
            unfinished = self.queue_count.float()
            self.metrics.terminal_unfinished += unfinished.sum(dim=1)
            vehicle_timeout_cost += unfinished * float(self.cfg.system.timeout_penalty_seconds)
        vehicle_drop_cost = dropped_mask.float() * float(self.cfg.system.queue_drop_penalty_seconds)
        timeout_cost = vehicle_timeout_cost.sum(dim=1)
        drop_cost = vehicle_drop_cost.sum(dim=1)
        delay_objective = delay_cost + timeout_cost + drop_cost
        reward = scalarized_delay_energy_reward(
            self.preference,
            delay_objective,
            step_energy,
            delay_bounds=(
                float(self.cfg.reward.delay_min),
                float(self.cfg.reward.delay_max),
            ),
            energy_bounds=(
                float(self.cfg.reward.energy_min_j),
                float(self.cfg.reward.energy_max_j),
            ),
        )
        # Identical team return, including flight and failed transfers, for every
        # vehicle actor and the coordinator. No direction bonuses or local proxy.
        vehicle_reward = reward.unsqueeze(1).expand(-1, self.v)
        self.metrics.reward += reward
        self.metrics.delay_cost += delay_objective
        self.metrics.energy_j += step_energy
        done = torch.full(
            (self.b,),
            self.step_index >= self.horizon,
            dtype=torch.bool,
            device=self.device,
        )
        info = {
            "delay_cost": delay_objective,
            "energy_j": step_energy,
            "completed": head_done.sum(dim=1),
            "timed_out": timed_out.sum(dim=1),
            "dropped": new_dropped,
            "vehicle_reward": vehicle_reward,
            "aoi": self.aoi.mean(dim=1),
        }
        return self.observe(), reward, done, info
