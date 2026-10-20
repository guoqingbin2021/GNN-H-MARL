# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .config import Config
from .variants import get_variant, macro_interval


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(hidden_dim, output_dim), nn.SiLU()])
    return nn.Sequential(*layers)


def _masked_logit_value(tensor: torch.Tensor) -> float:
    if tensor.dtype.is_floating_point:
        return float(torch.finfo(tensor.dtype).min)
    return -1e9


class DAGAttentionEncoder(nn.Module):
    """Encode both dependency directions and retain the current node context.

    ``use_edges=False`` is the topology ablation: every node receives only its
    own message, with exactly the same features, transforms and readout.  It
    therefore remains a capable set encoder without observing dependency edges.
    """

    def __init__(
        self, node_dim: int, hidden_dim: int, layers: int, use_edges: bool = True,
    ):
        super().__init__()
        self.use_edges = use_edges
        self.input = nn.Linear(node_dim, hidden_dim)
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)]
        )
        self.attention = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim * 2)) for _ in range(layers)]
        )
        self.successor_attention = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim * 2)) for _ in range(layers)]
        )
        self.direction_mix = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim, bias=False) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.readout = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU())
        for vector in [*self.attention, *self.successor_attention]:
            nn.init.xavier_uniform_(vector.view(1, -1))

    @staticmethod
    def _messages(
        projected: torch.Tensor, attention: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = projected.shape[-1]
        # Factoring additive attention avoids the [batch, vehicle, N, N, 2H]
        # concatenation, which is prohibitively large for real workflows.
        destination = projected @ attention[:hidden]
        source = projected @ attention[hidden:]
        score = F.leaky_relu(
            destination.unsqueeze(-1) + source.unsqueeze(-2), negative_slope=0.2,
        )
        score = score.masked_fill(~mask, _masked_logit_value(score))
        return torch.softmax(score, dim=-1) @ projected

    @staticmethod
    def _mean(h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (h * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)

    def forward(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        selected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_mask is None:
            node_mask = torch.ones_like(nodes[..., 0], dtype=torch.bool)
        else:
            node_mask = node_mask.bool()
        valid = node_mask.unsqueeze(-1).to(nodes.dtype)
        h = F.silu(self.input(nodes)) * valid
        node_count = h.shape[-2]
        identity = torch.eye(
            node_count,
            dtype=torch.bool,
            device=h.device,
        )
        edges = adjacency.bool() if self.use_edges else torch.zeros_like(adjacency, dtype=torch.bool)
        edges = edges & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        predecessor_mask = edges.transpose(-1, -2) | identity
        successor_mask = edges | identity
        for projection, attention, successor_attention, direction_mix, norm in zip(
            self.projections,
            self.attention,
            self.successor_attention,
            self.direction_mix,
            self.norms,
        ):
            projected = projection(h)
            if self.use_edges:
                predecessor = self._messages(projected, attention, predecessor_mask)
                successor = self._messages(projected, successor_attention, successor_mask)
            else:
                predecessor = successor = projected
            aggregated = direction_mix(torch.cat([predecessor, successor], dim=-1))
            h = norm(h + F.elu(aggregated)) * valid
        unfinished = (1.0 - nodes[..., 2:3]).clamp(0.0, 1.0) * valid
        ready = nodes[..., 4:5].clamp(0.0, 1.0) * unfinished
        ready_context = self._mean(h, ready)
        if selected is None:
            selected_context = ready_context
        else:
            index = selected.long().unsqueeze(-1).unsqueeze(-1)
            selected_context = h.gather(
                -2, index.expand(*index.shape[:-1], h.shape[-1]),
            ).squeeze(-2)
            selected_context = selected_context * ready.gather(-2, index).squeeze(-2)
        return self.readout(torch.cat(
            [self._mean(h, unfinished), ready_context, selected_context], dim=-1,
        )) * node_mask.any(dim=-1, keepdim=True)


@dataclass
class PolicyOutput:
    actions: torch.Tensor
    dispatch: torch.Tensor
    vehicle_logprob: torch.Tensor
    uav_gate_logprob: torch.Tensor
    uav_target_logprob: torch.Tensor
    vehicle_entropy: torch.Tensor
    uav_gate_entropy: torch.Tensor
    uav_target_entropy: torch.Tensor
    vehicle_value: torch.Tensor
    global_value: torch.Tensor


@dataclass
class VehiclePolicyOutput:
    actions: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


@dataclass
class UAVPolicyOutput:
    dispatch: torch.Tensor
    gate_logprob: torch.Tensor
    target_logprob: torch.Tensor
    gate_entropy: torch.Tensor
    target_entropy: torch.Tensor
    value: torch.Tensor


class PreferencePolicy(nn.Module):
    """Shared actor with centralized critic and optional DAG message passing."""

    def __init__(self, config: Config, kind: str = "uamco"):
        super().__init__()
        self.kind = kind
        self.variant = get_variant(kind)
        self.v = int(config.system.num_vehicles)
        self.r = int(config.system.num_rsus)
        self.u = int(config.system.num_uavs)
        self.region_n = int(config.system.get("demand_region_rows", 5)) * int(
            config.system.get("demand_region_cols", 5)
        )
        self.queue_size = int(config.system.task_queue_size)
        self.subtasks = int(config.system.num_subtasks)
        self.max_subtask_cycles_g = float(config.task.max_cycles_g)
        self.macro_duration = (
            float(macro_interval(kind, int(config.system.macro_steps)))
            * float(config.system.dt_seconds)
        )
        self.hidden = int(config.model.hidden_dim)
        dropout = float(config.model.dropout)
        self.use_dag = self.variant.dag_encoder != "none"
        self.use_preference = self.variant.use_preference
        self.use_action_mask = self.variant.use_action_mask
        self.delivery_aware = self.variant.delivery_aware
        self.use_contact_feature = (
            self.delivery_aware if self.variant.contact_feature is None
            else self.variant.contact_feature
        )
        self.use_predictive_admission = (
            self.delivery_aware if self.variant.predictive_admission is None
            else self.variant.predictive_admission
        )

        self.dag_encoder = DAGAttentionEncoder(
            node_dim=9,
            hidden_dim=self.hidden,
            layers=int(config.model.dag_layers),
            use_edges=self.use_dag,
        )
        dag_dim = self.hidden

        self.vehicle_encoder = _mlp(13 + dag_dim + 2, self.hidden, self.hidden, dropout)
        self.preference_modulator = nn.Sequential(
            nn.Linear(2, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden * 2),
        )
        self.resource_encoder = _mlp(8, self.hidden, self.hidden, dropout)
        self.relative_encoder = _mlp(8, self.hidden, self.hidden, dropout)
        self.vehicle_query = nn.Linear(self.hidden, self.hidden)
        self.resource_key = nn.Linear(self.hidden, self.hidden)

        self.dispatch_uav = nn.Linear(self.hidden, self.hidden)
        self.demand_region_encoder = _mlp(9, self.hidden, self.hidden, dropout)
        self.dispatch_region = nn.Linear(self.hidden, self.hidden)
        self.dispatch_relative = _mlp(9, self.hidden, self.hidden, dropout)
        self.dispatch_residual = _mlp(2, self.hidden, self.hidden, dropout)
        self.dispatch_preference = _mlp(2, self.hidden, self.hidden, dropout)
        self.dispatch_global = nn.Linear(self.hidden * 4, self.hidden)
        self.dispatch_recurrent = nn.GRUCell(self.hidden, self.hidden)
        self.dispatch_base = nn.Parameter(torch.zeros(self.hidden))
        # A single continuously preference-conditioned actor lets all channels
        # share every transition.  The preference is already present in the
        # encoded features and FiLM modulation above; no anchor routing, prior,
        # or post-training alignment is used here.
        # Keep a direct continuous path from the objective weights to every
        # final action head.  Conditioning only the deep encoders made the
        # small preference perturbation vanish before categorical sampling;
        # these are learned inputs, not fixed channel biases or routed heads.
        self.vehicle_gate_head = nn.Linear(self.hidden * 2 + 2, 2)
        self.remote_head = nn.Linear(self.hidden + 2, 1)
        self.dispatch_head = nn.Linear(self.hidden + 2, 1)
        self.dispatch_gate = nn.Linear(self.hidden + 2, 2)

        self.critic = nn.Sequential(
            nn.Linear(self.hidden * 3 + 2, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        self.vehicle_critic = nn.Sequential(
            nn.Linear(self.hidden * 2 + 2, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )

    def _preference_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_preference:
            return obs["preference"]
        return torch.full_like(obs["preference"], 0.5)

    def _encode(self, obs: dict[str, torch.Tensor]):
        dag = self.dag_encoder(
            obs["dag_nodes"], obs["dag_adjacency"],
            node_mask=obs.get("dag_node_mask"), selected=obs.get("dag_selected"),
        )
        preference = self._preference_features(obs).unsqueeze(1).expand(-1, self.v, -1)
        vehicle = self.vehicle_encoder(torch.cat([obs["vehicle"], dag, preference], dim=-1))
        scale, shift = self.preference_modulator(
            self._preference_features(obs)
        ).chunk(2, dim=-1)
        vehicle = (
            vehicle * (1.0 + 0.5 * torch.tanh(scale).unsqueeze(1))
            + 0.5 * torch.tanh(shift).unsqueeze(1)
        )
        resource = self.resource_encoder(obs["resource"])
        return vehicle, resource

    def _vehicle_action_mask(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.use_predictive_admission:
            return obs["action_mask"]
        return obs.get("connectivity_action_mask", obs["action_mask"])

    def _pair_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pair = obs["pair"]
        if self.use_contact_feature:
            return pair
        pair = pair.clone()
        pair[..., 6] = 0.0
        return pair

    def _logits(self, obs: dict[str, torch.Tensor]):
        vehicle, resource = self._encode(obs)
        preference_features = self._preference_features(obs)
        relative = self.relative_encoder(self._pair_features(obs))
        pair = torch.tanh(
            self.vehicle_query(vehicle).unsqueeze(2)
            + self.resource_key(resource).unsqueeze(1)
            + relative
        )
        pair_preference = preference_features[:, None, None, :].expand(
            -1,
            self.v,
            pair.shape[2],
            -1,
        )
        remote_logits = self.remote_head(
            torch.cat([pair, pair_preference], dim=-1)
        ).squeeze(-1)
        actual_remote_mask = self._vehicle_action_mask(obs)[..., 1:]
        remote_mask = (
            actual_remote_mask
            if self.use_action_mask
            else torch.ones_like(actual_remote_mask)
        )
        remote_logits = remote_logits.masked_fill(
            ~remote_mask,
            _masked_logit_value(remote_logits),
        )
        remote_context = pair.masked_fill(
            ~remote_mask.unsqueeze(-1),
            -1e4,
        ).max(dim=2).values
        vehicle_gate_input = torch.cat(
            [
                vehicle,
                remote_context,
                preference_features.unsqueeze(1).expand(-1, self.v, -1),
            ],
            dim=-1,
        )
        vehicle_gate_logits = self.vehicle_gate_head(vehicle_gate_input)
        offload_gate_logits = vehicle_gate_logits[..., 1:2].masked_fill(
            ~remote_mask.any(dim=-1, keepdim=True),
            _masked_logit_value(vehicle_gate_logits),
        )
        vehicle_gate_logits = torch.cat(
            [vehicle_gate_logits[..., :1], offload_gate_logits],
            dim=-1,
        )

        resource_context = resource.mean(dim=1)
        critic_input = torch.cat(
            [
                vehicle.mean(dim=1),
                vehicle.max(dim=1).values,
                resource_context,
                preference_features,
            ],
            dim=-1,
        )
        global_value = self.critic(critic_input).squeeze(-1)
        vehicle_value_input = torch.cat(
            [
                vehicle,
                resource_context.unsqueeze(1).expand(-1, self.v, -1),
                preference_features.unsqueeze(1).expand(-1, self.v, -1),
            ],
            dim=-1,
        )
        vehicle_value = self.vehicle_critic(vehicle_value_input).squeeze(-1)
        return (
            vehicle_gate_logits,
            remote_logits,
            global_value,
            vehicle_value,
        )

    def _joint_uav_decisions(
        self,
        obs: dict[str, torch.Tensor],
        dispatch: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> UAVPolicyOutput:
        vehicle, resource = self._encode(obs)
        preference = self._preference_features(obs)
        preference_context = self.dispatch_preference(preference)
        base_region_state = obs["demand_regions"]
        regions = self.demand_region_encoder(base_region_state)
        region_mask = obs["demand_region_mask"]
        physical_region_mask = obs["uav_action_mask"][..., : self.region_n]
        mask_float = region_mask.float().unsqueeze(-1)
        region_mean = (
            (regions * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0)
        )
        region_max = regions.masked_fill(
            ~region_mask.unsqueeze(-1),
            -1e4,
        ).max(dim=1).values
        resource_context = resource.mean(dim=1)
        coordinator = torch.tanh(
            self.dispatch_global(
                torch.cat(
                    [
                        vehicle.mean(dim=1),
                        region_mean,
                        region_max,
                        resource_context,
                    ],
                    dim=-1,
                )
            )
            + preference_context
        )
        global_value = self.critic(
            torch.cat(
                [
                    vehicle.mean(dim=1),
                    vehicle.max(dim=1).values,
                    resource_context,
                    preference,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        batch = regions.shape[0]
        selected_count = torch.zeros(
            (batch, self.region_n),
            dtype=regions.dtype,
            device=regions.device,
        )
        planned_capacity_g = torch.zeros_like(selected_count)
        region_workload_g = (
            base_region_state[..., 6]
            * float(
                self.v
                * self.queue_size
                * self.subtasks
                * self.max_subtask_cycles_g
            )
        )
        dispatches = []
        gate_logprobs = []
        target_logprobs = []
        gate_entropies = []
        target_entropies = []

        for uav_id in range(self.u):
            remaining_fraction = torch.where(
                region_workload_g > 1e-6,
                (
                    (region_workload_g - planned_capacity_g).clamp_min(0.0)
                    / region_workload_g.clamp_min(1e-6)
                ),
                torch.zeros_like(region_workload_g),
            )
            uav_resource = resource[:, self.r + uav_id]
            uav_feature = self.dispatch_uav(uav_resource)
            physical_pair = obs["uav_region_pair"][:, uav_id]
            # Demand/residual workload are observations, never action filters.
            # The physical mask keeps an already-started flight non-preemptive.
            target_mask = physical_region_mask[:, uav_id]
            can_deploy = target_mask.any(dim=1)
            can_idle = obs["uav_action_mask"][:, uav_id, self.region_n]
            residual_regions = regions
            residual = torch.stack(
                [
                    selected_count / max(float(self.u), 1.0),
                    remaining_fraction,
                ],
                dim=-1,
            )
            pair = torch.tanh(
                uav_feature.unsqueeze(1)
                + self.dispatch_region(residual_regions)
                + self.dispatch_relative(physical_pair)
                + self.dispatch_residual(residual)
                + coordinator.unsqueeze(1)
                + preference_context.unsqueeze(1)
            )
            pair = pair.masked_fill(
                ~target_mask.unsqueeze(-1),
                -1e4,
            )
            target_preference = preference[:, None, :].expand(
                -1,
                self.region_n,
                -1,
            )
            target_logits = self.dispatch_head(
                torch.cat([pair, target_preference], dim=-1)
            ).squeeze(-1)
            target_logits = target_logits.masked_fill(
                ~target_mask,
                _masked_logit_value(target_logits),
            )
            target_distribution = Categorical(logits=target_logits)

            target_context = pair.max(dim=1).values
            gate_features = torch.tanh(
                uav_feature
                + target_context
                + coordinator
                + preference_context
            )
            gate_logits = self.dispatch_gate(
                torch.cat([gate_features, preference], dim=-1)
            )
            gate_logits = torch.cat(
                [
                    gate_logits[..., :1].masked_fill(
                        ~can_idle.unsqueeze(-1),
                        _masked_logit_value(gate_logits),
                    ),
                    gate_logits[..., 1:2].masked_fill(
                        ~can_deploy.unsqueeze(-1),
                        _masked_logit_value(gate_logits),
                    ),
                ],
                dim=-1,
            )
            gate_distribution = Categorical(logits=gate_logits)

            if dispatch is None:
                gate = (
                    gate_logits.argmax(dim=-1)
                    if deterministic
                    else gate_distribution.sample()
                )
                target = (
                    target_logits.argmax(dim=-1)
                    if deterministic
                    else target_distribution.sample()
                )
            else:
                gate = (dispatch[:, uav_id] < self.region_n).long()
                target = dispatch[:, uav_id].clamp_max(self.region_n - 1)

            active = gate.bool()
            action = torch.where(
                active,
                target,
                torch.full_like(target, self.region_n),
            )
            selected_region = residual_regions.gather(
                1,
                target.view(batch, 1, 1).expand(-1, 1, self.hidden),
            ).squeeze(1)
            coordinator_input = torch.where(
                active.unsqueeze(-1),
                selected_region + uav_feature,
                self.dispatch_base.unsqueeze(0).expand(batch, -1),
            )
            coordinator = self.dispatch_recurrent(
                coordinator_input,
                coordinator,
            )

            selected_one_hot = F.one_hot(
                target,
                num_classes=self.region_n,
            ).to(regions.dtype) * active.unsqueeze(-1)
            selected_count = selected_count + selected_one_hot
            selected_capacity_g = (
                obs["resource"][
                :, self.r + uav_id, 4
                ]
                * 30.0
                * self.macro_duration
            ).unsqueeze(-1)
            planned_capacity_g = planned_capacity_g + (
                selected_one_hot * selected_capacity_g
            )

            dispatches.append(action)
            gate_logprobs.append(gate_distribution.log_prob(gate))
            target_logprobs.append(target_distribution.log_prob(target))
            gate_entropies.append(gate_distribution.entropy())
            target_entropies.append(target_distribution.entropy())

        return UAVPolicyOutput(
            dispatch=torch.stack(dispatches, dim=1),
            gate_logprob=torch.stack(gate_logprobs, dim=1),
            target_logprob=torch.stack(target_logprobs, dim=1),
            gate_entropy=torch.stack(gate_entropies, dim=1),
            target_entropy=torch.stack(target_entropies, dim=1),
            value=global_value,
        )

    def act_vehicles(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> VehiclePolicyOutput:
        (
            vehicle_gate_logits,
            resource_logits,
            _,
            vehicle_value,
        ) = self._logits(obs)
        vehicle_gate_dist = Categorical(logits=vehicle_gate_logits)
        resource_dist = Categorical(logits=resource_logits)
        vehicle_gate = (
            vehicle_gate_logits.argmax(dim=-1)
            if deterministic
            else vehicle_gate_dist.sample()
        )
        resource = (
            resource_logits.argmax(dim=-1)
            if deterministic
            else resource_dist.sample()
        )
        actions = torch.where(vehicle_gate.bool(), resource + 1, torch.zeros_like(resource))
        return VehiclePolicyOutput(
            actions=actions,
            logprob=(
                vehicle_gate_dist.log_prob(vehicle_gate)
                + vehicle_gate.float() * resource_dist.log_prob(resource)
            ),
            entropy=(
                vehicle_gate_dist.entropy()
                + vehicle_gate.float() * resource_dist.entropy()
            ),
            value=vehicle_value,
        )

    def act_uavs(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> UAVPolicyOutput:
        return self._joint_uav_decisions(
            obs,
            deterministic=deterministic,
        )

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> PolicyOutput:
        vehicle = self.act_vehicles(obs, deterministic)
        uav = self.act_uavs(obs, deterministic)
        return PolicyOutput(
            actions=vehicle.actions,
            dispatch=uav.dispatch,
            vehicle_logprob=vehicle.logprob,
            uav_gate_logprob=uav.gate_logprob,
            uav_target_logprob=uav.target_logprob,
            vehicle_entropy=vehicle.entropy,
            uav_gate_entropy=uav.gate_entropy,
            uav_target_entropy=uav.target_entropy,
            vehicle_value=vehicle.value,
            global_value=uav.value,
        )

    def evaluate_vehicle_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            vehicle_gate_logits,
            resource_logits,
            _,
            vehicle_value,
        ) = self._logits(obs)
        vehicle_gate_dist = Categorical(logits=vehicle_gate_logits)
        resource_dist = Categorical(logits=resource_logits)
        vehicle_gate = (actions > 0).long()
        resource = (actions - 1).clamp_min(0)
        return (
            (
                vehicle_gate_dist.log_prob(vehicle_gate)
                + vehicle_gate.float() * resource_dist.log_prob(resource)
            ),
            (
                vehicle_gate_dist.entropy()
                + vehicle_gate.float() * resource_dist.entropy()
            ),
            vehicle_value,
        )

    def evaluate_uav_actions(
        self,
        obs: dict[str, torch.Tensor],
        dispatch: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        output = self._joint_uav_decisions(
            obs,
            dispatch=dispatch,
        )
        return (
            output.gate_logprob,
            output.target_logprob,
            output.gate_entropy,
            output.target_entropy,
            output.value,
        )

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        dispatch: torch.Tensor,
        macro_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del macro_mask
        vehicle_logprob, vehicle_entropy, vehicle_value = (
            self.evaluate_vehicle_actions(obs, actions)
        )
        (
            uav_gate_logprob,
            uav_target_logprob,
            uav_gate_entropy,
            uav_target_entropy,
            global_value,
        ) = self.evaluate_uav_actions(obs, dispatch)
        return (
            vehicle_logprob,
            uav_gate_logprob,
            uav_target_logprob,
            vehicle_entropy,
            uav_gate_entropy,
            uav_target_entropy,
            vehicle_value,
            global_value,
        )
