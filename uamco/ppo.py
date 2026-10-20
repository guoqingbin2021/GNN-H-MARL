# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import (
    Config,
    choose_num_envs,
    sample_training_load_multipliers,
    training_load_multipliers,
)
from .env import VectorOffloadEnv
from .model import PreferencePolicy
from .variants import macro_interval
from .workflows import workflow_source_fingerprint
from .contracts import implementation_fingerprint, implementation_manifest


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _gpu_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_properties(device).total_memory / (1024**3)


def sample_preferences(config: Config, batch: int, device: torch.device) -> torch.Tensor:
    anchors = torch.tensor(config.reward.anchors, dtype=torch.float32, device=device)
    fixed_anchor = config.train.get("fixed_preference_anchor")
    if fixed_anchor is not None:
        if isinstance(fixed_anchor, bool) or not isinstance(fixed_anchor, int):
            raise ValueError("fixed_preference_anchor must be an integer anchor index")
        if not 0 <= fixed_anchor < len(anchors):
            raise ValueError("fixed_preference_anchor is outside the configured anchors")
        # Objective specialization is fixed before training. The policy still
        # receives its true objective, and no weights or rewards are patched.
        return anchors[fixed_anchor].expand(batch, -1).clone()
    preference = anchors[torch.arange(batch, device=device) % len(anchors)]
    return preference[torch.randperm(batch, device=device)]


def normalize_advantages(
    vehicle_advantage: torch.Tensor,
    global_advantage: torch.Tensor,
    preference: torch.Tensor,
    anchors: list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_vehicle = torch.empty_like(vehicle_advantage)
    normalized_global = torch.empty_like(global_advantage)
    assigned = torch.zeros(
        preference.shape[0],
        dtype=torch.bool,
        device=preference.device,
    )
    for anchor in anchors:
        anchor_tensor = torch.tensor(
            anchor,
            dtype=preference.dtype,
            device=preference.device,
        )
        mask = torch.isclose(
            preference,
            anchor_tensor.unsqueeze(0),
            atol=1e-6,
        ).all(dim=1)
        if not mask.any():
            continue
        vehicle_values = vehicle_advantage[:, mask]
        global_values = global_advantage[:, mask]
        normalized_vehicle[:, mask] = (
            vehicle_values - vehicle_values.mean()
        ) / vehicle_values.std(unbiased=False).clamp_min(1e-6)
        normalized_global[:, mask] = (
            global_values - global_values.mean()
        ) / global_values.std(unbiased=False).clamp_min(1e-6)
        assigned |= mask
    if (~assigned).any():
        vehicle_values = vehicle_advantage[:, ~assigned]
        global_values = global_advantage[:, ~assigned]
        normalized_vehicle[:, ~assigned] = (
            vehicle_values - vehicle_values.mean()
        ) / vehicle_values.std(unbiased=False).clamp_min(1e-6)
        normalized_global[:, ~assigned] = (
            global_values - global_values.mean()
        ) / global_values.std(unbiased=False).clamp_min(1e-6)
    return normalized_vehicle, normalized_global


def normalize_macro_advantages(
    global_advantage: torch.Tensor,
    preference: torch.Tensor,
    anchors: list[list[float]],
    macro_mask: torch.Tensor,
) -> torch.Tensor:
    normalized = torch.zeros_like(global_advantage)
    assigned = torch.zeros(
        preference.shape[0],
        dtype=torch.bool,
        device=preference.device,
    )
    for anchor in anchors:
        anchor_tensor = torch.tensor(
            anchor,
            dtype=preference.dtype,
            device=preference.device,
        )
        env_mask = torch.isclose(
            preference,
            anchor_tensor.unsqueeze(0),
            atol=1e-6,
        ).all(dim=1)
        selected = macro_mask & env_mask.unsqueeze(0)
        if selected.any():
            values = global_advantage[selected]
            normalized[selected] = (
                values - values.mean()
            ) / values.std(unbiased=False).clamp_min(1e-6)
        assigned |= env_mask
    remaining = macro_mask & (~assigned).unsqueeze(0)
    if remaining.any():
        values = global_advantage[remaining]
        normalized[remaining] = (
            values - values.mean()
        ) / values.std(unbiased=False).clamp_min(1e-6)
    return normalized


def compute_macro_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    interval: int,
) -> torch.Tensor:
    """Compute GAE over macro transitions for the slow UAV controller."""
    result = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(rewards[0])
    horizon = rewards.shape[0]
    for step in reversed(range(0, horizon, int(interval))):
        end = min(step + int(interval), horizon)
        duration = end - step
        discounts = torch.pow(
            torch.as_tensor(
                gamma,
                dtype=rewards.dtype,
                device=rewards.device,
            ),
            torch.arange(
                duration,
                dtype=rewards.dtype,
                device=rewards.device,
            ),
        )
        discount_shape = [duration] + [1] * (rewards.ndim - 1)
        macro_reward = (
            rewards[step:end] * discounts.view(*discount_shape)
        ).sum(dim=0)
        next_value = (
            values[end]
            if end < horizon
            else torch.zeros_like(values[0])
        )
        delta = (
            macro_reward
            + (gamma**duration) * next_value
            - values[step]
        )
        next_advantage = (
            delta
            + (gamma**duration) * gae_lambda * next_advantage
        )
        result[step] = next_advantage
    return result


@dataclass
class TrainResult:
    checkpoint: Path
    history_path: Path
    summary_path: Path


class PPOTrainer:
    def __init__(
        self,
        config: Config,
        output_dir: str | Path,
        seed: int,
        device: str = "auto",
        model_kind: str = "uamco",
    ):
        set_seed(seed)
        self.cfg = config
        self.seed = int(seed)
        self.device = resolve_device(device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.num_envs = choose_num_envs(config, self.device.type, _gpu_memory_gb(self.device))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.implementation_manifest = implementation_manifest()
        (self.output_dir / "implementation_manifest.json").write_text(
            json.dumps(self.implementation_manifest, indent=2), encoding="utf-8"
        )
        self.model_kind = model_kind
        self.macro_interval = macro_interval(
            model_kind,
            int(config.system.macro_steps),
        )
        self.load_multipliers = training_load_multipliers(config)
        self.load_generator = torch.Generator(device=self.device)
        self.load_generator.manual_seed(seed + 917_611)
        self.env = VectorOffloadEnv(config, self.num_envs, self.device, seed)
        self.workflow_source_id = self.env.workflow_bank.metadata["source_fingerprint"]
        (self.output_dir / "workflow_manifest.json").write_text(
            json.dumps(self.env.workflow_bank.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if model_kind == "ablation_completion_only":
            self.env.completion_only_dependencies = True
        self.raw_model = PreferencePolicy(config, kind=model_kind).to(self.device)
        self.model = self.raw_model
        if (
            bool(config.model.compile)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        ):
            self.model = torch.compile(self.raw_model, mode="reduce-overhead")
        self.optimizer = torch.optim.Adam(
            self.raw_model.parameters(),
            lr=float(config.train.learning_rate),
            eps=1e-5,
        )
        self.use_amp = bool(config.train.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history_path = self.output_dir / "training.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoint.pt"
        self.history_path.unlink(missing_ok=True)
        self.current_entropy_coef = float(config.train.entropy_coef)
        self.current_uav_entropy_coef = float(config.train.uav_entropy_coef)

    def _flatten_obs(self, observations: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([obs[key] for obs in observations], dim=0).flatten(0, 1)
            for key in observations[0]
        }

    def _update(
        self,
        vehicle_obs: dict[str, torch.Tensor],
        macro_obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        dispatch: torch.Tensor,
        macro_mask: torch.Tensor,
        old_vehicle_logprob: torch.Tensor,
        old_uav_gate_logprob: torch.Tensor,
        old_uav_target_logprob: torch.Tensor,
        vehicle_returns: torch.Tensor,
        global_returns: torch.Tensor,
        vehicle_advantages: torch.Tensor,
        global_advantages: torch.Tensor,
        target_advantages: torch.Tensor,
    ) -> dict[str, float]:
        total = len(actions)
        minibatch = min(int(self.cfg.train.minibatch_size), total)
        losses = []
        for _ in range(int(self.cfg.train.update_epochs)):
            order = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch):
                index = order[start : start + minibatch]
                batch_vehicle_obs = {
                    key: value[index] for key, value in vehicle_obs.items()
                }
                batch_macro_obs = {
                    key: value[index] for key, value in macro_obs.items()
                }
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    (
                        vehicle_logprob,
                        vehicle_entropy,
                        vehicle_value,
                    ) = self.model.evaluate_vehicle_actions(
                        batch_vehicle_obs,
                        actions[index],
                    )
                    (
                        uav_gate_logprob,
                        uav_target_logprob,
                        uav_gate_entropy,
                        uav_target_entropy,
                        global_value,
                    ) = self.model.evaluate_uav_actions(
                        batch_macro_obs,
                        dispatch[index],
                    )
                    vehicle_ratio = (
                        vehicle_logprob - old_vehicle_logprob[index]
                    ).clamp(-10.0, 10.0).exp()
                    vehicle_unclipped = vehicle_ratio * vehicle_advantages[index]
                    vehicle_clipped = vehicle_ratio.clamp(
                        1.0 - float(self.cfg.train.clip_ratio),
                        1.0 + float(self.cfg.train.clip_ratio),
                    ) * vehicle_advantages[index]
                    vehicle_policy_loss = -torch.minimum(
                        vehicle_unclipped,
                        vehicle_clipped,
                    ).mean()

                    macro_weight = macro_mask[index].float().unsqueeze(-1)
                    uav_gate_ratio = (
                        uav_gate_logprob - old_uav_gate_logprob[index]
                    ).clamp(-10.0, 10.0).exp()
                    uav_advantage = global_advantages[index].unsqueeze(-1)
                    uav_gate_unclipped = uav_gate_ratio * uav_advantage
                    uav_gate_clipped = uav_gate_ratio.clamp(
                        1.0 - float(self.cfg.train.clip_ratio),
                        1.0 + float(self.cfg.train.clip_ratio),
                    ) * uav_advantage
                    macro_denominator = (
                        macro_weight.sum() * uav_gate_logprob.shape[-1]
                    ).clamp_min(1.0)
                    uav_gate_loss = -(
                        torch.minimum(
                            uav_gate_unclipped,
                            uav_gate_clipped,
                        ) * macro_weight
                    ).sum() / macro_denominator

                    target_weight = (
                        macro_weight
                        * (dispatch[index] < self.raw_model.region_n).float()
                    )
                    uav_target_ratio = (
                        uav_target_logprob - old_uav_target_logprob[index]
                    ).clamp(-10.0, 10.0).exp()
                    target_unclipped = (
                        uav_target_ratio * target_advantages[index]
                    )
                    target_clipped = uav_target_ratio.clamp(
                        1.0 - float(self.cfg.train.clip_ratio),
                        1.0 + float(self.cfg.train.clip_ratio),
                    ) * target_advantages[index]
                    target_denominator = target_weight.sum().clamp_min(1.0)
                    uav_target_loss = -(
                        torch.minimum(target_unclipped, target_clipped)
                        * target_weight
                    ).sum() / target_denominator
                    policy_loss = (
                        vehicle_policy_loss
                        + uav_gate_loss
                        + uav_target_loss
                    )

                    vehicle_value_loss = torch.nn.functional.smooth_l1_loss(
                        vehicle_value,
                        vehicle_returns[index],
                    )
                    global_value_error = torch.nn.functional.smooth_l1_loss(
                        global_value,
                        global_returns[index],
                        reduction="none",
                    )
                    global_value_loss = (
                        global_value_error * macro_mask[index].float()
                    ).sum() / macro_mask[index].float().sum().clamp_min(1.0)
                    value_loss = vehicle_value_loss + global_value_loss
                    vehicle_entropy_mean = vehicle_entropy.mean()
                    uav_gate_entropy_mean = (
                        uav_gate_entropy * macro_weight
                    ).sum() / macro_denominator
                    uav_target_entropy_mean = (
                        uav_target_entropy * target_weight
                    ).sum() / target_denominator
                    uav_entropy_mean = (
                        uav_gate_entropy_mean + uav_target_entropy_mean
                    )
                    loss = (
                        policy_loss
                        + float(self.cfg.train.value_coef) * value_loss
                        - self.current_entropy_coef * vehicle_entropy_mean
                        - self.current_uav_entropy_coef * uav_entropy_mean
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.max_grad_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                losses.append(
                    (
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float((vehicle_entropy_mean + uav_entropy_mean).detach()),
                    )
                )
        mean = np.asarray(losses, dtype=np.float64).mean(axis=0)
        return {
            "policy_loss": float(mean[0]),
            "value_loss": float(mean[1]),
            "entropy": float(mean[2]),
        }

    def train(self) -> TrainResult:
        self.save_checkpoint(0)
        import shutil
        shutil.copy2(self.checkpoint_path, self.output_dir / "initial_checkpoint.pt")
        history: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        episodes = int(self.cfg.train.episodes)
        horizon = int(self.cfg.train.rollout_steps)
        for episode in range(1, episodes + 1):
            progress = (episode - 1) / max(episodes - 1, 1)
            self.current_entropy_coef = (
                (1.0 - progress) * float(self.cfg.train.entropy_coef)
                + progress * float(self.cfg.train.entropy_final_coef)
            )
            self.current_uav_entropy_coef = (
                (1.0 - progress) * float(self.cfg.train.uav_entropy_coef)
                + progress * float(self.cfg.train.uav_entropy_final_coef)
            )
            if self.raw_model.use_preference:
                preference = sample_preferences(self.cfg, self.num_envs, self.device)
            else:
                preference = torch.full((self.num_envs, 2), 0.5, device=self.device)
            load_multiplier = sample_training_load_multipliers(
                self.cfg,
                self.num_envs,
                self.device,
                self.load_generator,
            )
            obs = self.env.reset(preference, load_multiplier=load_multiplier)
            vehicle_observations = []
            macro_observations = []
            actions = []
            dispatches = []
            macro_masks = []
            vehicle_logprobs = []
            uav_gate_logprobs = []
            uav_target_logprobs = []
            vehicle_values = []
            global_values = []
            vehicle_rewards = []
            global_rewards = []
            with torch.no_grad():
                for step in range(horizon):
                    macro = torch.full(
                        (self.num_envs,),
                        step % self.macro_interval == 0,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    macro_obs = obs
                    if bool(macro[0]):
                        uav_output = self.model.act_uavs(macro_obs)
                        self.env.set_uav_dispatch(uav_output.dispatch)
                        obs = self.env.observe()
                    else:
                        uav_output = None
                    vehicle_output = self.model.act_vehicles(obs)
                    vehicle_observations.append(
                        {key: value.detach() for key, value in obs.items()}
                    )
                    macro_observations.append(
                        {key: value.detach() for key, value in macro_obs.items()}
                    )
                    actions.append(vehicle_output.actions)
                    if uav_output is None:
                        dispatches.append(
                            torch.full(
                                (self.num_envs, self.raw_model.u),
                                self.raw_model.region_n,
                                dtype=torch.long,
                                device=self.device,
                            )
                        )
                        uav_gate_logprobs.append(
                            torch.zeros(
                                (self.num_envs, self.raw_model.u),
                                device=self.device,
                            )
                        )
                        uav_target_logprobs.append(
                            torch.zeros(
                                (self.num_envs, self.raw_model.u),
                                device=self.device,
                            )
                        )
                        global_values.append(
                            torch.zeros(self.num_envs, device=self.device)
                        )
                    else:
                        dispatches.append(uav_output.dispatch)
                        uav_gate_logprobs.append(uav_output.gate_logprob)
                        uav_target_logprobs.append(uav_output.target_logprob)
                        global_values.append(uav_output.value)
                    macro_masks.append(macro)
                    vehicle_logprobs.append(vehicle_output.logprob)
                    vehicle_values.append(vehicle_output.value)
                    obs, reward, _, info = self.env.step(
                        vehicle_output.actions,
                    )
                    vehicle_rewards.append(info["vehicle_reward"])
                    global_rewards.append(reward)

            global_reward_t = torch.stack(global_rewards)
            vehicle_reward_t = torch.stack(vehicle_rewards)
            vehicle_value_t = torch.stack(vehicle_values)
            global_value_t = torch.stack(global_values)
            vehicle_advantage = torch.zeros_like(vehicle_reward_t)
            global_advantage = torch.zeros_like(global_reward_t)
            vehicle_gae = torch.zeros(
                (self.num_envs, int(self.cfg.system.num_vehicles)),
                device=self.device,
            )
            global_gae = torch.zeros(self.num_envs, device=self.device)
            next_vehicle_value = torch.zeros_like(vehicle_gae)
            next_global_value = torch.zeros_like(global_gae)
            for step in range(horizon - 1, -1, -1):
                vehicle_delta = (
                    vehicle_reward_t[step]
                    + float(self.cfg.train.gamma) * next_vehicle_value
                    - vehicle_value_t[step]
                )
                vehicle_gae = (
                    vehicle_delta
                    + float(self.cfg.train.gamma)
                    * float(self.cfg.train.gae_lambda)
                    * vehicle_gae
                )
                vehicle_advantage[step] = vehicle_gae
                next_vehicle_value = vehicle_value_t[step]

                global_delta = (
                    global_reward_t[step]
                    + float(self.cfg.train.gamma) * next_global_value
                    - global_value_t[step]
                )
                global_gae = (
                    global_delta
                    + float(self.cfg.train.gamma)
                    * float(self.cfg.train.gae_lambda)
                    * global_gae
                )
                global_advantage[step] = global_gae
                next_global_value = global_value_t[step]

            vehicle_returns = vehicle_advantage + vehicle_value_t
            normalized_vehicle_advantage, _ = (
                normalize_advantages(
                    vehicle_advantage,
                    global_advantage,
                    preference,
                    [list(map(float, anchor)) for anchor in self.cfg.reward.anchors],
                )
            )
            macro_mask_tensor = torch.stack(macro_masks)
            macro_advantage = compute_macro_advantages(
                global_reward_t,
                global_value_t,
                float(self.cfg.train.gamma),
                float(self.cfg.train.gae_lambda),
                self.macro_interval,
            )
            macro_returns = macro_advantage + global_value_t
            normalized_global_advantage = normalize_macro_advantages(
                macro_advantage,
                preference,
                [list(map(float, anchor)) for anchor in self.cfg.reward.anchors],
                macro_mask_tensor,
            )
            dispatch_tensor = torch.stack(dispatches)
            normalized_target_advantage = (
                normalized_global_advantage.unsqueeze(-1)
                .expand(-1, -1, self.raw_model.u)
            )
            update_metrics = self._update(
                self._flatten_obs(vehicle_observations),
                self._flatten_obs(macro_observations),
                torch.stack(actions).flatten(0, 1),
                torch.stack(dispatches).flatten(0, 1),
                torch.stack(macro_masks).flatten(0, 1),
                torch.stack(vehicle_logprobs).flatten(0, 1).detach(),
                torch.stack(uav_gate_logprobs).flatten(0, 1).detach(),
                torch.stack(uav_target_logprobs).flatten(0, 1).detach(),
                vehicle_returns.flatten(0, 1).detach(),
                macro_returns.flatten().detach(),
                normalized_vehicle_advantage.flatten(0, 1).detach(),
                normalized_global_advantage.flatten().detach(),
                normalized_target_advantage.flatten(0, 1).detach(),
            )
            metrics = self.env.metrics.mean_dict()
            metrics.update(self.env.region_metrics())
            channel_metrics = {}
            if self.raw_model.use_preference:
                for channel, anchor in enumerate(self.cfg.reward.anchors):
                    anchor_tensor = torch.tensor(
                        anchor,
                        dtype=preference.dtype,
                        device=self.device,
                    )
                    channel_mask = torch.isclose(
                        preference,
                        anchor_tensor.unsqueeze(0),
                        atol=1e-6,
                    ).all(dim=1)
                    if channel_mask.any():
                        channel_metrics[f"ch{channel}"] = {
                            **self.env.metrics.mean_dict(channel_mask),
                            **self.env.region_metrics(channel_mask),
                        }
            record = {
                "episode": episode,
                "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                "load_multiplier": float(load_multiplier.mean().item()),
                "load_multiplier_std": float(
                    load_multiplier.float().std(unbiased=False).item()
                ),
                **metrics,
                **update_metrics,
                "entropy_coef": self.current_entropy_coef,
                "uav_entropy_coef": self.current_uav_entropy_coef,
                "channels": channel_metrics,
            }
            history.append(record)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{self.model_kind}] episode {episode:03d}/{episodes} "
                f"load={record['load_multiplier']:.2f}±{record['load_multiplier_std']:.2f} "
                f"reward={record['reward']:.3f} "
                f"completion={record['completion_rate']:.3f} "
                f"delay={record['effective_delay']:.3f} energy={record['energy_per_created_j']:.1f}J "
                f"L/R/U={record['local_ratio']:.2f}/{record['rsu_ratio']:.2f}/{record['uav_ratio']:.2f} "
                f"hotspot={record['uav_selected_region_demand_ratio']:.2f} "
                f"overlap={record['uav_joint_overlap_rate']:.2f}"
            )
            if channel_metrics:
                channel_text = " ".join(
                    f"{name}:D={values['effective_delay']:.2f},"
                    f"E={values['energy_per_created_j']:.1f},"
                    f"R={values['rsu_ratio'] + values['uav_ratio']:.2f}"
                    for name, values in channel_metrics.items()
                )
                print(f"[channels] {channel_text}")
            if episode % int(self.cfg.train.checkpoint_every) == 0 or episode == episodes:
                self.save_checkpoint(episode)

        summary_path = self.output_dir / "training_summary.json"
        from .training_audit import audit_history
        (self.output_dir / "training_audit.json").write_text(
            json.dumps(
                audit_history(
                    history,
                    tuple(
                        self.cfg.evaluation.get(
                            "training_hypervolume_reference",
                            self.cfg.evaluation.hypervolume_reference,
                        )
                    ),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "seed": self.seed,
            "model_kind": self.model_kind,
            "device": str(self.device),
            "num_envs": self.num_envs,
            "episodes": episodes,
            "last": history[-1],
            "best_completion_rate": max(item["completion_rate"] for item in history),
            "best_reward": max(item["reward"] for item in history),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return TrainResult(self.checkpoint_path, self.history_path, summary_path)

    def save_checkpoint(self, episode: int) -> None:
        torch.save(
            {
                "model_format": 6,
                "environment_version": "pegasus_micro_motion_v1",
                "workflow_source_fingerprint": self.workflow_source_id,
                "implementation_fingerprint": self.implementation_manifest["fingerprint"],
                "objective_version": "team_undiscounted_causal_delivery_v4",
                "model": self.raw_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode": int(episode),
                "seed": self.seed,
                "model_kind": self.model_kind,
                "config": self.cfg.to_dict(),
            },
            self.checkpoint_path,
        )


def load_policy(
    config: Config,
    checkpoint: str | Path,
    device: torch.device,
) -> PreferencePolicy:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("Checkpoint numerical implementation differs from current code; use a matching run or retrain")
    if payload.get("objective_version") != "team_undiscounted_causal_delivery_v4":
        raise ValueError("Checkpoint predates the corrected objective/delivery contract; retraining is required")
    if payload.get("model_format") != 6:
        raise ValueError(
            "Checkpoint predates the Pegasus bidirectional topology encoder; retraining is required"
        )
    if payload.get("environment_version") != "pegasus_micro_motion_v1":
        raise ValueError("Checkpoint uses an obsolete synthetic-workload or UAV-motion environment")
    if payload.get("workflow_source_fingerprint") != workflow_source_fingerprint(config):
        raise ValueError("Checkpoint source datasets differ from the active Pegasus source datasets")
    model = PreferencePolicy(config, kind=payload.get("model_kind", "uamco")).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model
