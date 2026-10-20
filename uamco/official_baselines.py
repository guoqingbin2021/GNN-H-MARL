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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

from .config import (
    Config,
    choose_num_envs,
    sample_training_load_multipliers,
    training_load_multipliers,
)
from .env import VectorOffloadEnv
from .contracts import implementation_fingerprint
from .ppo import resolve_device, sample_preferences, set_seed
from .workflows import workflow_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MAPPO_ROOT = ROOT / "third_party" / "on-policy"
HARL_ROOT = ROOT / "third_party" / "HARL"
MAPPO_COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
HARL_COMMIT = "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
OFFICIAL_METHODS = (
    "official_mappo",
    "official_happo",
    "official_cavdn",
    "official_droo",
)


def baseline_checkpoint_contract(config: Config, source_fingerprint: str | None = None) -> dict[str, str]:
    return {
        "environment_version": "pegasus_micro_motion_v1",
        "workflow_source_fingerprint": (
            workflow_source_fingerprint(config) if source_fingerprint is None else source_fingerprint
        ),
        "implementation_fingerprint": implementation_fingerprint(),
    }


def validate_baseline_checkpoint(config: Config, payload: dict[str, Any]) -> None:
    expected = baseline_checkpoint_contract(config)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Baseline checkpoint {key} mismatch; retrain with the current "
                "Pegasus dataset and physical environment instead of reusing a legacy checkpoint."
            )


@dataclass(frozen=True)
class Box:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Discrete:
    n: int


@dataclass
class OfficialTrainResult:
    checkpoint: Path
    history_path: Path
    summary_path: Path


def _add_import_root(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _source_for(method: str) -> dict[str, str]:
    if method == "official_mappo":
        return {
            "repository": "https://github.com/marlbenchmark/on-policy",
            "commit": MAPPO_COMMIT,
            "algorithm_class": "onpolicy.algorithms.r_mappo.r_mappo.R_MAPPO",
        }
    if method == "official_happo":
        return {
            "repository": "https://github.com/PKU-MARL/HARL",
            "commit": HARL_COMMIT,
            "algorithm_class": "harl.algorithms.actors.happo.HAPPO",
        }
    raise ValueError(f"Unknown official method: {method}")


class OfficialObservationAdapter:
    """Expose vehicles and UAV coordinators as agents for official MARL code."""

    def __init__(self, config: Config, env: VectorOffloadEnv):
        self.cfg = config
        self.env = env
        self.v = env.v
        self.u = env.u
        self.s = env.s
        self.remote_n = env.remote_n
        self.n_agents = self.v + self.u
        self.vehicle_action_n = env.action_n
        self.uav_action_n = env.region_n + 1
        self.shared_action_n = max(self.vehicle_action_n, self.uav_action_n)
        self.vehicle_dim = 13
        self.resource_dim = 8
        self.pair_dim = 8
        self.dag_node_dim = 9
        self.pair_local_dim = max(self.remote_n, self.v) * self.pair_dim

        self.local_dim = (
            self.vehicle_dim
            + self.s * self.dag_node_dim
            + self.s * self.s
            + self.resource_dim
            + self.pair_local_dim
            + self.v * self.vehicle_dim
            + self.remote_n * self.resource_dim
            + 2
            + 2
            + self.n_agents
            + 1
        )
        self.state_dim = (
            self.v * self.vehicle_dim
            + self.v * self.dag_node_dim
            + self.s * self.s
            + self.remote_n * self.resource_dim
            + self.v * self.remote_n * self.pair_dim
            + 2
            + 2
        )
        self.observation_spaces = [Box((self.local_dim,)) for _ in range(self.n_agents)]
        self.share_observation_space = Box((self.state_dim,))
        self.heterogeneous_action_spaces = [
            *[Discrete(self.vehicle_action_n) for _ in range(self.v)],
            *[Discrete(self.uav_action_n) for _ in range(self.u)],
        ]
        self.shared_action_space = Discrete(self.shared_action_n)

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()

    def encode(
        self,
        obs: dict[str, torch.Tensor],
        macro_active: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        local, state = self.encode_torch(obs, macro_active)
        return self._numpy(local), self._numpy(state)

    def encode_torch(
        self,
        obs: dict[str, torch.Tensor],
        macro_active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs["vehicle"].shape[0]
        device = obs["vehicle"].device
        dtype = obs["vehicle"].dtype
        local = torch.zeros(
            (batch, self.n_agents, self.local_dim),
            dtype=dtype,
            device=device,
        )
        cursor = 0

        local[:, : self.v, cursor : cursor + self.vehicle_dim] = obs["vehicle"]
        cursor += self.vehicle_dim
        dag_size = self.s * self.dag_node_dim
        local[:, : self.v, cursor : cursor + dag_size] = obs["dag_nodes"].flatten(2)
        cursor += dag_size
        adjacency_size = self.s * self.s
        local[:, : self.v, cursor : cursor + adjacency_size] = obs[
            "dag_adjacency"
        ].flatten(2)
        cursor += adjacency_size
        for uav_id in range(self.u):
            local[
                :,
                self.v + uav_id,
                cursor : cursor + self.resource_dim,
            ] = obs["resource"][
                :, self.env.r + uav_id
            ]
        cursor += self.resource_dim

        vehicle_pair_size = self.remote_n * self.pair_dim
        local[:, : self.v, cursor : cursor + vehicle_pair_size] = obs[
            "pair"
        ].flatten(2)
        uav_pair_size = self.v * self.pair_dim
        for uav_id in range(self.u):
            local[
                :,
                self.v + uav_id,
                cursor : cursor + uav_pair_size,
            ] = obs["pair"][:, :, self.env.r + uav_id].flatten(1)
        cursor += self.pair_local_dim

        vehicle_targets = obs["vehicle"].flatten(1)
        target_size = self.v * self.vehicle_dim
        local[:, self.v :, cursor : cursor + target_size] = vehicle_targets.unsqueeze(
            1
        )
        cursor += target_size
        resource_view = obs["resource"].flatten(1)
        resource_size = self.remote_n * self.resource_dim
        local[:, :, cursor : cursor + resource_size] = resource_view.unsqueeze(1)
        cursor += resource_size
        local[:, :, cursor : cursor + 2] = obs["preference"].unsqueeze(1)
        cursor += 2
        local[:, : self.v, cursor] = 1.0
        local[:, self.v :, cursor + 1] = 1.0
        cursor += 2
        identity = torch.eye(self.n_agents, dtype=dtype, device=device)
        local[:, :, cursor : cursor + self.n_agents] = identity.unsqueeze(0)
        cursor += self.n_agents
        local[:, :, cursor] = float(macro_active)

        dag_summary = obs["dag_nodes"].mean(dim=2).flatten(1)
        adjacency = obs["dag_adjacency"].float().mean(dim=1).flatten(1)
        progress = torch.full(
            (batch, 1),
            float(self.env.step_index) / max(float(self.env.horizon), 1.0),
            dtype=dtype,
            device=device,
        )
        macro = torch.full(
            (batch, 1),
            float(macro_active),
            dtype=dtype,
            device=device,
        )
        pair_view = obs["pair"].flatten(1)
        state = torch.cat(
            [
                vehicle_targets,
                dag_summary,
                adjacency,
                resource_view,
                pair_view,
                obs["preference"],
                progress,
                macro,
            ],
            dim=-1,
        )
        return local, state

    def shared_available_actions(
        self,
        obs: dict[str, torch.Tensor],
        macro_active: bool,
    ) -> np.ndarray:
        batch = obs["vehicle"].shape[0]
        mask = torch.zeros(
            (batch, self.n_agents, self.shared_action_n),
            dtype=torch.float32,
            device=obs["vehicle"].device,
        )
        mask[:, : self.v, : self.vehicle_action_n] = obs["action_mask"].float()
        if macro_active:
            mask[:, self.v :, : self.uav_action_n] = obs[
                "uav_action_mask"
            ].float()
        else:
            mask[:, self.v :, 0] = 1.0
        return self._numpy(mask)

    def heterogeneous_available_actions(
        self,
        obs: dict[str, torch.Tensor],
        macro_active: bool,
    ) -> list[np.ndarray]:
        vehicle_mask = self._numpy(obs["action_mask"].float())
        result = [vehicle_mask[:, agent_id] for agent_id in range(self.v)]
        uav_mask = self._numpy(obs["uav_action_mask"].float())
        for uav_id in range(self.u):
            if macro_active:
                mask = uav_mask[:, uav_id]
            else:
                mask = np.zeros(
                    (len(vehicle_mask), self.uav_action_n),
                    dtype=np.float32,
                )
                mask[:, 0] = 1.0
            result.append(mask)
        return result

    def step(
        self,
        actions: np.ndarray,
        step: int,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        self.apply_dispatch(actions, step)
        return self.step_vehicles(actions)

    def apply_dispatch(
        self,
        actions: np.ndarray,
        step: int,
    ) -> dict[str, torch.Tensor]:
        action_values = np.asarray(actions).reshape(self.env.b, self.n_agents)
        macro_active = step % int(self.cfg.system.macro_steps) == 0
        if macro_active:
            dispatch = torch.as_tensor(
                action_values[:, self.v :],
                dtype=torch.long,
                device=self.env.device,
            )
            self.env.set_uav_dispatch(dispatch)
        return self.env.observe()

    def step_vehicles(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        action_values = np.asarray(actions).reshape(self.env.b, self.n_agents)
        vehicle_actions = torch.as_tensor(
            action_values[:, : self.v],
            dtype=torch.long,
            device=self.env.device,
        )
        obs, reward, done, _ = self.env.step(vehicle_actions)
        team_reward = self._numpy(reward).reshape(self.env.b, 1, 1)
        rewards = np.repeat(team_reward, self.n_agents, axis=1)
        dones = np.repeat(
            done.detach().cpu().numpy().reshape(self.env.b, 1),
            self.n_agents,
            axis=1,
        )
        return obs, rewards, dones


def _baseline_settings(config: Config) -> dict[str, Any]:
    section = config.official_baselines
    return {
        "hidden_dim": int(section.hidden_dim),
        "recurrent": bool(section.recurrent),
        "recurrent_layers": int(section.recurrent_layers),
        "data_chunk_length": int(section.data_chunk_length),
        "actor_lr": float(section.actor_lr),
        "critic_lr": float(section.critic_lr),
        "update_epochs": int(section.update_epochs),
        "num_mini_batch": int(section.num_mini_batch),
        "entropy_coef": float(section.entropy_coef),
        "value_coef": float(section.value_coef),
        "value_norm": bool(section.value_norm),
        "preference": list(map(float, section.preference)),
    }


def _mappo_args(config: Config, num_envs: int) -> SimpleNamespace:
    settings = _baseline_settings(config)
    return SimpleNamespace(
        algorithm_name="mappo",
        episode_length=int(config.train.rollout_steps),
        n_rollout_threads=num_envs,
        hidden_size=settings["hidden_dim"],
        recurrent_N=settings["recurrent_layers"],
        gamma=float(config.train.gamma),
        gae_lambda=float(config.train.gae_lambda),
        use_gae=True,
        use_popart=False,
        use_valuenorm=settings["value_norm"],
        use_proper_time_limits=False,
        lr=settings["actor_lr"],
        critic_lr=settings["critic_lr"],
        opti_eps=1e-5,
        weight_decay=0.0,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=settings["recurrent"],
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        use_stacked_frames=False,
        layer_N=1,
        data_chunk_length=settings["data_chunk_length"],
        clip_param=float(config.train.clip_ratio),
        ppo_epoch=settings["update_epochs"],
        num_mini_batch=settings["num_mini_batch"],
        value_loss_coef=settings["value_coef"],
        entropy_coef=settings["entropy_coef"],
        max_grad_norm=float(config.train.max_grad_norm),
        huber_delta=10.0,
        use_max_grad_norm=True,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        use_value_active_masks=True,
    )


def _harl_args(config: Config, num_envs: int) -> dict[str, Any]:
    settings = _baseline_settings(config)
    return {
        "episode_length": int(config.train.rollout_steps),
        "n_rollout_threads": num_envs,
        "hidden_sizes": [settings["hidden_dim"], settings["hidden_dim"]],
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": settings["recurrent"],
        "recurrent_n": settings["recurrent_layers"],
        "data_chunk_length": settings["data_chunk_length"],
        "lr": settings["actor_lr"],
        "critic_lr": settings["critic_lr"],
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "std_x_coef": 1.0,
        "std_y_coef": 0.5,
        "ppo_epoch": settings["update_epochs"],
        "critic_epoch": settings["update_epochs"],
        "use_clipped_value_loss": True,
        "clip_param": float(config.train.clip_ratio),
        "actor_num_mini_batch": settings["num_mini_batch"],
        "critic_num_mini_batch": settings["num_mini_batch"],
        "entropy_coef": settings["entropy_coef"],
        "value_loss_coef": settings["value_coef"],
        "use_max_grad_norm": True,
        "max_grad_norm": float(config.train.max_grad_norm),
        "use_gae": True,
        "gamma": float(config.train.gamma),
        "gae_lambda": float(config.train.gae_lambda),
        "use_huber_loss": True,
        "use_policy_active_masks": True,
        "huber_delta": 10.0,
        "action_aggregation": "prod",
        "use_proper_time_limits": False,
    }


class _OfficialTrainerBase:
    method: str

    def __init__(
        self,
        config: Config,
        output_dir: str | Path,
        seed: int,
        device: str,
    ):
        set_seed(seed)
        self.cfg = config
        self.seed = int(seed)
        self.device = resolve_device(device)
        self.num_envs = choose_num_envs(
            config,
            self.device.type,
            (
                torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                if self.device.type == "cuda"
                else None
            ),
        )
        if int(config.train.rollout_steps) != int(config.system.episode_steps):
            raise ValueError(
                "Official baselines require train.rollout_steps == system.episode_steps."
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoint.pt"
        self.summary_path = self.output_dir / "training_summary.json"
        self.history_path.unlink(missing_ok=True)
        self.env = VectorOffloadEnv(config, self.num_envs, self.device, seed)
        self._checkpoint_contract = baseline_checkpoint_contract(
            config, self.env.workflow_bank.metadata["source_fingerprint"],
        )
        self.adapter = OfficialObservationAdapter(config, self.env)
        self.settings = _baseline_settings(config)
        self.load_multipliers = training_load_multipliers(config)
        self.load_generator = torch.Generator(device=self.device)
        self.load_generator.manual_seed(seed + 917_611)
        self.current_load_multiplier = 1.0
        self.current_load_multiplier_std = 0.0

    def _reset(
        self,
        load_multiplier: float | torch.Tensor | None = None,
        preference: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        if load_multiplier is None:
            load_tensor = self.env.load_multiplier
        else:
            load_tensor = torch.as_tensor(
                load_multiplier,
                dtype=torch.float32,
                device=self.device,
            )
            if load_tensor.ndim == 0:
                load_tensor = load_tensor.expand(self.num_envs).clone()
            else:
                load_tensor = load_tensor.reshape(self.num_envs).clone()
        self.current_load_multiplier = float(load_tensor.mean().item())
        self.current_load_multiplier_std = float(
            load_tensor.float().std(unbiased=False).item()
        )
        if preference is None:
            preference = torch.tensor(
                self.settings["preference"],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 2).expand(self.num_envs, -1)
        obs = self.env.reset(preference, load_multiplier=load_tensor)
        local, state = self.adapter.encode(obs, macro_active=True)
        return obs, local, state

    def _record(
        self,
        episode: int,
        start_time: float,
        train_metrics: dict[str, float],
    ) -> dict[str, Any]:
        record = {
            "episode": episode,
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "load_multiplier": self.current_load_multiplier,
            "load_multiplier_std": self.current_load_multiplier_std,
            **self.env.metrics.mean_dict(),
            **train_metrics,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{self.method}] episode {episode:03d}/{self.cfg.train.episodes} "
            f"load={self.current_load_multiplier:.2f}±{self.current_load_multiplier_std:.2f} "
            f"reward={record['reward']:.3f} "
            f"completion={record['completion_rate']:.3f} "
            f"delay={record['effective_delay']:.3f} "
            f"energy={record['energy_per_created_j']:.1f}J "
            f"L/R/U={record['local_ratio']:.2f}/"
            f"{record['rsu_ratio']:.2f}/{record['uav_ratio']:.2f}",
            flush=True,
        )
        return record

    def _finish(self, history: list[dict[str, Any]]) -> OfficialTrainResult:
        summary = {
            "seed": self.seed,
            "method": self.method,
            "device": str(self.device),
            "num_envs": self.num_envs,
            "episodes": int(self.cfg.train.episodes),
            "source": _source_for(self.method),
            "last": history[-1],
            "best_completion_rate": max(item["completion_rate"] for item in history),
            "best_reward": max(item["reward"] for item in history),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return OfficialTrainResult(
            self.checkpoint_path,
            self.history_path,
            self.summary_path,
        )


class OfficialMAPPOTrainer(_OfficialTrainerBase):
    method = "official_mappo"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _add_import_root(MAPPO_ROOT)
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import (
            R_MAPPOPolicy,
        )
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
        from onpolicy.utils.shared_buffer import SharedReplayBuffer

        self.args = _mappo_args(self.cfg, self.num_envs)
        self.policy = R_MAPPOPolicy(
            self.args,
            Box((self.adapter.local_dim,)),
            self.adapter.share_observation_space,
            self.adapter.shared_action_space,
            self.device,
        )
        self.trainer = R_MAPPO(self.args, self.policy, self.device)
        self.buffer = SharedReplayBuffer(
            self.args,
            self.adapter.n_agents,
            Box((self.adapter.local_dim,)),
            self.adapter.share_observation_space,
            self.adapter.shared_action_space,
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def train(self) -> OfficialTrainResult:
        history: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        horizon = int(self.cfg.train.rollout_steps)
        agents = self.adapter.n_agents
        for episode in range(1, int(self.cfg.train.episodes) + 1):
            load_multiplier = sample_training_load_multipliers(
                self.cfg,
                self.num_envs,
                self.device,
                self.load_generator,
            )
            preference = sample_preferences(
                self.cfg,
                self.num_envs,
                self.device,
            )
            obs, local, state = self._reset(load_multiplier, preference)
            shared = np.repeat(state[:, None, :], agents, axis=1)
            available = self.adapter.shared_available_actions(obs, True)
            self.buffer.obs[0] = local
            self.buffer.share_obs[0] = shared
            self.buffer.available_actions[0] = available
            self.buffer.rnn_states[0].fill(0.0)
            self.buffer.rnn_states_critic[0].fill(0.0)
            self.buffer.masks[0].fill(1.0)
            self.buffer.active_masks[0].fill(1.0)
            self.trainer.prep_rollout()

            for step in range(horizon):
                macro = step % int(self.cfg.system.macro_steps) == 0
                (
                    pre_values,
                    pre_actions,
                    pre_action_log_probs,
                    pre_rnn_states,
                    pre_rnn_states_critic,
                ) = self.policy.get_actions(
                    self.buffer.share_obs[step].reshape(-1, self.adapter.state_dim),
                    self.buffer.obs[step].reshape(-1, self.adapter.local_dim),
                    self.buffer.rnn_states[step].reshape(
                        -1,
                        self.args.recurrent_N,
                        self.args.hidden_size,
                    ),
                    self.buffer.rnn_states_critic[step].reshape(
                        -1,
                        self.args.recurrent_N,
                        self.args.hidden_size,
                    ),
                    self.buffer.masks[step].reshape(-1, 1),
                    self.buffer.available_actions[step].reshape(
                        -1,
                        self.adapter.shared_action_n,
                    ),
                )
                values_np = self._to_numpy(pre_values).reshape(
                    self.num_envs,
                    agents,
                    1,
                )
                actions_np = self._to_numpy(pre_actions).reshape(
                    self.num_envs,
                    agents,
                    1,
                )
                log_probs_np = self._to_numpy(pre_action_log_probs).reshape(
                    self.num_envs,
                    agents,
                    1,
                )
                rnn_np = self._to_numpy(pre_rnn_states).reshape(
                    self.num_envs,
                    agents,
                    self.args.recurrent_N,
                    self.args.hidden_size,
                )
                critic_rnn_np = self._to_numpy(pre_rnn_states_critic).reshape(
                    self.num_envs,
                    agents,
                    self.args.recurrent_N,
                    self.args.hidden_size,
                )
                if macro:
                    obs = self.adapter.apply_dispatch(actions_np, step)
                    post_local, _ = self.adapter.encode(obs, macro_active=False)
                    post_available = self.adapter.shared_available_actions(
                        obs,
                        macro_active=False,
                    )
                    self.buffer.obs[step, :, : self.adapter.v] = post_local[
                        :, : self.adapter.v
                    ]
                    self.buffer.available_actions[
                        step, :, : self.adapter.v
                    ] = post_available[:, : self.adapter.v]
                    (
                        _,
                        post_actions,
                        post_action_log_probs,
                        post_rnn_states,
                        _,
                    ) = self.policy.get_actions(
                        self.buffer.share_obs[step].reshape(
                            -1,
                            self.adapter.state_dim,
                        ),
                        self.buffer.obs[step].reshape(
                            -1,
                            self.adapter.local_dim,
                        ),
                        self.buffer.rnn_states[step].reshape(
                            -1,
                            self.args.recurrent_N,
                            self.args.hidden_size,
                        ),
                        self.buffer.rnn_states_critic[step].reshape(
                            -1,
                            self.args.recurrent_N,
                            self.args.hidden_size,
                        ),
                        self.buffer.masks[step].reshape(-1, 1),
                        self.buffer.available_actions[step].reshape(
                            -1,
                            self.adapter.shared_action_n,
                        ),
                    )
                    post_actions_np = self._to_numpy(post_actions).reshape(
                        self.num_envs,
                        agents,
                        1,
                    )
                    post_log_probs_np = self._to_numpy(
                        post_action_log_probs
                    ).reshape(self.num_envs, agents, 1)
                    post_rnn_np = self._to_numpy(post_rnn_states).reshape(
                        self.num_envs,
                        agents,
                        self.args.recurrent_N,
                        self.args.hidden_size,
                    )
                    actions_np[:, : self.adapter.v] = post_actions_np[
                        :, : self.adapter.v
                    ]
                    log_probs_np[:, : self.adapter.v] = post_log_probs_np[
                        :, : self.adapter.v
                    ]
                    rnn_np[:, : self.adapter.v] = post_rnn_np[
                        :, : self.adapter.v
                    ]
                obs, rewards, dones = self.adapter.step_vehicles(actions_np)
                next_macro = (step + 1) % int(self.cfg.system.macro_steps) == 0
                local, state = self.adapter.encode(obs, next_macro)
                shared = np.repeat(state[:, None, :], agents, axis=1)
                available = self.adapter.shared_available_actions(obs, next_macro)
                masks = np.ones((self.num_envs, agents, 1), dtype=np.float32)
                masks[dones] = 0.0
                self.buffer.insert(
                    shared,
                    local,
                    rnn_np,
                    critic_rnn_np,
                    actions_np,
                    log_probs_np,
                    values_np,
                    rewards,
                    masks,
                    active_masks=np.ones_like(masks),
                    available_actions=available,
                )

            next_value = self.policy.get_values(
                self.buffer.share_obs[-1].reshape(-1, self.adapter.state_dim),
                self.buffer.rnn_states_critic[-1].reshape(
                    -1,
                    self.args.recurrent_N,
                    self.args.hidden_size,
                ),
                self.buffer.masks[-1].reshape(-1, 1),
            )
            next_value_np = self._to_numpy(next_value).reshape(
                self.num_envs,
                agents,
                1,
            )
            self.buffer.compute_returns(
                next_value_np,
                self.trainer.value_normalizer,
            )
            self.trainer.prep_training()
            train_info = self.trainer.train(self.buffer)
            train_metrics = {
                "policy_loss": _float(train_info["policy_loss"]),
                "value_loss": _float(train_info["value_loss"]),
                "entropy": _float(train_info["dist_entropy"]),
            }
            history.append(self._record(episode, start_time, train_metrics))
            if (
                episode % int(self.cfg.train.checkpoint_every) == 0
                or episode == int(self.cfg.train.episodes)
            ):
                self.save_checkpoint(episode)
        return self._finish(history)

    def save_checkpoint(self, episode: int) -> None:
        torch.save(
            {
                **self._checkpoint_contract,
                "method": self.method,
                "episode": episode,
                "seed": self.seed,
                "source": _source_for(self.method),
                "config": self.cfg.to_dict(),
                "actor": self.policy.actor.state_dict(),
                "critic": self.policy.critic.state_dict(),
                "actor_optimizer": self.policy.actor_optimizer.state_dict(),
                "critic_optimizer": self.policy.critic_optimizer.state_dict(),
                "value_normalizer": (
                    self.trainer.value_normalizer.state_dict()
                    if self.trainer.value_normalizer is not None
                    else None
                ),
            },
            self.checkpoint_path,
        )


class OfficialHAPPOTrainer(_OfficialTrainerBase):
    method = "official_happo"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _add_import_root(HARL_ROOT)
        from harl.algorithms.actors.happo import HAPPO
        from harl.algorithms.critics.v_critic import VCritic
        from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
        from harl.common.buffers.on_policy_critic_buffer_ep import (
            OnPolicyCriticBufferEP,
        )
        from harl.common.valuenorm import ValueNorm

        self.args = _harl_args(self.cfg, self.num_envs)
        self.actors = [
            HAPPO(
                self.args,
                self.adapter.observation_spaces[agent_id],
                self.adapter.heterogeneous_action_spaces[agent_id],
                self.device,
            )
            for agent_id in range(self.adapter.n_agents)
        ]
        self.critic = VCritic(
            self.args,
            self.adapter.share_observation_space,
            self.device,
        )
        self.actor_buffers = [
            OnPolicyActorBuffer(
                self.args,
                self.adapter.observation_spaces[agent_id],
                self.adapter.heterogeneous_action_spaces[agent_id],
            )
            for agent_id in range(self.adapter.n_agents)
        ]
        self.critic_buffer = OnPolicyCriticBufferEP(
            self.args,
            self.adapter.share_observation_space,
        )
        self.value_normalizer = (
            ValueNorm(1, device=self.device)
            if self.settings["value_norm"]
            else None
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def _update(self) -> dict[str, float]:
        factor = np.ones(
            (
                int(self.cfg.train.rollout_steps),
                self.num_envs,
                1,
            ),
            dtype=np.float32,
        )
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            advantages = (
                self.critic_buffer.returns[:-1]
                - self.critic_buffer.value_preds[:-1]
            )
        actor_infos = []
        order = list(range(self.adapter.n_agents))
        random.shuffle(order)
        for agent_id in order:
            actor = self.actors[agent_id]
            buffer = self.actor_buffers[agent_id]
            buffer.update_factor(factor)
            available = buffer.available_actions[:-1].reshape(
                -1,
                buffer.available_actions.shape[-1],
            )
            old_logprob, _, _ = actor.evaluate_actions(
                buffer.obs[:-1].reshape(-1, self.adapter.local_dim),
                buffer.rnn_states[0:1].reshape(
                    -1,
                    self.args["recurrent_n"],
                    self.settings["hidden_dim"],
                ),
                buffer.actions.reshape(-1, 1),
                buffer.masks[:-1].reshape(-1, 1),
                available,
                buffer.active_masks[:-1].reshape(-1, 1),
            )
            info = actor.train(buffer, advantages.copy(), "EP")
            new_logprob, _, _ = actor.evaluate_actions(
                buffer.obs[:-1].reshape(-1, self.adapter.local_dim),
                buffer.rnn_states[0:1].reshape(
                    -1,
                    self.args["recurrent_n"],
                    self.settings["hidden_dim"],
                ),
                buffer.actions.reshape(-1, 1),
                buffer.masks[:-1].reshape(-1, 1),
                available,
                buffer.active_masks[:-1].reshape(-1, 1),
            )
            ratio = torch.exp(new_logprob - old_logprob).reshape(
                int(self.cfg.train.rollout_steps),
                self.num_envs,
                1,
            )
            factor *= self._to_numpy(ratio)
            actor_infos.append(info)
        critic_info = self.critic.train(
            self.critic_buffer,
            self.value_normalizer,
        )
        return {
            "policy_loss": float(
                np.mean([_float(info["policy_loss"]) for info in actor_infos])
            ),
            "value_loss": _float(critic_info["value_loss"]),
            "entropy": float(
                np.mean([_float(info["dist_entropy"]) for info in actor_infos])
            ),
        }

    def train(self) -> OfficialTrainResult:
        history: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        horizon = int(self.cfg.train.rollout_steps)
        for episode in range(1, int(self.cfg.train.episodes) + 1):
            load_multiplier = sample_training_load_multipliers(
                self.cfg,
                self.num_envs,
                self.device,
                self.load_generator,
            )
            preference = sample_preferences(
                self.cfg,
                self.num_envs,
                self.device,
            )
            obs, local, state = self._reset(load_multiplier, preference)
            available = self.adapter.heterogeneous_available_actions(obs, True)
            for agent_id, buffer in enumerate(self.actor_buffers):
                buffer.obs[0] = local[:, agent_id]
                buffer.available_actions[0] = available[agent_id]
                buffer.rnn_states[0].fill(0.0)
                buffer.masks[0].fill(1.0)
                buffer.active_masks[0].fill(1.0)
            self.critic_buffer.share_obs[0] = state
            self.critic_buffer.rnn_states_critic[0].fill(0.0)
            self.critic_buffer.masks[0].fill(1.0)
            for actor in self.actors:
                actor.prep_rollout()
            self.critic.prep_rollout()

            for step in range(horizon):
                macro = step % int(self.cfg.system.macro_steps) == 0
                action_parts = []
                logprob_parts = []
                rnn_parts = []
                for agent_id, actor in enumerate(self.actors):
                    action, logprob, rnn_state = actor.get_actions(
                        self.actor_buffers[agent_id].obs[step],
                        self.actor_buffers[agent_id].rnn_states[step],
                        self.actor_buffers[agent_id].masks[step],
                        self.actor_buffers[agent_id].available_actions[step],
                    )
                    action_parts.append(self._to_numpy(action))
                    logprob_parts.append(self._to_numpy(logprob))
                    rnn_parts.append(self._to_numpy(rnn_state))
                actions = np.stack(action_parts, axis=1)
                value, critic_rnn = self.critic.get_values(
                    self.critic_buffer.share_obs[step],
                    self.critic_buffer.rnn_states_critic[step],
                    self.critic_buffer.masks[step],
                )
                value_np = self._to_numpy(value)
                critic_rnn_np = self._to_numpy(critic_rnn)
                if macro:
                    obs = self.adapter.apply_dispatch(actions, step)
                    post_local, _ = self.adapter.encode(obs, macro_active=False)
                    post_available = self.adapter.heterogeneous_available_actions(
                        obs,
                        macro_active=False,
                    )
                    for agent_id in range(self.adapter.v):
                        buffer = self.actor_buffers[agent_id]
                        buffer.obs[step] = post_local[:, agent_id]
                        buffer.available_actions[step] = post_available[agent_id]
                        action, logprob, rnn_state = self.actors[
                            agent_id
                        ].get_actions(
                            buffer.obs[step],
                            buffer.rnn_states[step],
                            buffer.masks[step],
                            buffer.available_actions[step],
                        )
                        actions[:, agent_id] = self._to_numpy(action)
                        logprob_parts[agent_id] = self._to_numpy(logprob)
                        rnn_parts[agent_id] = self._to_numpy(rnn_state)
                obs, rewards, dones = self.adapter.step_vehicles(actions)
                next_macro = (step + 1) % int(self.cfg.system.macro_steps) == 0
                local, state = self.adapter.encode(obs, next_macro)
                available = self.adapter.heterogeneous_available_actions(
                    obs,
                    next_macro,
                )
                masks = np.ones(
                    (self.num_envs, self.adapter.n_agents, 1),
                    dtype=np.float32,
                )
                masks[dones] = 0.0
                for agent_id, buffer in enumerate(self.actor_buffers):
                    buffer.insert(
                        local[:, agent_id],
                        rnn_parts[agent_id],
                        actions[:, agent_id],
                        logprob_parts[agent_id],
                        masks[:, agent_id],
                        np.ones_like(masks[:, agent_id]),
                        available[agent_id],
                    )
                self.critic_buffer.insert(
                    state,
                    critic_rnn_np,
                    value_np,
                    rewards[:, 0],
                    masks[:, 0],
                    np.ones_like(masks[:, 0]),
                )

            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            self.critic_buffer.compute_returns(
                self._to_numpy(next_value),
                self.value_normalizer,
            )
            for actor in self.actors:
                actor.prep_training()
            self.critic.prep_training()
            history.append(
                self._record(
                    episode,
                    start_time,
                    self._update(),
                )
            )
            if (
                episode % int(self.cfg.train.checkpoint_every) == 0
                or episode == int(self.cfg.train.episodes)
            ):
                self.save_checkpoint(episode)
        return self._finish(history)

    def save_checkpoint(self, episode: int) -> None:
        torch.save(
            {
                **self._checkpoint_contract,
                "method": self.method,
                "episode": episode,
                "seed": self.seed,
                "source": _source_for(self.method),
                "config": self.cfg.to_dict(),
                "actors": [actor.actor.state_dict() for actor in self.actors],
                "actor_optimizers": [
                    actor.actor_optimizer.state_dict() for actor in self.actors
                ],
                "critic": self.critic.critic.state_dict(),
                "critic_optimizer": self.critic.critic_optimizer.state_dict(),
                "value_normalizer": (
                    self.value_normalizer.state_dict()
                    if self.value_normalizer is not None
                    else None
                ),
            },
            self.checkpoint_path,
        )


def train_official_baseline(
    config: Config,
    output_dir: str | Path,
    seed: int,
    device: str,
    method: str,
) -> OfficialTrainResult:
    if method == "official_mappo":
        return OfficialMAPPOTrainer(config, output_dir, seed, device).train()
    if method == "official_happo":
        return OfficialHAPPOTrainer(config, output_dir, seed, device).train()
    if method in {"official_cavdn", "official_droo"}:
        from .sota_baselines import train_sota_baseline

        return train_sota_baseline(config, output_dir, seed, device, method)
    raise ValueError(f"Unknown official method: {method}")


@torch.no_grad()
def evaluate_official_baseline(
    config: Config,
    checkpoint: str | Path,
    episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool,
    load_multiplier: float = 1.0,
    preference: list[float] | torch.Tensor | None = None,
    obs_filter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    scenario: Any | None = None,
) -> dict[str, float]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    validate_baseline_checkpoint(config, payload)
    method = payload["method"]
    if method in {"official_cavdn", "official_droo"}:
        from .sota_baselines import evaluate_sota_baseline

        return evaluate_sota_baseline(
            config,
            checkpoint,
            episodes,
            seed,
            device,
            deterministic,
            load_multiplier,
            preference,
            obs_filter,
            scenario,
        )
    settings = _baseline_settings(config)
    preference_values = (
        settings["preference"]
        if preference is None
        else preference
    )
    preference = torch.as_tensor(
        preference_values,
        dtype=torch.float32,
        device=device,
    ).view(1, 2).expand(episodes, -1)
    env = VectorOffloadEnv(config, episodes, device, seed, workflow_split="eval")
    adapter = OfficialObservationAdapter(config, env)
    obs = env.reset(preference, load_multiplier=load_multiplier)
    if scenario is not None:
        scenario.after_reset(env)
        obs = env.observe()
    if obs_filter is not None:
        obs = obs_filter(obs)
    local, state = adapter.encode(obs, macro_active=True)
    horizon = int(config.system.episode_steps)

    if method == "official_mappo":
        _add_import_root(MAPPO_ROOT)
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import (
            R_MAPPOPolicy,
        )

        args = _mappo_args(config, episodes)
        policy = R_MAPPOPolicy(
            args,
            Box((adapter.local_dim,)),
            adapter.share_observation_space,
            adapter.shared_action_space,
            device,
        )
        policy.actor.load_state_dict(payload["actor"])
        policy.critic.load_state_dict(payload["critic"])
        policy.actor.eval()
        rnn = np.zeros(
            (
                episodes,
                adapter.n_agents,
                args.recurrent_N,
                args.hidden_size,
            ),
            dtype=np.float32,
        )
        masks = np.ones((episodes, adapter.n_agents, 1), dtype=np.float32)
        for step in range(horizon):
            macro = step % int(config.system.macro_steps) == 0
            available = adapter.shared_available_actions(obs, macro)
            previous_rnn = rnn.copy()
            actions, next_rnn = policy.act(
                local.reshape(-1, adapter.local_dim),
                rnn.reshape(-1, args.recurrent_N, args.hidden_size),
                masks.reshape(-1, 1),
                available.reshape(-1, adapter.shared_action_n),
                deterministic=deterministic,
            )
            actions_np = actions.detach().cpu().numpy().reshape(
                episodes,
                adapter.n_agents,
                1,
            )
            rnn = next_rnn.detach().cpu().numpy().reshape(rnn.shape)
            if macro:
                obs = adapter.apply_dispatch(actions_np, step)
                if obs_filter is not None:
                    obs = obs_filter(obs)
                post_local, _ = adapter.encode(obs, macro_active=False)
                post_available = adapter.shared_available_actions(
                    obs,
                    macro_active=False,
                )
                vehicle_actions, vehicle_rnn = policy.act(
                    post_local[:, : adapter.v].reshape(
                        -1,
                        adapter.local_dim,
                    ),
                    previous_rnn[:, : adapter.v].reshape(
                        -1,
                        args.recurrent_N,
                        args.hidden_size,
                    ),
                    masks[:, : adapter.v].reshape(-1, 1),
                    post_available[:, : adapter.v].reshape(
                        -1,
                        adapter.shared_action_n,
                    ),
                    deterministic=deterministic,
                )
                actions_np[:, : adapter.v] = (
                    vehicle_actions.detach().cpu().numpy().reshape(
                        episodes,
                        adapter.v,
                        1,
                    )
                )
                rnn[:, : adapter.v] = (
                    vehicle_rnn.detach().cpu().numpy().reshape(
                        episodes,
                        adapter.v,
                        args.recurrent_N,
                        args.hidden_size,
                    )
                )
            obs, _, _ = adapter.step_vehicles(actions_np)
            if scenario is not None:
                scenario.after_step(env)
                obs = env.observe()
            if obs_filter is not None:
                obs = obs_filter(obs)
            next_macro = (step + 1) % int(config.system.macro_steps) == 0
            local, state = adapter.encode(obs, next_macro)
    elif method == "official_happo":
        _add_import_root(HARL_ROOT)
        from harl.algorithms.actors.happo import HAPPO

        args = _harl_args(config, episodes)
        actors = [
            HAPPO(
                args,
                adapter.observation_spaces[agent_id],
                adapter.heterogeneous_action_spaces[agent_id],
                device,
            )
            for agent_id in range(adapter.n_agents)
        ]
        for actor, state_dict in zip(actors, payload["actors"]):
            actor.actor.load_state_dict(state_dict)
            actor.prep_rollout()
        rnn = [
            np.zeros(
                (
                    episodes,
                    args["recurrent_n"],
                    settings["hidden_dim"],
                ),
                dtype=np.float32,
            )
            for _ in actors
        ]
        masks = np.ones((episodes, 1), dtype=np.float32)
        for step in range(horizon):
            macro = step % int(config.system.macro_steps) == 0
            available = adapter.heterogeneous_available_actions(obs, macro)
            previous_rnn = [state.copy() for state in rnn]
            parts = []
            for agent_id, actor in enumerate(actors):
                action, next_rnn = actor.act(
                    local[:, agent_id],
                    rnn[agent_id],
                    masks,
                    available[agent_id],
                    deterministic=deterministic,
                )
                parts.append(action.detach().cpu().numpy())
                rnn[agent_id] = next_rnn.detach().cpu().numpy()
            actions_np = np.stack(parts, axis=1)
            if macro:
                obs = adapter.apply_dispatch(actions_np, step)
                if obs_filter is not None:
                    obs = obs_filter(obs)
                post_local, _ = adapter.encode(obs, macro_active=False)
                post_available = adapter.heterogeneous_available_actions(
                    obs,
                    macro_active=False,
                )
                for agent_id in range(adapter.v):
                    action, next_rnn = actors[agent_id].act(
                        post_local[:, agent_id],
                        previous_rnn[agent_id],
                        masks,
                        post_available[agent_id],
                        deterministic=deterministic,
                    )
                    actions_np[:, agent_id] = action.detach().cpu().numpy()
                    rnn[agent_id] = next_rnn.detach().cpu().numpy()
            obs, _, _ = adapter.step_vehicles(actions_np)
            if scenario is not None:
                scenario.after_step(env)
                obs = env.observe()
            if obs_filter is not None:
                obs = obs_filter(obs)
            next_macro = (step + 1) % int(config.system.macro_steps) == 0
            local, state = adapter.encode(obs, next_macro)
    else:
        raise ValueError(f"Unsupported official checkpoint method: {method}")

    return env.summary_metrics()
