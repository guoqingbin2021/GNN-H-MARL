# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    Config,
    choose_num_envs,
    sample_training_load_multipliers,
    training_load_multipliers,
)
from .env import VectorOffloadEnv
from .official_baselines import (
    OfficialObservationAdapter,
    OfficialTrainResult,
    baseline_checkpoint_contract,
    validate_baseline_checkpoint,
)
from .physics import rotary_wing_level_power
from .physics import compute_energy
from .ppo import resolve_device, sample_preferences, set_seed


ROOT = Path(__file__).resolve().parents[1]
CAVDN_ROOT = ROOT / "third_party" / "SOTA" / "CAVDN" / "CAVDN"
DROO_ROOT = ROOT / "third_party" / "SOTA" / "DROO" / "DROO"
CAVDN_COMMIT = "17126934c71e64b0669ad662cdbd6b1e569a8864"
DROO_COMMIT = "563d6c7723463c3e6b77b470e3200268c5fad4f3"
SOTA_METHODS = ("official_cavdn", "official_droo")


def _add_import_root(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def source_for(method: str) -> dict[str, Any]:
    if method == "official_cavdn":
        return {
            "repository": "https://github.com/chenbq/CAVDN",
            "commit": CAVDN_COMMIT,
            "upstream_files": [
                "network/base_net.py:RNN_FC",
                "network/coord_net.py:CoordNet,Q_out",
                "network/qmix_net.py:QMixNet",
            ],
            "adapter": "GPU-vectorized recurrent replay and environment bridge",
        }
    if method == "official_droo":
        return {
            "repository": "https://github.com/revenol/DROO",
            "commit": DROO_COMMIT,
            "upstream_files": [
                "memoryPyTorch.py:MemoryDNN",
                "memoryPyTorch.py:MemoryDNN._build_net",
                "memoryPyTorch.py:MemoryDNN.knm (vectorized-equivalent OP decoder)",
            ],
            "adapter": "RSU/UAV candidate evaluator replacing WPMEC-only bisection",
        }
    raise ValueError(f"Unknown SOTA method: {method}")


def _balanced_preference(config: Config, batch: int, device: torch.device) -> torch.Tensor:
    values = config.official_baselines.preference
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, 2).expand(batch, -1)


class _EpisodeReplay:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.chunks: deque[dict[str, torch.Tensor]] = deque()
        self.size = 0

    def add(self, rollout: dict[str, torch.Tensor]) -> None:
        count = rollout["actions"].shape[0]
        keep = min(count, self.capacity)
        index = torch.randperm(count, device=rollout["actions"].device)[:keep]
        chunk = {
            key: value[index].detach().clone()
            for key, value in rollout.items()
        }
        self.chunks.append(chunk)
        self.size += keep
        while self.size > self.capacity and len(self.chunks) > 1:
            removed = self.chunks.popleft()
            self.size -= removed["actions"].shape[0]

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        merged = {
            key: torch.cat([chunk[key] for chunk in self.chunks], dim=0)
            for key in self.chunks[0]
        }
        count = merged["actions"].shape[0]
        index = torch.randint(
            count,
            (min(int(batch_size), count),),
            device=merged["actions"].device,
        )
        return {key: value[index] for key, value in merged.items()}


class _CAVDNPolicy:
    def __init__(
        self,
        adapter: OfficialObservationAdapter,
        config: Config,
        device: torch.device,
    ):
        _add_import_root(CAVDN_ROOT)
        from network.base_net import RNN_FC
        from network.coord_net import CoordNet, Q_out
        from network.qmix_net import QMixNet

        section = config.cavdn
        self.adapter = adapter
        self.device = device
        self.n_agents = adapter.n_agents
        self.n_actions = adapter.shared_action_n
        self.hidden_dim = int(section.rnn_hidden_dim)
        self.input_dim = adapter.local_dim + self.n_actions + self.n_agents
        self.args = SimpleNamespace(
            rnn_hidden_dim=self.hidden_dim,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            state_shape=adapter.state_dim,
            qmix_hidden_dim=int(section.qmix_hidden_dim),
            hyper_hidden_dim=int(section.hyper_hidden_dim),
            two_hyper_layers=bool(section.two_hyper_layers),
        )
        self.eval_rnn = RNN_FC(self.input_dim, self.args).to(device)
        self.target_rnn = RNN_FC(self.input_dim, self.args).to(device)
        self.eval_coord = CoordNet(self.args).to(device)
        self.target_coord = CoordNet(self.args).to(device)
        self.eval_q_out = Q_out(self.args).to(device)
        self.target_q_out = Q_out(self.args).to(device)
        self.eval_mixer = QMixNet(self.args).to(device)
        self.target_mixer = QMixNet(self.args).to(device)
        self._copy_targets()
        self.agent_ids = torch.eye(
            self.n_agents,
            dtype=torch.float32,
            device=device,
        )

    def _copy_targets(self) -> None:
        for target, source in self.target_pairs():
            target.load_state_dict(source.state_dict())

    def target_pairs(self) -> list[tuple[torch.nn.Module, torch.nn.Module]]:
        return [
            (self.target_rnn, self.eval_rnn),
            (self.target_coord, self.eval_coord),
            (self.target_q_out, self.eval_q_out),
            (self.target_mixer, self.eval_mixer),
        ]

    def parameters(self) -> list[torch.nn.Parameter]:
        modules = [
            self.eval_rnn,
            self.eval_coord,
            self.eval_q_out,
            self.eval_mixer,
        ]
        return [parameter for module in modules for parameter in module.parameters()]

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "rnn": self.eval_rnn.state_dict(),
            "coord": self.eval_coord.state_dict(),
            "q_out": self.eval_q_out.state_dict(),
            "mixer": self.eval_mixer.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, dict[str, torch.Tensor]]) -> None:
        self.eval_rnn.load_state_dict(payload["rnn"])
        self.eval_coord.load_state_dict(payload["coord"])
        self.eval_q_out.load_state_dict(payload["q_out"])
        self.eval_mixer.load_state_dict(payload["mixer"])
        self._copy_targets()

    def initial_hidden(self, batch: int) -> torch.Tensor:
        return torch.zeros(
            (batch, self.n_agents, self.hidden_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def inputs(
        self,
        local: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        batch = local.shape[0]
        identities = self.agent_ids.unsqueeze(0).expand(batch, -1, -1)
        return torch.cat([local, last_action, identities], dim=-1).flatten(0, 1)

    def q_values(
        self,
        local: torch.Tensor,
        last_action: torch.Tensor,
        hidden: torch.Tensor,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rnn = self.target_rnn if target else self.eval_rnn
        coord = self.target_coord if target else self.eval_coord
        q_out = self.target_q_out if target else self.eval_q_out
        batch = local.shape[0]
        _, next_hidden, _ = rnn(
            self.inputs(local, last_action),
            hidden.flatten(0, 1),
        )
        next_hidden = next_hidden.view(batch, self.n_agents, self.hidden_dim)
        communicated = coord(next_hidden)
        q_values = q_out(torch.cat([next_hidden, communicated], dim=-1))
        return q_values, next_hidden


class OfficialCAVDNTrainer:
    method = "official_cavdn"

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
        memory_gb = (
            torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            if self.device.type == "cuda"
            else None
        )
        self.num_envs = choose_num_envs(config, self.device.type, memory_gb)
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
        self.policy = _CAVDNPolicy(self.adapter, config, self.device)
        self.section = config.cavdn
        self.load_multipliers = training_load_multipliers(config)
        self.load_generator = torch.Generator(device=self.device)
        self.load_generator.manual_seed(seed + 917_611)
        self.current_load_multiplier = 1.0
        self.current_load_multiplier_std = 0.0
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.section.learning_rate),
        )
        self.replay = _EpisodeReplay(
            max(
                int(self.section.replay_episodes),
                self.num_envs,
            )
        )
        self.updates_per_rollout = max(
            1,
            math.ceil(
                self.num_envs
                / max(int(self.section.batch_episodes), 1)
            ),
        )

    def _epsilon(self, episode: int) -> float:
        start = float(self.section.epsilon_start)
        end = float(self.section.epsilon_end)
        duration = max(int(self.cfg.train.episodes * float(self.section.epsilon_fraction)), 1)
        fraction = min(max((episode - 1) / duration, 0.0), 1.0)
        return start + fraction * (end - start)

    def _available(self, obs: dict[str, torch.Tensor], macro: bool) -> torch.Tensor:
        mask = torch.zeros(
            (
                self.num_envs,
                self.adapter.n_agents,
                self.adapter.shared_action_n,
            ),
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, : self.env.v, : self.env.action_n] = obs["action_mask"]
        if macro:
            mask[:, self.env.v :, : self.env.region_n + 1] = obs[
                "uav_action_mask"
            ]
        else:
            mask[:, self.env.v :, 0] = True
        return mask

    def _encode(self, obs: dict[str, torch.Tensor], macro: bool) -> tuple[torch.Tensor, torch.Tensor]:
        local, state = self.adapter.encode_torch(obs, macro)
        return local, state

    def _collect(
        self,
        epsilon: float,
        load_multiplier: float | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        horizon = int(self.cfg.train.rollout_steps)
        preference = sample_preferences(
            self.cfg,
            self.num_envs,
            self.device,
        )
        obs = self.env.reset(
            preference,
            load_multiplier=load_tensor,
        )
        local, state = self._encode(obs, True)
        hidden = self.policy.initial_hidden(self.num_envs)
        last_action = torch.zeros(
            (
                self.num_envs,
                self.adapter.n_agents,
                self.adapter.shared_action_n,
            ),
            device=self.device,
        )
        local_steps = [local]
        state_steps = [state]
        available_steps = [self._available(obs, True)]
        actions_steps = []
        reward_steps = []
        done_steps = []

        for step in range(horizon):
            macro = step % int(self.cfg.system.macro_steps) == 0
            hidden_before = hidden
            with torch.no_grad():
                pre_q_values, pre_hidden = self.policy.q_values(
                    local,
                    last_action,
                    hidden_before,
                )
                available = available_steps[-1]
                masked = pre_q_values.masked_fill(~available, -1e9)
                greedy = masked.argmax(dim=-1)
                random_action = torch.multinomial(
                    available.float().flatten(0, 1),
                    1,
                ).view(self.num_envs, self.adapter.n_agents)
                explore = torch.rand_like(greedy.float()) < epsilon
                pre_actions = torch.where(explore, random_action, greedy)

            if macro:
                self.env.set_uav_dispatch(pre_actions[:, self.env.v :])
                post_obs = self.env.observe()
                post_local, _ = self._encode(post_obs, False)
                post_available = self._available(post_obs, False)
                hybrid_local = local.clone()
                hybrid_local[:, : self.env.v] = post_local[:, : self.env.v]
                hybrid_available = available.clone()
                hybrid_available[:, : self.env.v] = post_available[
                    :, : self.env.v
                ]
                with torch.no_grad():
                    hybrid_q_values, hybrid_hidden = self.policy.q_values(
                        hybrid_local,
                        last_action,
                        hidden_before,
                    )
                    hybrid_masked = hybrid_q_values.masked_fill(
                        ~hybrid_available,
                        -1e9,
                    )
                    hybrid_greedy = hybrid_masked.argmax(dim=-1)
                    hybrid_random = torch.multinomial(
                        hybrid_available.float().flatten(0, 1),
                        1,
                    ).view(self.num_envs, self.adapter.n_agents)
                    hybrid_explore = (
                        torch.rand_like(hybrid_greedy.float()) < epsilon
                    )
                    actions = torch.where(
                        hybrid_explore,
                        hybrid_random,
                        hybrid_greedy,
                    )
                actions[:, self.env.v :] = pre_actions[:, self.env.v :]
                hidden = hybrid_hidden
                hidden[:, self.env.v :] = pre_hidden[:, self.env.v :]
                local_steps[-1] = hybrid_local
                available_steps[-1] = hybrid_available
            else:
                actions = pre_actions
                hidden = pre_hidden

            next_obs, reward, done, _ = self.env.step(
                actions[:, : self.env.v],
            )
            next_macro = (step + 1) % int(self.cfg.system.macro_steps) == 0
            next_local, next_state = self._encode(next_obs, next_macro)
            next_available = self._available(next_obs, next_macro)

            actions_steps.append(actions)
            reward_steps.append(reward.unsqueeze(-1))
            done_steps.append(done.float().unsqueeze(-1))
            local_steps.append(next_local)
            state_steps.append(next_state)
            available_steps.append(next_available)
            last_action = F.one_hot(
                actions,
                self.adapter.shared_action_n,
            ).float()
            obs, local, state = next_obs, next_local, next_state

        return {
            "local": torch.stack(local_steps, dim=1),
            "state": torch.stack(state_steps, dim=1),
            "available": torch.stack(available_steps, dim=1),
            "actions": torch.stack(actions_steps, dim=1),
            "reward": torch.stack(reward_steps, dim=1),
            "done": torch.stack(done_steps, dim=1),
        }

    def _update(self) -> float:
        batch = self.replay.sample(int(self.section.batch_episodes))
        count, horizon = batch["actions"].shape[:2]
        hidden = self.policy.initial_hidden(count)
        target_hidden = self.policy.initial_hidden(count)
        last_action = torch.zeros(
            (count, self.adapter.n_agents, self.adapter.shared_action_n),
            device=self.device,
        )
        eval_values = []
        target_values = []
        for step in range(horizon):
            q_eval, hidden = self.policy.q_values(
                batch["local"][:, step],
                last_action,
                hidden,
            )
            current_action = F.one_hot(
                batch["actions"][:, step],
                self.adapter.shared_action_n,
            ).float()
            with torch.no_grad():
                q_target, target_hidden = self.policy.q_values(
                    batch["local"][:, step + 1],
                    current_action,
                    target_hidden,
                    target=True,
                )
            eval_values.append(q_eval)
            target_values.append(q_target)
            last_action = current_action

        q_eval = torch.stack(eval_values, dim=1)
        q_target = torch.stack(target_values, dim=1)
        selected = q_eval.gather(
            -1,
            batch["actions"].unsqueeze(-1),
        ).squeeze(-1)
        q_target = q_target.masked_fill(
            ~batch["available"][:, 1:],
            -1e9,
        ).max(dim=-1).values
        total_eval = self.policy.eval_mixer(selected, batch["state"][:, :-1])
        with torch.no_grad():
            total_target = self.policy.target_mixer(
                q_target,
                batch["state"][:, 1:],
            )
            target = batch["reward"] + float(self.section.gamma) * total_target * (
                1.0 - batch["done"]
            )
        loss = F.mse_loss(total_eval, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            float(self.section.grad_norm_clip),
        )
        self.optimizer.step()

        return float(loss.detach().cpu())

    def _update_targets(self, episode: int) -> None:
        if episode % int(self.section.target_update_interval) == 0:
            tau = float(self.section.target_tau)
            with torch.no_grad():
                for target_module, eval_module in self.policy.target_pairs():
                    for target_parameter, eval_parameter in zip(
                        target_module.parameters(),
                        eval_module.parameters(),
                    ):
                        target_parameter.mul_(1.0 - tau).add_(eval_parameter, alpha=tau)

    def _save(self, episode: int) -> None:
        torch.save(
            {
                **self._checkpoint_contract,
                "method": self.method,
                "episode": episode,
                "seed": self.seed,
                "source": source_for(self.method),
                "config": self.cfg.to_dict(),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.checkpoint_path,
        )

    def train(self) -> OfficialTrainResult:
        history = []
        start = time.perf_counter()
        episodes = int(self.cfg.train.episodes)
        for episode in range(1, episodes + 1):
            epsilon = self._epsilon(episode)
            load_multiplier = sample_training_load_multipliers(
                self.cfg,
                self.num_envs,
                self.device,
                self.load_generator,
            )
            rollout = self._collect(epsilon, load_multiplier)
            self.replay.add(rollout)
            losses = [
                self._update()
                for _ in range(self.updates_per_rollout)
            ]
            self._update_targets(episode)
            loss = float(np.mean(losses))
            record = {
                "episode": episode,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "load_multiplier": self.current_load_multiplier,
                "load_multiplier_std": self.current_load_multiplier_std,
                **self.env.metrics.mean_dict(),
                "td_loss": loss,
                "epsilon": epsilon,
                "gradient_updates": self.updates_per_rollout,
            }
            history.append(record)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[official_cavdn] episode {episode:03d}/{episodes} "
                f"load={self.current_load_multiplier:.2f}±{self.current_load_multiplier_std:.2f} "
                f"reward={record['reward']:.3f} completion={record['completion_rate']:.3f} "
                f"L/R/U={record['local_ratio']:.2f}/{record['rsu_ratio']:.2f}/"
                f"{record['uav_ratio']:.2f} loss={loss:.4f}",
                flush=True,
            )
            if episode % int(self.cfg.train.checkpoint_every) == 0 or episode == episodes:
                self._save(episode)

        summary = {
            "seed": self.seed,
            "method": self.method,
            "device": str(self.device),
            "num_envs": self.num_envs,
            "replay_capacity_episodes": self.replay.capacity,
            "gradient_updates_per_rollout": self.updates_per_rollout,
            "episodes": episodes,
            "source": source_for(self.method),
            "last": history[-1],
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


class _DROOPolicy:
    def __init__(self, config: Config, env: VectorOffloadEnv, device: torch.device):
        _add_import_root(DROO_ROOT)
        from memoryPyTorch import MemoryDNN

        section = config.droo
        self.cfg = config
        self.env = env
        self.device = device
        self.memory_dnn = MemoryDNN(
            net=[
                env.v,
                int(section.hidden_dim_1),
                int(section.hidden_dim_2),
                env.v,
            ],
            learning_rate=float(section.learning_rate),
            training_interval=int(section.training_interval),
            batch_size=int(section.batch_size),
            memory_size=int(section.memory_size),
            output_graph=False,
        )
        self.model = self.memory_dnn.model.to(device)
        self.section = section
        self.seen = 0
        self.frame = 0
        self.current_k = min(int(section.initial_k), env.v)
        self.winner_history: deque[int] = deque(maxlen=int(section.adaptive_interval))
        self.last_loss = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "current_k": self.current_k,
            "frame": self.frame,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.model.load_state_dict(payload["model"])
        self.current_k = int(payload.get("current_k", self.current_k))
        self.frame = int(payload.get("frame", 0))

    def _scores(
        self,
        preference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        costs = self.env.estimate_action_costs()
        scores = (
            preference[:, None, None, 0] * costs[..., 0]
            / float(self.cfg.reward.delay_reference)
            + preference[:, None, None, 1] * costs[..., 1]
            / float(self.cfg.reward.energy_reference_j)
        )
        _, selected, has_ready, _ = self.env._ready_state()
        cycles = self.env._head_tensor(self.env.task_cycles).gather(
            2,
            selected.unsqueeze(-1),
        ).squeeze(-1)
        data_mb = self.env._upload_payload_mb(selected)
        distance = torch.linalg.norm(
            self.env.vehicle_xy.unsqueeze(2) - self.env.uav_xy.unsqueeze(1),
            dim=-1,
        )
        travel = distance / float(self.cfg.uav.max_speed_mps)
        vertical_distance = max(
            float(self.cfg.radio.uav_height_m)
            - float(self.cfg.radio.vehicle_height_m),
            1.0,
        )
        path_loss_db = (
            32.4
            + 20.0 * math.log10(float(self.cfg.radio.carrier_ghz))
            + 20.0 * math.log10(vertical_distance)
        )
        tx_dbm = 10.0 * math.log10(float(self.cfg.radio.tx_power_w) * 1000.0)
        noise_dbm = (
            -174.0
            + 10.0 * math.log10(float(self.cfg.radio.bandwidth_mhz) * 1e6)
            + float(self.cfg.radio.noise_figure_db)
        )
        snr = 10.0 ** ((tx_dbm - path_loss_db - noise_dbm) / 10.0)
        target_rate = (
            float(self.cfg.radio.bandwidth_mhz)
            * 1e6
            * math.log2(1.0 + snr)
        )
        upload = data_mb.unsqueeze(-1) * 8e6 / target_rate
        wait = (
            self.env.node_available[:, self.env.v + self.env.r :].unsqueeze(1)
            - self.env.time.view(self.env.b, 1, 1)
        ).clamp_min(0.0)
        compute = cycles.unsqueeze(-1) / self.env.cpu_hz[
            self.env.v + self.env.r :
        ].view(1, 1, -1)
        planned_delay = travel + upload + wait + compute
        planned_task_energy = (
            float(self.cfg.radio.tx_power_w) * upload
            + compute_energy(
                cycles.unsqueeze(-1),
                float(self.cfg.compute.uav_energy_per_gcycle_j),
            )
        )
        macro_duration = (
            float(self.cfg.system.macro_steps)
            * float(self.cfg.system.dt_seconds)
        )
        planned_speed = (distance / macro_duration).clamp_max(
            float(self.cfg.uav.max_speed_mps)
        )
        planned_flight_energy = (
            float(self.cfg.uav.base_power_w)
            + rotary_wing_level_power(
                planned_speed,
                profile_power_w=float(self.cfg.uav.profile_power_w),
                induced_power_w=float(self.cfg.uav.induced_power_w),
                tip_speed_mps=float(self.cfg.uav.tip_speed_mps),
                mean_rotor_velocity_mps=float(
                    self.cfg.uav.mean_rotor_velocity_mps
                ),
                fuselage_drag_ratio=float(self.cfg.uav.fuselage_drag_ratio),
                air_density_kg_m3=float(self.cfg.uav.air_density_kg_m3),
                rotor_solidity=float(self.cfg.uav.rotor_solidity),
                rotor_area_m2=float(self.cfg.uav.rotor_area_m2),
            )
        ) * macro_duration
        planned_uav_score = (
            preference[:, None, None, 0]
            * planned_delay
            / float(self.cfg.reward.delay_reference)
            + preference[:, None, None, 1]
            * (
                planned_task_energy
                / float(self.cfg.reward.energy_reference_j)
                + planned_flight_energy
                / float(self.cfg.reward.energy_reference_j)
            )
        )
        planned_uav_score = torch.where(
            has_ready.unsqueeze(-1),
            planned_uav_score,
            torch.full_like(planned_uav_score, 1e6),
        )
        scores = scores.clone()
        scores[..., 1 + self.env.r :] = torch.minimum(
            scores[..., 1 + self.env.r :],
            planned_uav_score,
        )
        local = scores[..., 0]
        remote_score, remote_index = scores[..., 1:].min(dim=-1)
        remote_action = remote_index + 1
        remote_valid = remote_score < 1e5
        remote_action = torch.where(
            remote_valid,
            remote_action,
            torch.zeros_like(remote_action),
        )
        scale = torch.maximum(local.abs(), torch.ones_like(local))
        feature = torch.tanh((local - remote_score) / scale)
        feature = torch.where(remote_valid, feature, torch.full_like(feature, -1.0))
        return feature, local, torch.stack([remote_score, remote_action.float()], dim=-1)

    @staticmethod
    def _op_candidates(prediction: torch.Tensor, k: int) -> torch.Tensor:
        candidates = [(prediction > 0.5)]
        if k > 1:
            uncertain = (prediction - 0.5).abs().argsort(dim=-1)[:, : k - 1]
            thresholds = prediction.gather(1, uncertain)
            for index in range(k - 1):
                threshold = thresholds[:, index : index + 1]
                strict = threshold > 0.5
                candidate = torch.where(
                    strict,
                    prediction > threshold,
                    prediction >= threshold,
                )
                candidates.append(candidate)
        return torch.stack(candidates, dim=1)

    def _select(
        self,
        feature: torch.Tensor,
        local: torch.Tensor,
        remote: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(feature)
        candidates = self._op_candidates(prediction, self.current_k)
        remote_score = remote[..., 0]
        remote_action = remote[..., 1].long()
        valid_remote = remote_action > 0
        candidate_remote = candidates & valid_remote.unsqueeze(1)
        candidate_cost = torch.where(
            candidate_remote,
            remote_score.unsqueeze(1),
            local.unsqueeze(1),
        ).sum(dim=-1)
        winner = candidate_cost.argmin(dim=1)
        best_gate = candidates[
            torch.arange(feature.shape[0], device=self.device),
            winner,
        ]
        planned_actions = torch.where(best_gate & valid_remote, remote_action, 0)
        executable = self.env.action_mask().gather(
            -1,
            planned_actions.unsqueeze(-1),
        ).squeeze(-1)
        actions = torch.where(executable, planned_actions, 0)
        return actions, planned_actions, best_gate.float(), winner

    def _remember(self, feature: torch.Tensor, label: torch.Tensor) -> None:
        values = torch.cat([feature, label], dim=-1).detach().cpu().numpy()
        size = int(self.memory_dnn.memory_size)
        indices = (np.arange(len(values)) + self.seen) % size
        self.memory_dnn.memory[indices] = values
        self.seen += len(values)
        self.memory_dnn.memory_counter = self.seen + 1

    def _learn(self) -> None:
        available = min(self.seen, int(self.memory_dnn.memory_size))
        if available <= 0:
            return
        sample = np.random.choice(
            available,
            size=int(self.memory_dnn.batch_size),
            replace=available < int(self.memory_dnn.batch_size),
        )
        batch = torch.as_tensor(
            self.memory_dnn.memory[sample],
            dtype=torch.float32,
            device=self.device,
        )
        inputs = batch[:, : self.env.v]
        labels = batch[:, self.env.v :]
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.memory_dnn.lr),
            betas=(0.09, 0.999),
            weight_decay=1e-4,
        )
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy(self.model(inputs), labels)
        loss.backward()
        optimizer.step()
        self.last_loss = float(loss.detach().cpu())

    def act(
        self,
        preference: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature, local, remote = self._scores(preference)
        actions, planned_actions, label, winner = self._select(
            feature,
            local,
            remote,
        )
        self.planned_actions = planned_actions
        if training:
            self._remember(feature, label)
            self.winner_history.extend(winner.detach().cpu().tolist())
            self.frame += 1
            if self.frame % int(self.section.training_interval) == 0:
                self._learn()
            if self.frame % int(self.section.adaptive_interval) == 0:
                maximum = max(self.winner_history, default=0) + 2
                self.current_k = min(maximum, self.env.v)
        return actions, feature

    def dispatch(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        urgency = (
            obs["vehicle"][..., 4]
            + obs["vehicle"][..., 5]
            + obs["vehicle"][..., 7]
        )
        dispatch = torch.full(
            (self.env.b, self.env.u),
            self.env.region_n,
            dtype=torch.long,
            device=self.device,
        )
        for uav_id in range(self.env.u):
            requested = self.planned_actions == 1 + self.env.r + uav_id
            score = urgency.masked_fill(~requested, -1e9)
            best_score, target = score.max(dim=1)
            target_region = self.env.vehicle_region_index().gather(
                1,
                target.unsqueeze(1),
            ).squeeze(1)
            dispatch[:, uav_id] = torch.where(
                best_score > -1e8,
                target_region,
                dispatch[:, uav_id],
            )
        return dispatch


class OfficialDROOTrainer:
    method = "official_droo"

    def __init__(
        self,
        config: Config,
        output_dir: str | Path,
        seed: int,
        device: str,
    ):
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.cfg = config
        self.seed = int(seed)
        self.device = resolve_device(device)
        memory_gb = (
            torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            if self.device.type == "cuda"
            else None
        )
        self.num_envs = choose_num_envs(config, self.device.type, memory_gb)
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
        self.policy = _DROOPolicy(config, self.env, self.device)
        self.load_multipliers = training_load_multipliers(config)
        self.load_generator = torch.Generator(device=self.device)
        self.load_generator.manual_seed(seed + 917_611)
        self.current_load_multiplier = 1.0
        self.current_load_multiplier_std = 0.0

    def _save(self, episode: int) -> None:
        torch.save(
            {
                **self._checkpoint_contract,
                "method": self.method,
                "episode": episode,
                "seed": self.seed,
                "source": source_for(self.method),
                "config": self.cfg.to_dict(),
                "policy": self.policy.state_dict(),
            },
            self.checkpoint_path,
        )

    def train(self) -> OfficialTrainResult:
        history = []
        start = time.perf_counter()
        episodes = int(self.cfg.train.episodes)
        horizon = int(self.cfg.train.rollout_steps)
        for episode in range(1, episodes + 1):
            preference = sample_preferences(
                self.cfg,
                self.num_envs,
                self.device,
            )
            load_multiplier = sample_training_load_multipliers(
                self.cfg,
                self.num_envs,
                self.device,
                self.load_generator,
            )
            self.current_load_multiplier = float(load_multiplier.mean().item())
            self.current_load_multiplier_std = float(
                load_multiplier.float().std(unbiased=False).item()
            )
            obs = self.env.reset(
                preference,
                load_multiplier=load_multiplier,
            )
            for step in range(horizon):
                macro = step % int(self.cfg.system.macro_steps) == 0
                if macro:
                    self.policy.act(preference, training=False)
                    self.env.set_uav_dispatch(self.policy.dispatch(obs))
                    obs = self.env.observe()
                actions, _ = self.policy.act(preference, training=True)
                obs, _, _, _ = self.env.step(actions)
            record = {
                "episode": episode,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "load_multiplier": self.current_load_multiplier,
                "load_multiplier_std": self.current_load_multiplier_std,
                **self.env.metrics.mean_dict(),
                "bce_loss": self.policy.last_loss,
                "candidate_k": self.policy.current_k,
            }
            history.append(record)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[official_droo] episode {episode:03d}/{episodes} "
                f"load={self.current_load_multiplier:.2f}±{self.current_load_multiplier_std:.2f} "
                f"reward={record['reward']:.3f} completion={record['completion_rate']:.3f} "
                f"L/R/U={record['local_ratio']:.2f}/{record['rsu_ratio']:.2f}/"
                f"{record['uav_ratio']:.2f} K={self.policy.current_k}",
                flush=True,
            )
            if episode % int(self.cfg.train.checkpoint_every) == 0 or episode == episodes:
                self._save(episode)

        summary = {
            "seed": self.seed,
            "method": self.method,
            "device": str(self.device),
            "num_envs": self.num_envs,
            "episodes": episodes,
            "source": source_for(self.method),
            "last": history[-1],
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


def train_sota_baseline(
    config: Config,
    output_dir: str | Path,
    seed: int,
    device: str,
    method: str,
) -> OfficialTrainResult:
    if method == "official_cavdn":
        return OfficialCAVDNTrainer(config, output_dir, seed, device).train()
    if method == "official_droo":
        return OfficialDROOTrainer(config, output_dir, seed, device).train()
    raise ValueError(f"Unknown SOTA method: {method}")


@torch.no_grad()
def evaluate_sota_baseline(
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
    preference_values = (
        config.official_baselines.preference
        if preference is None
        else preference
    )
    preference = torch.as_tensor(
        preference_values,
        dtype=torch.float32,
        device=device,
    ).view(1, 2).expand(episodes, -1)
    env = VectorOffloadEnv(config, episodes, device, seed, workflow_split="eval")
    obs = env.reset(preference, load_multiplier=load_multiplier)
    if scenario is not None:
        scenario.after_reset(env)
        obs = env.observe()
    if obs_filter is not None:
        obs = obs_filter(obs)
    horizon = int(config.system.episode_steps)

    if method == "official_cavdn":
        adapter = OfficialObservationAdapter(config, env)
        policy = _CAVDNPolicy(adapter, config, device)
        policy.load_state_dict(payload["policy"])
        local, _ = adapter.encode_torch(obs, True)
        hidden = policy.initial_hidden(episodes)
        last_action = torch.zeros(
            (episodes, adapter.n_agents, adapter.shared_action_n),
            device=device,
        )
        epsilon = 0.0 if deterministic else float(config.cavdn.epsilon_end)
        for step in range(horizon):
            macro = step % int(config.system.macro_steps) == 0
            available = torch.zeros(
                (episodes, adapter.n_agents, adapter.shared_action_n),
                dtype=torch.bool,
                device=device,
            )
            available[:, : env.v, : env.action_n] = obs["action_mask"]
            if macro:
                available[:, env.v :, : env.region_n + 1] = obs[
                    "uav_action_mask"
                ]
            else:
                available[:, env.v :, 0] = True
            hidden_before = hidden
            q_values, pre_hidden = policy.q_values(
                local,
                last_action,
                hidden_before,
            )
            greedy = q_values.masked_fill(~available, -1e9).argmax(dim=-1)
            random_action = torch.multinomial(
                available.float().flatten(0, 1),
                1,
            ).view(episodes, adapter.n_agents)
            explore = torch.rand_like(greedy.float()) < epsilon
            pre_actions = torch.where(explore, random_action, greedy)
            if macro:
                env.set_uav_dispatch(pre_actions[:, env.v :])
                obs = env.observe()
                if obs_filter is not None:
                    obs = obs_filter(obs)
                post_local, _ = adapter.encode_torch(obs, False)
                post_available = torch.zeros_like(available)
                post_available[:, : env.v, : env.action_n] = obs["action_mask"]
                post_available[:, env.v :, 0] = True
                hybrid_local = local.clone()
                hybrid_local[:, : env.v] = post_local[:, : env.v]
                hybrid_available = available.clone()
                hybrid_available[:, : env.v] = post_available[:, : env.v]
                hybrid_q, hybrid_hidden = policy.q_values(
                    hybrid_local,
                    last_action,
                    hidden_before,
                )
                hybrid_greedy = hybrid_q.masked_fill(
                    ~hybrid_available,
                    -1e9,
                ).argmax(dim=-1)
                hybrid_random = torch.multinomial(
                    hybrid_available.float().flatten(0, 1),
                    1,
                ).view(episodes, adapter.n_agents)
                hybrid_explore = (
                    torch.rand_like(hybrid_greedy.float()) < epsilon
                )
                actions = torch.where(
                    hybrid_explore,
                    hybrid_random,
                    hybrid_greedy,
                )
                actions[:, env.v :] = pre_actions[:, env.v :]
                hidden = hybrid_hidden
                hidden[:, env.v :] = pre_hidden[:, env.v :]
                local = hybrid_local
            else:
                actions = pre_actions
                hidden = pre_hidden
            obs, _, _, _ = env.step(actions[:, : env.v])
            if scenario is not None:
                scenario.after_step(env)
                obs = env.observe()
            if obs_filter is not None:
                obs = obs_filter(obs)
            next_macro = (step + 1) % int(config.system.macro_steps) == 0
            local, _ = adapter.encode_torch(obs, next_macro)
            last_action = F.one_hot(actions, adapter.shared_action_n).float()
    elif method == "official_droo":
        policy = _DROOPolicy(config, env, device)
        policy.load_state_dict(payload["policy"])
        for step in range(horizon):
            macro = step % int(config.system.macro_steps) == 0
            if macro:
                policy.act(preference, training=False)
                env.set_uav_dispatch(policy.dispatch(obs))
                obs = env.observe()
                if obs_filter is not None:
                    obs = obs_filter(obs)
            actions, _ = policy.act(preference, training=False)
            obs, _, _, _ = env.step(actions)
            if scenario is not None:
                scenario.after_step(env)
                obs = env.observe()
            if obs_filter is not None:
                obs = obs_filter(obs)
    else:
        raise ValueError(f"Unsupported SOTA checkpoint method: {method}")
    return env.summary_metrics()
