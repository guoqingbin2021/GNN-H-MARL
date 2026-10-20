# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import math

import torch


def semantic_priority_class(
    high_value: torch.Tensor,
    urgent: torch.Tensor,
) -> torch.Tensor:
    """Map (high-value, urgent) tags to paper priority classes 1..4."""
    high = high_value.to(torch.long)
    urgent_value = urgent.to(torch.long)
    return 4 - 2 * high - urgent_value


def lookup_priority_weight(
    priority_class: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return weights[(priority_class.long() - 1).clamp(0, len(weights) - 1)]


def weighted_proportional_share(
    active: torch.Tensor,
    node_index: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Closed-form weighted proportional-fair share for each active request."""
    totals = torch.zeros(
        (*active.shape[:-1], num_nodes),
        dtype=weights.dtype,
        device=weights.device,
    )
    safe_index = node_index.long().clamp(0, num_nodes - 1)
    totals.scatter_add_(-1, safe_index, torch.where(active, weights, 0.0))
    selected_total = totals.gather(-1, safe_index).clamp_min(1e-12)
    return torch.where(active, weights / selected_total, 0.0)


def shannon_rate(
    bandwidth_hz: float,
    share: torch.Tensor,
    tx_power_w: float,
    channel_gain: torch.Tensor,
    noise_psd_w_hz: float,
) -> torch.Tensor:
    allocated_bandwidth = float(bandwidth_hz) * share.clamp_min(1e-8)
    noise_power = float(noise_psd_w_hz) * allocated_bandwidth
    snr = float(tx_power_w) * channel_gain / noise_power.clamp_min(1e-30)
    rate = allocated_bandwidth * torch.log2(1.0 + snr)
    return torch.where(share > 0.0, rate, torch.zeros_like(rate))


def remaining_contact_time(
    vehicle_xy: torch.Tensor,
    resource_xy: torch.Tensor,
    vehicle_velocity_xy: torch.Tensor,
    coverage_radius_m: torch.Tensor,
    resource_velocity_xy: torch.Tensor | None = None,
    epsilon: float = 1e-3,
    *,
    resource_motion_segments: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> torch.Tensor:
    """Return the exact time until an active line--circle contact ends.

    The larger root of ``||r + v_rel * tau||^2 = R^2`` is the exit time
    for a terminal that is currently inside the coverage disk.  A terminal
    outside the disk has no active contact and therefore receives zero,
    irrespective of whether it may enter the disk later.

    With ``resource_motion_segments``, each pair is a resource velocity and
    its finite duration (broadcastable over the resource dimension). The
    resource stops after the last segment; the vehicle keeps its observed
    velocity. This models a UAV reaching its current mission endpoint without
    extrapolating that flight forever. It does not anticipate future dispatches.
    """
    displacement = vehicle_xy.unsqueeze(-2) - resource_xy
    radius = torch.as_tensor(
        coverage_radius_m,
        dtype=displacement.dtype,
        device=displacement.device,
    )
    vehicle_velocity = vehicle_velocity_xy.unsqueeze(-2)
    if resource_motion_segments is not None:
        if resource_velocity_xy is not None:
            raise ValueError("Specify either constant resource velocity or finite motion segments")
        inside = displacement.square().sum(dim=-1) <= radius.square()
        elapsed = torch.zeros_like(displacement[..., 0])
        result = torch.where(inside, torch.full_like(elapsed, torch.inf), elapsed)
        connected = inside
        for velocity, duration in resource_motion_segments:
            duration = torch.as_tensor(
                duration, dtype=displacement.dtype, device=displacement.device
            ).clamp_min(0.0)
            relative_velocity = vehicle_velocity - velocity
            exit_time = _constant_velocity_contact_time(
                displacement, relative_velocity, radius, epsilon
            )
            # Reaching the boundary exactly at a velocity change need not end
            # contact: the next segment may stop or head back inside the disk.
            leaves = connected & (exit_time < duration)
            result = torch.where(leaves, elapsed + exit_time, result)
            connected &= ~leaves
            displacement = displacement + relative_velocity * duration.unsqueeze(-1)
            elapsed = elapsed + duration
        stopped_exit = _constant_velocity_contact_time(
            displacement, vehicle_velocity, radius, epsilon
        )
        return torch.where(connected, elapsed + stopped_exit, result)
    if resource_velocity_xy is None:
        resource_velocity_xy = torch.zeros_like(resource_xy)
    return _constant_velocity_contact_time(
        displacement, vehicle_velocity - resource_velocity_xy, radius, epsilon
    )


def _constant_velocity_contact_time(
    displacement: torch.Tensor,
    relative_velocity: torch.Tensor,
    radius: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Exit time for one straight relative-motion segment, from its start."""
    a = relative_velocity.square().sum(dim=-1)
    b = 2.0 * (displacement * relative_velocity).sum(dim=-1)
    c = displacement.square().sum(dim=-1) - radius.square()
    inside = c <= 0.0
    stationary = a <= float(epsilon) ** 2
    discriminant = (b.square() - 4.0 * a * c).clamp_min(0.0)
    exit_time = (-b + torch.sqrt(discriminant)) / (
        2.0 * a.clamp_min(float(epsilon) ** 2)
    )
    finite_or_infinite = torch.where(
        stationary,
        torch.full_like(exit_time, torch.inf),
        exit_time.clamp_min(0.0),
    )
    return torch.where(inside, finite_or_infinite, torch.zeros_like(exit_time))


def scalarized_delay_energy_reward(
    preference: torch.Tensor,
    delay_cost: torch.Tensor,
    energy_cost: torch.Tensor,
    *,
    delay_bounds: tuple[float, float],
    energy_bounds: tuple[float, float],
) -> torch.Tensor:
    """Fixed-bound two-objective reward used by every policy and channel."""
    delay_min, delay_max = map(float, delay_bounds)
    energy_min, energy_max = map(float, energy_bounds)
    delay_score = -(delay_cost - delay_min) / max(delay_max - delay_min, 1e-6)
    energy_score = -(energy_cost - energy_min) / max(
        energy_max - energy_min,
        1e-6,
    )
    delay_weight = preference[..., 0]
    energy_weight = preference[..., 1]
    while delay_weight.ndim < delay_score.ndim:
        delay_weight = delay_weight.unsqueeze(-1)
        energy_weight = energy_weight.unsqueeze(-1)
    return delay_weight * delay_score + energy_weight * energy_score


def compute_duration(
    cycles: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> torch.Tensor:
    """Continuous event-time duration under an allocated aggregate service rate."""
    return cycles / frequency_hz.clamp_min(1.0)


def compute_energy(
    cycles: torch.Tensor,
    energy_per_gcycle_j: torch.Tensor | float,
) -> torch.Tensor:
    """Hardware-calibrated active energy without treating aggregate GHz as core DVFS."""
    return cycles / 1e9 * torch.as_tensor(
        energy_per_gcycle_j,
        dtype=cycles.dtype,
        device=cycles.device,
    )


def rotary_wing_level_power(
    speed_mps: torch.Tensor,
    *,
    profile_power_w: float,
    induced_power_w: float,
    tip_speed_mps: float,
    mean_rotor_velocity_mps: float,
    fuselage_drag_ratio: float,
    air_density_kg_m3: float,
    rotor_solidity: float,
    rotor_area_m2: float,
) -> torch.Tensor:
    speed2 = speed_mps.square()
    profile = float(profile_power_w) * (
        1.0 + 3.0 * speed2 / float(tip_speed_mps) ** 2
    )
    induced_inner = torch.sqrt(
        1.0 + speed2.square() / (4.0 * float(mean_rotor_velocity_mps) ** 4)
    ) - speed2 / (2.0 * float(mean_rotor_velocity_mps) ** 2)
    induced = float(induced_power_w) * torch.sqrt(induced_inner.clamp_min(0.0))
    parasite = (
        0.5
        * float(fuselage_drag_ratio)
        * float(air_density_kg_m3)
        * float(rotor_solidity)
        * float(rotor_area_m2)
        * speed_mps.pow(3)
    )
    return profile + induced + parasite


def dbm_per_hz_to_w_per_hz(value: float) -> float:
    return 10.0 ** ((float(value) - 30.0) / 10.0)


def free_space_path_loss_db(distance_m: torch.Tensor, carrier_ghz: float) -> torch.Tensor:
    wavelength_factor = 4.0 * math.pi * float(carrier_ghz) * 1e9 / 299_792_458.0
    return 20.0 * torch.log10(distance_m.clamp_min(1.0) * wavelength_factor)
