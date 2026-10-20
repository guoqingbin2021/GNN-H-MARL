from __future__ import annotations

import torch


def select_nonpreemptive_job(
    ready_time: torch.Tensor,
    priority_class: torch.Tensor,
    vehicle_id: torch.Tensor,
    subtask_id: torch.Tensor,
    now: float | torch.Tensor,
    *,
    epsilon: float = 1e-7,
) -> int | None:
    """Select the next arrived job using a deterministic priority discipline.

    Smaller priority classes are served first.  Ties are resolved by arrival
    time, vehicle id, subtask id, and finally the original array order.  Jobs
    arriving after ``now`` cannot preempt a job that can start immediately.
    """
    if ready_time.ndim != 1:
        raise ValueError("ready_time must be one-dimensional")
    current = torch.as_tensor(now, dtype=ready_time.dtype, device=ready_time.device)
    eligible = ready_time <= current + float(epsilon)
    if not bool(eligible.any()):
        return None

    candidate_indices = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    return min(
        candidate_indices,
        key=lambda index: (
            int(priority_class[index].item()),
            float(ready_time[index].item()),
            int(vehicle_id[index].item()),
            int(subtask_id[index].item()),
            int(index),
        ),
    )

