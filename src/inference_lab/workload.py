"""Workload generators for synthetic LLM inference experiments with strict reproducibility."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from inference_lab.simulator import Request


@dataclass(frozen=True)
class WorkloadConfig:
    """Parameters for synthetic request generation."""
    num_requests: int
    prompt_length: int
    output_length: int
    arrival_pattern: str = "poisson"  # "poisson" or "uniform" or "burst"
    arrival_rate_rps: float = 10.0    # Requests per second (for Poisson)
    seed: int = 42


def generate_workload(config: WorkloadConfig) -> list[Request]:
    """Generate a reproducible sequence of requests with explicit arrival times."""
    rng = random.Random(config.seed)
    requests: list[Request] = []

    current_arrival = 0.0
    for req_id in range(config.num_requests):
        if config.arrival_pattern == "burst":
            # All requests arrive simultaneously at t=0
            current_arrival = 0.0
        elif config.arrival_pattern == "uniform":
            # Deterministic uniform inter-arrival time
            if config.arrival_rate_rps > 0:
                current_arrival += 1.0 / config.arrival_rate_rps
        elif config.arrival_pattern == "poisson":
            # Exponential inter-arrival times
            if config.arrival_rate_rps > 0:
                inter_arrival = rng.expovariate(config.arrival_rate_rps)
                current_arrival += inter_arrival
        else:
            raise ValueError(f"Unknown arrival pattern: {config.arrival_pattern}")

        requests.append(
            Request(
                request_id=req_id,
                arrival_time=current_arrival,
                prompt_length=config.prompt_length,
                output_length=config.output_length,
            )
        )

    return requests
