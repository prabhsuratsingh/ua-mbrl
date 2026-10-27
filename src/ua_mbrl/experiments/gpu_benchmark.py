"""Ensemble MPPI latency: per-member loop vs stacked-einsum batched inference,
on CPU and CUDA (gpu_mppi_benchmark.py, tab:gpuspeedups row 1 with
``--horizon 10``)."""

import time

import numpy as np
import torch

from ua_mbrl.core.dynamics import GenericEnsembleDynamics
from ua_mbrl.core.mppi import GenericMPPI


def add_arguments(parser):
    parser.add_argument("--state-dim", type=int, default=13)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rollouts", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--devices", nargs="+", default=None, help="default: cpu and cuda if available")


def benchmark(device, state_dim, action_dim, ensemble_size, rollouts, horizon, steps, vectorized=True):
    model = GenericEnsembleDynamics(
        state_dim=state_dim, action_dim=action_dim, ensemble_size=ensemble_size, hidden_dim=128, device=device,
    )
    model.use_vectorized_inference = vectorized
    controller = GenericMPPI(
        dynamics=model, action_dim=action_dim, horizon=horizon, rollouts=rollouts,
        action_low=-np.ones(action_dim), action_high=np.ones(action_dim), target=np.zeros(state_dim),
        device=device,
    )
    state = np.zeros(state_dim, dtype=np.float32)
    for _ in range(5):
        controller.command(state)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(steps):
            controller.command(state)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        for _ in range(steps):
            controller.command(state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    total = steps * rollouts * horizon * ensemble_size
    return {
        "device": str(device), "ensemble": ensemble_size, "rollouts": rollouts, "horizon": horizon,
        "steps": steps, "total_model_evaluations": total, "total_ms": elapsed_ms,
        "ms_per_command": elapsed_ms / steps, "model_evaluations_per_second": total / (elapsed_ms / 1000.0),
        "vectorized": vectorized,
    }


def run(args, run_dir):
    if args.devices:
        devices = [torch.device(d) for d in args.devices]
    else:
        devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    print(f"state_dim={args.state_dim} action_dim={args.action_dim} horizon={args.horizon}")
    print("device    E  K     N   path    latency_ms model_eval/s")
    print("-" * 72)
    for device in devices:
        for ensemble_size in (1, 5):
            for rollouts in args.rollouts:
                for vectorized in (False, True) if ensemble_size == 5 else (True,):
                    result = benchmark(device, args.state_dim, args.action_dim, ensemble_size, rollouts,
                                       args.horizon, args.steps, vectorized)
                    run_dir.append(result)
                    print(
                        f"device={result['device']:<8} E={result['ensemble']:<2} K={result['rollouts']:<5} "
                        f"N={result['horizon']:<3} path={'batched' if result['vectorized'] else 'loop':<7} "
                        f"latency_ms={result['ms_per_command']:.3f} "
                        f"model_eval/s={result['model_evaluations_per_second']:.2e}"
                    )
