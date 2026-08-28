#!/usr/bin/env python3
"""Run one lightweight PET density-model inference and report its shape/time."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from ase import Atoms
from metatomic.torch import ModelOutput, load_atomistic_model

try:
    from metatomic.torch.ase_calculator import MetatomicCalculator
except ImportError:
    from metatomic_ase import MetatomicCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args()

    atoms = Atoms(
        numbers=[8, 6, 6, 1, 8, 8, 1],
        positions=[
            [-1.3058, -1.0321, -1.0321],
            [-1.0843, -0.1924, -0.1924],
            [0.3229, 0.2412, 0.2412],
            [-1.8735, 0.3561, 0.3561],
            [0.4843, 1.0926, 1.0926],
            [1.3055, -0.4059, -0.4059],
            [2.1509, -0.0596, -0.0596],
        ],
    )
    atoms.center(about=atoms.get_center_of_mass())

    start = time.perf_counter()
    model = load_atomistic_model(str(args.model))
    calculator = MetatomicCalculator(model, device=args.device)
    load_s = time.perf_counter() - start

    target_name = "mtt::rho_c_jfit_overlap"
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    coefficients = calculator.run_model(
        atoms, {target_name: ModelOutput(per_atom=True)}
    )[target_name]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    forward_s = time.perf_counter() - start

    for _ in range(args.warmup):
        calculator.run_model(atoms, {target_name: ModelOutput(per_atom=True)})
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    steady_times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        calculator.run_model(atoms, {target_name: ModelOutput(per_atom=True)})
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        steady_times.append(time.perf_counter() - start)

    blocks = list(coefficients.blocks())
    result = {
        "device_requested": args.device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "atoms": len(atoms),
        "output_blocks": len(blocks),
        "output_values": sum(block.values.numel() for block in blocks),
        "model_and_calculator_load_s": load_s,
        "first_forward_s": forward_s,
        "warmup_forwards": args.warmup,
        "steady_forward_s": (
            {
                "repeats": len(steady_times),
                "mean": statistics.fmean(steady_times),
                "median": statistics.median(steady_times),
                "min": min(steady_times),
                "max": max(steady_times),
                "all": steady_times,
            }
            if steady_times
            else None
        ),
    }
    print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
