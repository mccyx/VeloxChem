#!/usr/bin/env python3
"""Repeat and interleave SAD, reference-RI, and PET initial-density timings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import metatensor.torch as mts
import numpy as np
import pyscf
import torch
from ase import Atoms
from metatomic.torch import ModelOutput, load_atomistic_model

try:
    from metatomic.torch.ase_calculator import MetatomicCalculator
except ImportError:
    from metatomic_ase import MetatomicCalculator


DEFAULT_SCRATCH = Path(
    os.environ.get(
        "VLX_ML_DENSITY_SCRATCH",
        "/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe-dir", type=Path, default=DEFAULT_SCRATCH / "datasets/ml-density"
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_SCRATCH / "checkpoints/pet-density.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCRATCH / "benchmark_results/pet_repeated_timing",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(callable_, device: str = "cpu"):
    synchronize(device)
    start = time.perf_counter()
    value = callable_()
    synchronize(device)
    return value, time.perf_counter() - start


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values)
    return {
        "values_s": values,
        "median_s": float(np.median(array)),
        "mean_s": float(np.mean(array)),
        "min_s": float(np.min(array)),
        "max_s": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("use at least three repeats to rotate all SCF orders")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    data_dir = args.recipe_dir / "data"
    sys.path.insert(0, str(data_dir))
    from rho_utils import atoms_to_pyscf, dm_from_ri_coefficients

    reference_path = data_dir / "scfbench_test_molecule_3464_rho_c_jfit.mts"
    for path in (args.model, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

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

    basis = "def2-svp"
    auxbasis = "def2-universal-jfit"
    xc = "pbe"
    target_name = "mtt::rho_c_jfit_overlap"

    reference = mts.load(str(reference_path))
    model, model_load_s = timed(lambda: load_atomistic_model(str(args.model)))
    calculator, calculator_create_s = timed(
        lambda: MetatomicCalculator(model, device=args.device), args.device
    )

    def pet_forward():
        return calculator.run_model(
            atoms, {target_name: ModelOutput(per_atom=True)}
        )[target_name]

    # Pay CUDA/JIT cold-start before collecting steady-state measurements.
    _, cold_forward_s = timed(pet_forward, args.device)
    _, warmup_forward_s = timed(pet_forward, args.device)

    sad_guess_times: list[float] = []
    ri_to_dm_times: list[float] = []
    pet_forward_times: list[float] = []
    pet_to_dm_times: list[float] = []
    dm_sad = dm_ri = dm_pet = None

    for _ in range(args.repeats):
        mol = atoms_to_pyscf(atoms, basis)
        mf = pyscf.dft.RKS(mol)
        mf.xc = xc
        dm_sad, elapsed = timed(mf.get_init_guess)
        sad_guess_times.append(elapsed)

        dm_ri, elapsed = timed(
            lambda: dm_from_ri_coefficients(atoms, reference, xc, basis, auxbasis)
        )
        ri_to_dm_times.append(elapsed)

        coefficients, elapsed = timed(pet_forward, args.device)
        pet_forward_times.append(elapsed)
        dm_pet, elapsed = timed(
            lambda: dm_from_ri_coefficients(
                atoms, coefficients, xc, basis, auxbasis
            ),
            args.device,
        )
        pet_to_dm_times.append(elapsed)

    assert dm_sad is not None and dm_ri is not None and dm_pet is not None
    densities = {"sad": dm_sad, "ri_reference": dm_ri, "pet": dm_pet}
    scf_times = {name: [] for name in densities}
    scf_cycles = {name: [] for name in densities}
    scf_energies = {name: [] for name in densities}
    base_order = ["sad", "ri_reference", "pet"]
    execution_orders: list[list[str]] = []

    def run_quiet_scf(dm0):
        mol = atoms_to_pyscf(atoms, basis)
        mf = pyscf.dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0
        cycles = 0

        def count_cycle(_):
            nonlocal cycles
            cycles += 1

        mf.callback = count_cycle
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        return float(mf.e_tot), cycles

    for repeat in range(args.repeats):
        offset = repeat % len(base_order)
        order = base_order[offset:] + base_order[:offset]
        execution_orders.append(order)
        for name in order:
            (energy, cycles), elapsed = timed(
                lambda name=name: run_quiet_scf(densities[name])
            )
            scf_times[name].append(elapsed)
            scf_cycles[name].append(cycles)
            scf_energies[name].append(energy)

    all_energies = [energy for values in scf_energies.values() for energy in values]
    if max(all_energies) - min(all_energies) > 1.0e-8:
        raise RuntimeError("converged energies disagree")

    result = {
        "device": args.device,
        "threads": args.threads,
        "repeats": args.repeats,
        "torch_version": torch.__version__,
        "pyscf_version": pyscf.__version__,
        "basis": basis,
        "auxbasis": auxbasis,
        "xc": xc,
        "one_time_s": {
            "model_load": model_load_s,
            "calculator_create": calculator_create_s,
            "cold_pet_forward": cold_forward_s,
            "warmup_pet_forward": warmup_forward_s,
        },
        "density_generation": {
            "sad_guess": summarize(sad_guess_times),
            "ri_reference_to_dm": summarize(ri_to_dm_times),
            "pet_forward_steady": summarize(pet_forward_times),
            "pet_to_dm": summarize(pet_to_dm_times),
        },
        "scf": {
            name: {
                **summarize(scf_times[name]),
                "cycles": scf_cycles[name],
                "energies_hartree": scf_energies[name],
            }
            for name in densities
        },
        "scf_execution_orders": execution_orders,
    }
    output_path = args.output_dir / "result.json"
    with output_path.open("w", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
