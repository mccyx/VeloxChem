#!/usr/bin/env python3
"""Interleave repeated native VeloxChem SAD/PET/reference-RI SCF timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import veloxchem as vlx

from reproduce_pet_veloxchem import make_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def run_scf(initial_density):
    molecule, basis = make_system()
    driver = vlx.ScfRestrictedDriver()
    driver.xcfun = "PBE"
    driver.acc_type = "DIIS"
    driver.conv_thresh = 1.0e-8
    driver.print_level = 1
    start = time.perf_counter()
    driver.compute(molecule, basis, initial_density=initial_density)
    elapsed = time.perf_counter() - start
    if not driver.is_converged:
        raise RuntimeError("VeloxChem SCF did not converge")
    return elapsed, int(driver.num_iter), float(driver.scf_energy)


def summarize(values):
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
        raise ValueError("use at least three repeats to rotate all orders")
    arrays = np.load(args.densities)
    densities = {
        "sad": None,
        "pet": arrays["dm_pet"],
        "ri_reference": arrays["dm_ri_reference"],
    }
    warmup = {}
    for name, density in densities.items():
        elapsed, cycles, energy = run_scf(density)
        warmup[name] = {
            "elapsed_s": elapsed,
            "iterations": cycles,
            "energy_hartree": energy,
        }

    timings = {name: [] for name in densities}
    iterations = {name: [] for name in densities}
    energies = {name: [] for name in densities}
    base_order = list(densities)
    orders = []
    for repeat in range(args.repeats):
        offset = repeat % len(base_order)
        order = base_order[offset:] + base_order[:offset]
        orders.append(order)
        for name in order:
            elapsed, cycles, energy = run_scf(densities[name])
            timings[name].append(elapsed)
            iterations[name].append(cycles)
            energies[name].append(energy)

    all_energies = [energy for values in energies.values() for energy in values]
    result = {
        "repeats": args.repeats,
        "warmup": warmup,
        "timings": {
            name: {
                **summarize(timings[name]),
                "iterations": iterations[name],
                "energies_hartree": energies[name],
            }
            for name in densities
        },
        "execution_orders": orders,
        "energy_spread_hartree": max(all_energies) - min(all_energies),
    }
    if result["energy_spread_hartree"] > 1.0e-8:
        raise RuntimeError("VeloxChem converged energies disagree")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
