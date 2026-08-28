#!/usr/bin/env python3
"""Compare VeloxChem PBE SCF from SAD and an external PET AO density."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import veloxchem as vlx


POSITIONS = [
    ("O", -1.3058, -1.0321, -1.0321),
    ("C", -1.0843, -0.1924, -0.1924),
    ("C", 0.3229, 0.2412, 0.2412),
    ("H", -1.8735, 0.3561, 0.3561),
    ("O", 0.4843, 1.0926, 1.0926),
    ("O", 1.3055, -0.4059, -0.4059),
    ("H", 2.1509, -0.0596, -0.0596),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_system():
    text = "\n".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, x, y, z in POSITIONS
    )
    molecule = vlx.Molecule.read_molecule_string(text, "angstrom")
    basis = vlx.MolecularBasis.read(molecule, "def2-svp", ostream=None)
    return molecule, basis


def run_scf(initial_density=None):
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
    return {
        "energy_hartree": float(driver.scf_energy),
        "iterations": int(driver.num_iter),
        "elapsed_s": elapsed,
    }


def main() -> None:
    args = parse_args()
    densities = np.load(args.densities)
    result = {
        "basis": "def2-svp",
        "xc": "PBE",
        "sad": run_scf(),
        "pet": run_scf(densities["dm_pet"]),
        "ri_reference": run_scf(densities["dm_ri_reference"]),
    }
    energies = [
        entry["energy_hartree"]
        for entry in result.values()
        if isinstance(entry, dict)
    ]
    result["energy_spread_hartree"] = max(energies) - min(energies)
    if result["energy_spread_hartree"] > 1.0e-8:
        raise RuntimeError(f"VeloxChem converged energies disagree: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
