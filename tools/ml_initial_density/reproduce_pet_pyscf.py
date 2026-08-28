#!/usr/bin/env python3
"""Reproduce the Atomistic Cookbook PET initial-density SCF comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import ase
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
        "--recipe-dir",
        type=Path,
        default=DEFAULT_SCRATCH / "datasets/ml-density",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_SCRATCH / "checkpoints/pet-density.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCRATCH / "benchmark_results/pet_pyscf_reproduction",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Metatomic device, for example cpu or cuda (default: cpu)",
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed(callable_):
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    data_dir = args.recipe_dir / "data"
    sys.path.insert(0, str(data_dir))
    from rho_utils import (  # pylint: disable=import-outside-toplevel
        atoms_to_pyscf,
        dm_from_ri_coefficients,
        run_scf,
    )

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
    xc = "pbe"
    auxbasis = "def2-universal-jfit"
    target_name = "mtt::rho_c_jfit_overlap"

    molecule = atoms_to_pyscf(atoms, basis)
    mf_for_guess = pyscf.dft.RKS(molecule)
    mf_for_guess.xc = xc
    dm_sad, timings_sad_guess = timed(mf_for_guess.get_init_guess)
    (mf_sad, n_sad), timings_sad_scf = timed(
        lambda: run_scf(atoms, xc, basis, dm0=dm_sad)
    )

    ref_coefficients, timings_ref_load = timed(lambda: mts.load(str(reference_path)))
    dm_ri, timings_ri_conversion = timed(
        lambda: dm_from_ri_coefficients(
            atoms, ref_coefficients, xc, basis, auxbasis
        )
    )
    (mf_ri, n_ri), timings_ri_scf = timed(
        lambda: run_scf(atoms, xc, basis, dm0=dm_ri)
    )

    model, timings_model_load = timed(lambda: load_atomistic_model(str(args.model)))
    calculator = MetatomicCalculator(model, device=args.device)
    ml_coefficients, timings_pet_forward = timed(
        lambda: calculator.run_model(
            atoms, {target_name: ModelOutput(per_atom=True)}
        )[target_name]
    )
    dm_ml, timings_ml_conversion = timed(
        lambda: dm_from_ri_coefficients(
            atoms, ml_coefficients, xc, basis, auxbasis
        )
    )
    (mf_ml, n_ml), timings_ml_scf = timed(
        lambda: run_scf(atoms, xc, basis, dm0=dm_ml)
    )

    overlap = molecule.intor("int1e_ovlp")
    electron_counts = {
        "sad": float(np.einsum("ij,ji", dm_sad, overlap)),
        "ri_reference": float(np.einsum("ij,ji", dm_ri, overlap)),
        "pet": float(np.einsum("ij,ji", dm_ml, overlap)),
    }
    energies = {
        "sad": float(mf_sad.e_tot),
        "ri_reference": float(mf_ri.e_tot),
        "pet": float(mf_ml.e_tot),
    }
    cycles = {"sad": n_sad, "ri_reference": n_ri, "pet": n_ml}
    timings = {
        "sad_guess_s": timings_sad_guess,
        "sad_scf_s": timings_sad_scf,
        "reference_load_s": timings_ref_load,
        "ri_reference_to_dm_s": timings_ri_conversion,
        "ri_reference_scf_s": timings_ri_scf,
        "model_load_s": timings_model_load,
        "pet_forward_s": timings_pet_forward,
        "pet_to_dm_s": timings_ml_conversion,
        "pet_started_scf_s": timings_ml_scf,
    }

    if not all(mf.converged for mf in (mf_sad, mf_ri, mf_ml)):
        raise RuntimeError("at least one SCF calculation did not converge")
    if max(energies.values()) - min(energies.values()) > 1.0e-8:
        raise RuntimeError(f"converged energies disagree: {energies}")
    for name, count in electron_counts.items():
        if abs(count - molecule.nelectron) > 1.0e-8:
            raise RuntimeError(f"{name} density has {count} electrons")

    metadata = {
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "reference": str(reference_path.resolve()),
        "reference_sha256": sha256(reference_path),
        "device_requested": args.device,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "pyscf_version": pyscf.__version__,
        "ase_version": ase.__version__,
        "basis": basis,
        "xc": xc,
        "auxbasis": auxbasis,
        "electron_counts": electron_counts,
        "scf_cycles": cycles,
        "converged_energies_hartree": energies,
        "timings": timings,
    }

    np.savez_compressed(
        args.output_dir / "initial_densities_pyscf.npz",
        dm_sad=dm_sad,
        dm_ri_reference=dm_ri,
        dm_pet=dm_ml,
        overlap=overlap,
        numbers=atoms.numbers,
        positions_angstrom=atoms.positions,
    )
    with (args.output_dir / "result.json").open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output_dir / 'result.json'}")
    print(f"wrote {args.output_dir / 'initial_densities_pyscf.npz'}")


if __name__ == "__main__":
    main()
