#!/usr/bin/env python3
"""Export def2-SVP AO metadata and overlap matrices from PySCF or VeloxChem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NUMBERS = [8, 6, 6, 1, 8, 8, 1]
SYMBOLS = ["O", "C", "C", "H", "O", "O", "H"]
POSITIONS = np.array(
    [
        [-1.3058, -1.0321, -1.0321],
        [-1.0843, -0.1924, -0.1924],
        [0.3229, 0.2412, 0.2412],
        [-1.8735, 0.3561, 0.3561],
        [0.4843, 1.0926, 1.0926],
        [1.3055, -0.4059, -0.4059],
        [2.1509, -0.0596, -0.0596],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("pyscf", "veloxchem", "compare"))
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def export_pyscf():
    from pyscf import gto

    atom_spec = list(zip(SYMBOLS, POSITIONS))
    molecule = gto.M(atom=atom_spec, basis="def2-svp", unit="Angstrom")
    labels = [list(label) for label in molecule.ao_labels(fmt=False)]
    return molecule.intor("int1e_ovlp"), labels


def export_veloxchem():
    import veloxchem as vlx
    from veloxchem.veloxchemlib import GpuDevices, ScreeningData
    from veloxchem.veloxchemlib import compute_overlap_and_kinetic_energy_integrals_gpu

    molecule_text = "\n".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in zip(SYMBOLS, POSITIONS)
    )
    molecule = vlx.Molecule.read_molecule_string(molecule_text, "angstrom")
    basis = vlx.MolecularBasis.read(molecule, "def2-svp", ostream=None)
    num_gpus = GpuDevices().get_number_devices()
    screening = ScreeningData(
        molecule, basis, num_gpus, 1.0e-12, 1.0e-12, 0, 1
    )
    overlap, _ = compute_overlap_and_kinetic_energy_integrals_gpu(
        molecule, basis, screening
    )
    shell_metadata = []
    for atom in range(molecule.number_of_atoms()):
        for angular in range(basis.max_angular_momentum([atom]) + 1):
            functions = basis.get_basis_functions([atom], angular)
            for radial, function in enumerate(functions):
                shell_metadata.append(
                    {
                        "atom": atom,
                        "angular": angular,
                        "radial": radial,
                        "n_primitives": function.number_of_primitives(),
                    }
                )
    return overlap.to_numpy(), shell_metadata


def compare_exports(output: Path) -> None:
    with (output / "metadata_pyscf.json").open(encoding="utf8") as handle:
        pyscf_labels = json.load(handle)
    with (output / "metadata_veloxchem.json").open(encoding="utf8") as handle:
        veloxchem_shells = json.load(handle)
    overlap_pyscf = np.load(output / "overlap_pyscf.npy")
    overlap_veloxchem = np.load(output / "overlap_veloxchem.npy")

    angular_from_letter = {"s": 0, "p": 1, "d": 2}
    component_order = {
        0: [""],
        1: ["y", "z", "x"],
        2: ["xy", "yz", "z^2", "xz", "x2-y2"],
    }
    shell_names: dict[tuple[int, int], list[str]] = {}
    pyscf_lookup = {}
    for index, (atom, _, shell_name, component) in enumerate(pyscf_labels):
        angular = angular_from_letter[shell_name[-1]]
        key = (atom, angular)
        names = shell_names.setdefault(key, [])
        if shell_name not in names:
            names.append(shell_name)
        radial = names.index(shell_name)
        pyscf_lookup[(atom, angular, radial, component)] = index

    permutation = []
    descriptors = []
    max_angular = max(shell["angular"] for shell in veloxchem_shells)
    for angular in range(max_angular + 1):
        shells = [s for s in veloxchem_shells if s["angular"] == angular]
        for component in component_order[angular]:
            for shell in shells:
                descriptor = (
                    shell["atom"], angular, shell["radial"], component
                )
                permutation.append(pyscf_lookup[descriptor])
                descriptors.append(descriptor)

    transformed = overlap_pyscf[np.ix_(permutation, permutation)]
    difference = overlap_veloxchem - transformed
    result = {
        "nao": len(permutation),
        "veloxchem_to_pyscf_permutation": permutation,
        "veloxchem_descriptors": descriptors,
        "max_abs_overlap_difference": float(np.max(np.abs(difference))),
        "rms_overlap_difference": float(np.sqrt(np.mean(difference**2))),
    }
    with (output / "comparison.json").open("w", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    if result["max_abs_overlap_difference"] > 1.0e-10:
        raise RuntimeError("AO permutation/phase failed overlap validation")


def main() -> None:
    args = parse_args()
    if args.backend == "compare":
        compare_exports(args.output)
        return
    overlap, metadata = (
        export_pyscf() if args.backend == "pyscf" else export_veloxchem()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"overlap_{args.backend}.npy", overlap)
    with (args.output / f"metadata_{args.backend}.json").open(
        "w", encoding="utf8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(args.backend, "overlap shape", overlap.shape)
    print(args.backend, "max asymmetry", np.max(np.abs(overlap - overlap.T)))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
