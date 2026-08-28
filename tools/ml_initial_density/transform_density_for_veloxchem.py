#!/usr/bin/env python3
"""Transform spin-summed PySCF densities to VeloxChem restricted AO order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyscf-densities", type=Path, required=True)
    parser.add_argument("--ao-comparison", type=Path, required=True)
    parser.add_argument("--veloxchem-overlap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.ao_comparison.open(encoding="utf8") as handle:
        comparison = json.load(handle)
    permutation = np.asarray(
        comparison["veloxchem_to_pyscf_permutation"], dtype=np.int64
    )
    overlap = np.load(args.veloxchem_overlap)
    source = np.load(args.pyscf_densities)

    transformed = {}
    electron_counts = {}
    for name in ("dm_sad", "dm_ri_reference", "dm_pet"):
        # PySCF RKS stores D_alpha + D_beta. VeloxChem restricted SCF stores
        # one spin block, D_alpha == D_beta, hence the factor of one half.
        density = 0.5 * source[name][np.ix_(permutation, permutation)]
        transformed[name] = density
        electron_counts[name] = float(np.einsum("ij,ji", density, overlap))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **transformed, overlap=overlap)
    result = {
        "source": str(args.pyscf_densities.resolve()),
        "output": str(args.output.resolve()),
        "restricted_alpha_electron_counts": electron_counts,
        "expected_alpha_electrons": 19.0,
    }
    print(json.dumps(result, indent=2))
    if abs(electron_counts["dm_pet"] - 19.0) > 1.0e-8:
        raise RuntimeError("transformed PET density has the wrong electron count")


if __name__ == "__main__":
    main()
