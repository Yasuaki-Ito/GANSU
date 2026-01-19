# 
# GANSU: GPU Accelerated Numerical Simulation Utility
# 
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# All rights reserved.
# 
# This software is licensed under the BSD 3-Clause License.
# You may obtain a copy of the license in the LICENSE file
# located in the root directory of this source tree or at:
# https://opensource.org/licenses/BSD-3-Clause
# 
# SPDX-License-Identifier: BSD-3-Clause
# 

# This script generates precomputed SAD initial guess density matrices
# for atomic RHF/ROHF calculations using specified basis sets.
# Each output .sad file contains alpha and beta spin density matrices
# for each element, stored in plain text format (row-major order).

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import tempfile
from typing import Tuple
import re

# Periodic table to map element symbols to atomic numbers
ELEMENT_TO_Z = {
    'H': 1,  'He': 2, 'Li': 3,  'Be': 4,  'B': 5,   'C': 6,  'N': 7,  'O': 8,  'F': 9,  'Ne': 10,
    'Na': 11,'Mg': 12,'Al': 13,'Si': 14,'P': 15,  'S': 16, 'Cl':17, 'Ar':18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29,'Zn': 30,'Ga': 31,'Ge': 32,'As': 33,'Se': 34,'Br': 35,'Kr': 36,
    'Rb': 37,'Sr': 38,'Y': 39, 'Zr': 40,'Nb': 41,'Mo': 42,'Tc': 43,'Ru': 44,'Rh': 45,'Pd': 46,
    'Ag': 47,'Cd': 48,'In': 49,'Sn': 50,'Sb': 51,'Te': 52,'I': 53, 'Xe': 54,
    'Cs': 55,'Ba': 56,'La': 57,'Ce': 58,'Pr': 59,'Nd': 60,'Pm': 61,'Sm': 62,'Eu': 63,'Gd': 64,
    'Tb': 65,'Dy': 66,'Ho': 67,'Er': 68,'Tm': 69,'Yb': 70,'Lu': 71,
    'Hf': 72,'Ta': 73,'W': 74, 'Re': 75,'Os': 76,'Ir': 77,'Pt': 78,'Au': 79,'Hg': 80,
    'Tl': 81,'Pb': 82,'Bi': 83,'Po': 84,'At': 85,'Rn': 86,
    'Fr': 87,'Ra': 88,'Ac': 89,'Th': 90,'Pa': 91,'U': 92, 'Np': 93,'Pu': 94,'Am': 95,'Cm': 96,
    'Bk': 97,'Cf': 98,'Es': 99,'Fm':100,'Md':101,'No':102,'Lr':103,
    'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,'Nh':113,
    'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og':118
}

def extract_elements_from_basis_file(basis_path: str) -> list:
    elements = set()
    pattern = re.compile(r"^([A-Z][a-z]?)\s+0\s*$")
    with open(basis_path) as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                elements.add(match.group(1))
    return elements

def compute_sad_density_for(element: str, basis_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    z = ELEMENT_TO_Z[element]
    if z % 2 == 0:
        method = "rhf"
    else:
        method = "rohf"

    # Create temporary XYZ file for the atom at (0.0, 0.0, 0.0)
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".xyz", delete=False) as xyz_file:
        xyz_path = xyz_file.name
        xyz_file.write("1\n\n")
        xyz_file.write(f"{element} 0.0 0.0 0.0\n")

    try:
        print(f"Computing SAD density for {element} using {method} method...")
        # Invoke external program (HF_main) with the XYZ and basis file
        result = subprocess.run(["./HF_main", "-x", xyz_path, "-g", basis_path, "--method", method, "--export_sad_cache"], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"HF_main failed for {element}: {result.stderr.decode()}")

        temp_path = Path("temp_sad_cache.dat")
        if not temp_path.exists():
            raise FileNotFoundError("temp_sad_cache.dat not generated")

        with temp_path.open() as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("temp_sad_cache.dat is empty")

        num_basis = int(lines[0].strip())
        alpha_lines = lines[1:1 + num_basis]
        beta_lines = lines[1 + num_basis:1 + 2 * num_basis]

        alpha = np.loadtxt(alpha_lines)
        beta = np.loadtxt(beta_lines)

        if alpha.ndim == 0:
            alpha = np.array([[alpha]])
        if beta.ndim == 0:
            beta = np.array([[beta]])

        return alpha, beta, num_basis
    finally:
        Path(xyz_path).unlink(missing_ok=True)


def cache_has_element(sad_path: Path, element: str) -> bool:
    if not sad_path.exists():
        return False
    with sad_path.open() as f:
        for line in f:
            if line.startswith(element + " "):
                return True
    return False

def append_density_to_file(sad_path: Path, element: str, alpha: np.ndarray, beta: np.ndarray, num_basis: int):
    with sad_path.open("a") as f:
        f.write(f"{element} {num_basis}\n")
        np.savetxt(f, alpha, fmt="%.10e")
        np.savetxt(f, beta, fmt="%.10e")

def generate_sad_cache_for(basis_path: str):
    if not Path(basis_path).exists():
        raise FileNotFoundError(f"Basis file not found: {basis_path}")

    sad_path = Path(basis_path).with_suffix(".sad")
    elements = extract_elements_from_basis_file(basis_path)

#    for element in elements:
#        if not cache_has_element(sad_path, element):
#            alpha, beta, n = compute_sad_density_for(element, basis_path)
#            append_density_to_file(sad_path, element, alpha, beta, n)
    for element in elements:
        if cache_has_element(sad_path, element):
            continue

        try:
            alpha, beta, n = compute_sad_density_for(element, basis_path)
            append_density_to_file(sad_path, element, alpha, beta, n)
        except Exception as e:
            print(f"[Warning] Skipped {element} due to error: {e}")
        

    print(f"Generated: {sad_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_sad_cache.py file1.gbs [file2.gbs ...]")
        return

    for file in sys.argv[1:]:
        try:
            print(f"Generating SAD cache for {file}")
            generate_sad_cache_for(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
