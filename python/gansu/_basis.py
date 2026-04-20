"""
Basis set path resolution for GANSU.

Resolves short names like "cc-pvdz" to full paths of bundled .gbs files.
Also handles SAD cache and auxiliary basis lookup.
"""

import os

# Data directory: python/gansu/data/ (relative to this file)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_BASIS_DIR = os.path.join(_DATA_DIR, "basis")
_AUX_BASIS_DIR = os.path.join(_DATA_DIR, "auxiliary_basis")
_SAD_DIR = os.path.join(_DATA_DIR, "sad")

# Fallback: project source tree (development mode)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEV_BASIS_DIR = os.path.join(_PROJECT_ROOT, "basis")
_DEV_AUX_BASIS_DIR = os.path.join(_PROJECT_ROOT, "auxiliary_basis")


def resolve_basis(name_or_path):
    """Resolve a basis set name or path to an absolute .gbs file path.

    Args:
        name_or_path: One of:
            - Short name: "sto-3g", "cc-pvdz", "6-31g*", etc.
            - Filename: "sto-3g.gbs"
            - Full path: "/path/to/basis.gbs"

    Returns:
        Absolute path to .gbs file.

    Raises:
        FileNotFoundError: If basis set cannot be found.
    """
    # Already a full/relative path that exists
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    # Normalize: add .gbs if missing
    name = name_or_path
    if not name.endswith(".gbs"):
        name = name + ".gbs"

    # Search in bundled data (pip install)
    bundled = os.path.join(_BASIS_DIR, name)
    if os.path.isfile(bundled):
        return bundled

    # Search in project source tree (development mode)
    dev = os.path.join(_DEV_BASIS_DIR, name)
    if os.path.isfile(dev):
        return dev

    raise FileNotFoundError(
        f"Basis set '{name_or_path}' not found. "
        f"Searched: {_BASIS_DIR}, {_DEV_BASIS_DIR}. "
        f"Use full path or install gansu with bundled basis sets."
    )


def resolve_auxiliary_basis(name_or_path):
    """Resolve auxiliary basis set name to path (for RI calculations)."""
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    name = name_or_path
    if not name.endswith(".gbs"):
        name = name + ".gbs"

    for d in [_AUX_BASIS_DIR, _DEV_AUX_BASIS_DIR]:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(f"Auxiliary basis '{name_or_path}' not found.")


def list_basis_sets():
    """List available bundled basis sets."""
    sets = set()
    for d in [_BASIS_DIR, _DEV_BASIS_DIR]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".gbs"):
                    sets.add(f[:-4])  # remove .gbs
    return sorted(sets)
