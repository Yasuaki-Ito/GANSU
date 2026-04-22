"""
GANSU Python interface — ctypes wrapper around libgansu.so C API.

Usage:
    import gansu

    gansu.init()  # auto-detect GPU

    mol = gansu.Molecule("H2O.xyz", basis="cc-pvdz.gbs")
    result = mol.run(method="RHF", post_hf="ccsd")
    print(result.total_energy)
    print(result.post_hf_energy)

    gansu.finalize()

Or with context manager:
    with gansu.session():
        mol = gansu.Molecule("H2O.xyz", basis="cc-pvdz.gbs")
        result = mol.run()
        print(result.total_energy)
"""

import ctypes
import os
import sys
import contextlib
import numpy as np
from gansu._basis import resolve_basis, resolve_auxiliary_basis, list_basis_sets

# ---------------------------------------------------------------------------
#  Library loading
# ---------------------------------------------------------------------------

def _find_lib():
    """Find libgansu.so in standard locations."""
    _pkg = os.path.dirname(__file__)
    _root = os.path.join(_pkg, "..", "..")
    candidates = [
        os.environ.get("GANSU_LIB", ""),
        # Development: build/libgansu.so
        os.path.join(_root, "build", "libgansu.so"),
        os.path.join(_root, "build", "libgansu.dylib"),
        # Binary distribution: lib/libgansu.so
        os.path.join(_root, "lib", "libgansu.so"),
        os.path.join(_root, "lib", "libgansu.dylib"),
        # Bundled inside package: gansu/lib/
        os.path.join(_pkg, "lib", "libgansu.so"),
        os.path.join(_pkg, "lib", "libgansu.dylib"),
        "libgansu.so",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

_lib = None

def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    path = _find_lib()
    if path is None:
        raise RuntimeError(
            "Cannot find libgansu.so. Set GANSU_LIB env var or build with "
            "'cmake .. && make gansu_shared'")
    _lib = ctypes.CDLL(path)
    _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib):
    """Declare C function signatures for type safety."""
    c_handle = ctypes.c_void_p
    c_str = ctypes.c_char_p
    c_int = ctypes.c_int
    c_double = ctypes.c_double
    c_double_p = ctypes.POINTER(ctypes.c_double)

    lib.gansu_init.argtypes = [c_int]
    lib.gansu_init.restype = None

    lib.gansu_finalize.argtypes = []
    lib.gansu_finalize.restype = None

    lib.gansu_create.argtypes = []
    lib.gansu_create.restype = c_handle

    lib.gansu_destroy.argtypes = [c_handle]
    lib.gansu_destroy.restype = None

    lib.gansu_set.argtypes = [c_handle, c_str, c_str]
    lib.gansu_set.restype = c_int

    lib.gansu_set_xyz.argtypes = [c_handle, c_str]
    lib.gansu_set_xyz.restype = c_int

    lib.gansu_set_basis.argtypes = [c_handle, c_str]
    lib.gansu_set_basis.restype = c_int

    lib.gansu_set_method.argtypes = [c_handle, c_str]
    lib.gansu_set_method.restype = c_int

    lib.gansu_set_post_hf.argtypes = [c_handle, c_str]
    lib.gansu_set_post_hf.restype = c_int

    lib.gansu_run.argtypes = [c_handle]
    lib.gansu_run.restype = c_int

    lib.gansu_get_total_energy.argtypes = [c_handle]
    lib.gansu_get_total_energy.restype = c_double

    lib.gansu_get_post_hf_energy.argtypes = [c_handle]
    lib.gansu_get_post_hf_energy.restype = c_double

    lib.gansu_get_nuclear_repulsion_energy.argtypes = [c_handle]
    lib.gansu_get_nuclear_repulsion_energy.restype = c_double

    lib.gansu_get_num_basis.argtypes = [c_handle]
    lib.gansu_get_num_basis.restype = c_int

    lib.gansu_get_num_electrons.argtypes = [c_handle]
    lib.gansu_get_num_electrons.restype = c_int

    lib.gansu_get_num_atoms.argtypes = [c_handle]
    lib.gansu_get_num_atoms.restype = c_int

    lib.gansu_get_orbital_energies.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_orbital_energies.restype = c_int

    lib.gansu_get_mo_coefficients.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_mo_coefficients.restype = c_int

    lib.gansu_get_ccsd_1rdm_mo.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_ccsd_1rdm_mo.restype = c_int

    lib.gansu_get_excited_state_report.argtypes = [c_handle]
    lib.gansu_get_excited_state_report.restype = c_str

    lib.gansu_get_atomic_number.argtypes = [c_handle, c_int]
    lib.gansu_get_atomic_number.restype = c_int

    lib.gansu_get_atom_coords.argtypes = [c_handle, c_int, ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double)]
    lib.gansu_get_atom_coords.restype = c_int

    lib.gansu_get_mulliken_charges.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_mulliken_charges.restype = c_int

    lib.gansu_get_mayer_bond_order.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_mayer_bond_order.restype = c_int

    lib.gansu_get_wiberg_bond_order.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_wiberg_bond_order.restype = c_int

    lib.gansu_get_density_matrix.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_density_matrix.restype = c_int

    lib.gansu_get_overlap_matrix.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_overlap_matrix.restype = c_int

    lib.gansu_set_initial_density.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_set_initial_density.restype = c_int

    # Progress callback: void(const char*, int, int, const double*, void*)
    global PROGRESS_FUNC_TYPE
    PROGRESS_FUNC_TYPE = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, c_int, c_int, c_double_p, ctypes.c_void_p)
    lib.gansu_set_progress_callback.argtypes = [c_handle, PROGRESS_FUNC_TYPE, ctypes.c_void_p]
    lib.gansu_set_progress_callback.restype = None


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def init(force_cpu=False):
    """Initialize GANSU runtime. Call once."""
    _get_lib().gansu_init(1 if force_cpu else 0)

def finalize():
    """Finalize GANSU runtime."""
    _get_lib().gansu_finalize()

@contextlib.contextmanager
def session(force_cpu=False):
    """Context manager: init on enter, finalize on exit."""
    init(force_cpu=force_cpu)
    try:
        yield
    finally:
        finalize()


class Result:
    """Holds results from a GANSU calculation."""

    def __init__(self, handle, owner=None):
        self._h = handle
        self._lib = _get_lib()
        self._owner = owner  # prevent GC of Molecule while Result is alive

    @property
    def total_energy(self):
        """HF total energy (Hartree)."""
        return self._lib.gansu_get_total_energy(self._h)

    @property
    def post_hf_energy(self):
        """Post-HF correlation energy (Hartree). 0 if no post-HF."""
        return self._lib.gansu_get_post_hf_energy(self._h)

    @property
    def correlation_energy(self):
        """Alias for post_hf_energy."""
        return self.post_hf_energy

    @property
    def nuclear_repulsion_energy(self):
        return self._lib.gansu_get_nuclear_repulsion_energy(self._h)

    @property
    def num_basis(self):
        return self._lib.gansu_get_num_basis(self._h)

    @property
    def num_electrons(self):
        return self._lib.gansu_get_num_electrons(self._h)

    @property
    def num_atoms(self):
        return self._lib.gansu_get_num_atoms(self._h)

    @property
    def orbital_energies(self):
        """Return orbital energies as numpy array."""
        n = self.num_basis
        buf = (ctypes.c_double * n)()
        ret = self._lib.gansu_get_orbital_energies(self._h, buf, n)
        if ret < 0:
            raise RuntimeError("Failed to get orbital energies")
        return np.array(buf[:n])

    @property
    def mo_coefficients(self):
        """Return MO coefficient matrix (nao x nao) as numpy array."""
        n = self.num_basis
        n2 = n * n
        buf = (ctypes.c_double * n2)()
        ret = self._lib.gansu_get_mo_coefficients(self._h, buf, n2)
        if ret < 0:
            raise RuntimeError("Failed to get MO coefficients")
        return np.array(buf[:n2]).reshape(n, n)

    @property
    def ccsd_1rdm_mo(self):
        """Return CCSD 1-RDM in MO basis (nao x nao). Only valid after ccsd_density."""
        n = self.num_basis
        n2 = n * n
        buf = (ctypes.c_double * n2)()
        ret = self._lib.gansu_get_ccsd_1rdm_mo(self._h, buf, n2)
        if ret < 0:
            raise RuntimeError("CCSD 1-RDM not available (run with post_hf='ccsd_density')")
        return np.array(buf[:n2]).reshape(n, n)

    @property
    def excited_state_report(self):
        """Formatted excited-state summary string."""
        s = self._lib.gansu_get_excited_state_report(self._h)
        return s.decode("utf-8") if s else ""

    @property
    def atoms(self):
        """List of (atomic_number, x, y, z) tuples (coordinates in Bohr)."""
        n = self.num_atoms
        result = []
        x, y, z = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        for i in range(n):
            Z = self._lib.gansu_get_atomic_number(self._h, i)
            self._lib.gansu_get_atom_coords(self._h, i,
                ctypes.byref(x), ctypes.byref(y), ctypes.byref(z))
            result.append((Z, x.value, y.value, z.value))
        return result

    @property
    def mulliken_charges(self):
        """Mulliken atomic charges as numpy array."""
        n = self.num_atoms
        buf = (ctypes.c_double * n)()
        ret = self._lib.gansu_get_mulliken_charges(self._h, buf, n)
        if ret < 0:
            raise RuntimeError("Failed to compute Mulliken charges")
        return np.array(buf[:n])

    @property
    def mayer_bond_order(self):
        """Mayer bond order matrix (natom x natom)."""
        n = self.num_atoms
        n2 = n * n
        buf = (ctypes.c_double * n2)()
        ret = self._lib.gansu_get_mayer_bond_order(self._h, buf, n2)
        if ret < 0:
            raise RuntimeError("Failed to compute Mayer bond order")
        return np.array(buf[:n2]).reshape(n, n)

    @property
    def wiberg_bond_order(self):
        """Wiberg bond order matrix (natom x natom)."""
        n = self.num_atoms
        n2 = n * n
        buf = (ctypes.c_double * n2)()
        ret = self._lib.gansu_get_wiberg_bond_order(self._h, buf, n2)
        if ret < 0:
            raise RuntimeError("Failed to compute Wiberg bond order")
        return np.array(buf[:n2]).reshape(n, n)

    @property
    def density_matrix(self):
        """Density matrix in AO basis (nao x nao)."""
        n = self.num_basis
        n2 = n * n
        buf = (ctypes.c_double * n2)()
        ret = self._lib.gansu_get_density_matrix(self._h, buf, n2)
        if ret < 0:
            raise RuntimeError("Failed to get density matrix")
        return np.array(buf[:n2]).reshape(n, n)

    @property
    def overlap_matrix(self):
        """Overlap matrix (nao x nao)."""
        n = self.num_basis
        n2 = n * n
        buf = (ctypes.c_double * n2)()
        ret = self._lib.gansu_get_overlap_matrix(self._h, buf, n2)
        if ret < 0:
            raise RuntimeError("Failed to get overlap matrix")
        return np.array(buf[:n2]).reshape(n, n)


class Molecule:
    """GANSU calculation setup."""

    def __init__(self, xyz_path, basis="sto-3g", **kwargs):
        """
        Args:
            xyz_path: Path to xyz file.
            basis: Basis set name ("cc-pvdz") or full path to .gbs file.
            **kwargs: Additional parameters (method, post_hf_method, eri_method, etc.)
        """
        self._lib = _get_lib()
        self._h = self._lib.gansu_create()
        self._lib.gansu_set_xyz(self._h, os.path.abspath(xyz_path).encode())
        self._lib.gansu_set_basis(self._h, resolve_basis(basis).encode())
        for k, v in kwargs.items():
            self._lib.gansu_set(self._h, k.encode(), str(v).encode())

    def __del__(self):
        if hasattr(self, '_h') and self._h:
            self._lib.gansu_destroy(self._h)
            self._h = None

    def set_initial_density(self, density):
        """Set initial density matrix for next run (PES density reuse).
        Args: density — numpy array (nao x nao) or None to clear."""
        if density is None:
            self._lib.gansu_set_initial_density(self._h, None, 0)
        else:
            flat = np.ascontiguousarray(density.ravel(), dtype=np.float64)
            self._lib.gansu_set_initial_density(
                self._h, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(flat))

    def run(self, method="RHF", post_hf="none", quiet=True, on_progress=None, **kwargs):
        """Run the calculation.

        Args:
            method: HF method (RHF, UHF, ROHF).
            post_hf: Post-HF method (none, mp2, mp3, ccsd, ccsd_t, fci, ...).
            quiet: If True (default), suppress GANSU stdout output.
            on_progress: Optional callback fn(stage: str, iter: int, values: list[float]).
                         Called during SCF/CCSD/Davidson iterations with progress data.
                         Stage is "scf", "ccsd", "ccsd_lambda", "davidson", etc.
            **kwargs: Extra parameters to set before running.

        Returns:
            Result object with computed properties.
        """
        self._lib.gansu_set(self._h, b"quiet", b"true" if quiet else b"false")
        self._lib.gansu_set_method(self._h, method.encode())
        self._lib.gansu_set_post_hf(self._h, post_hf.encode())
        for k, v in kwargs.items():
            self._lib.gansu_set(self._h, k.encode(), str(v).encode())

        # Set up progress callback
        self._progress_ref = None  # prevent GC
        if on_progress is not None:
            def _c_callback(stage, iter_num, n_values, values_ptr, _user):
                stage_str = stage.decode("utf-8") if stage else ""
                vals = [values_ptr[i] for i in range(n_values)]
                on_progress(stage_str, iter_num, vals)
            self._progress_ref = PROGRESS_FUNC_TYPE(_c_callback)
            self._lib.gansu_set_progress_callback(self._h, self._progress_ref, None)
        else:
            self._lib.gansu_set_progress_callback(self._h, PROGRESS_FUNC_TYPE(), None)

        ret = self._lib.gansu_run(self._h)
        if ret != 0:
            raise RuntimeError("GANSU calculation failed")
        return Result(self._h, owner=self)
