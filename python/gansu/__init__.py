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
import hashlib
import json
import os
import platform as _platform_module
import sys
import contextlib
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np
from gansu._basis import resolve_basis, resolve_auxiliary_basis, list_basis_sets

# ---------------------------------------------------------------------------
#  Library loading
# ---------------------------------------------------------------------------

# GitHub Releases URL pattern for the native shared library. The CI workflow
# uploads `libgansu-<version>-<platform>.so` (and a matching .sha256) as
# release assets, and the version + checksum are baked into the wheel via
# _native_meta.json so a thin wheel can fetch the correct binary at runtime.
GITHUB_RELEASE_URL_TEMPLATE = (
    "https://github.com/Yasuaki-Ito/GANSU/releases/download/"
    "v{version}/libgansu-{version}-{platform}.so"
)


def _platform_id():
    """Return the platform identifier embedded in the .so asset filename."""
    if sys.platform.startswith("linux"):
        return f"linux-{_platform_module.machine().lower()}"
    raise RuntimeError(
        f"GANSU currently only ships binaries for Linux x86_64; "
        f"detected platform {sys.platform!r}.")


def _native_meta():
    """Load native-library metadata (version + sha256) shipped with the wheel."""
    meta_path = Path(__file__).parent / "_native_meta.json"
    if not meta_path.is_file():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _user_cache_dir(version):
    """Per-version directory for the cached native library."""
    base = os.environ.get("GANSU_CACHE")
    if base:
        return Path(base) / version
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(xdg) / "gansu" / version


def _sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url, dest):
    """Stream `url` to `dest` (atomic via .part), printing progress to stderr."""
    print(f"==> Downloading libgansu.so from {url}", file=sys.stderr)

    def hook(blocknum, blocksize, totalsize):
        if totalsize <= 0:
            return
        done = min(totalsize, blocknum * blocksize)
        pct = done * 100 // totalsize
        print(f"\r    {done >> 20} / {totalsize >> 20} MB ({pct}%) ",
              end="", file=sys.stderr, flush=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, str(tmp), reporthook=hook)
    print("", file=sys.stderr)
    tmp.replace(dest)


def _find_or_fetch_lib():
    """Resolve libgansu.so via env override / wheel-bundled / dev tree /
    user cache / auto-download from GitHub Releases, in that priority order.

    This makes the package work in three modes without any user action:
    * `pip install gansu` → thin wheel, downloads the .so on first use.
    * Direct-URL or offline install with the .so already on disk →
      pointed at via `GANSU_LIB` or placed under `~/.cache/gansu/<ver>/`.
    * In-tree development → picks up `build/libgansu.so` from a local cmake.
    """
    pkg_dir = Path(__file__).parent
    repo_root = pkg_dir.parent.parent

    # 1. Explicit override
    env = os.environ.get("GANSU_LIB", "")
    if env and Path(env).is_file():
        return env

    # 2. Bundled with this wheel (offline / heavy-wheel install)
    for name in ("libgansu.so", "libgansu.dylib"):
        p = pkg_dir / "lib" / name
        if p.is_file():
            return str(p)

    # 3. In-tree development build
    for p in (repo_root / "build" / "libgansu.so",
              repo_root / "build" / "libgansu.dylib",
              repo_root / "lib" / "libgansu.so"):
        if p.is_file():
            return str(p)

    # 4. User cache (already downloaded for this version)
    meta = _native_meta()
    if meta is None:
        raise RuntimeError(
            "libgansu.so not found and no _native_meta.json shipped in this "
            "wheel — cannot auto-download. Set GANSU_LIB to a local copy of "
            "the library or reinstall the wheel.")

    version = meta["version"]
    expected_sha = meta["sha256"]
    cached = _user_cache_dir(version) / "libgansu.so"
    if cached.is_file():
        if _sha256_of_file(cached) == expected_sha:
            return str(cached)
        cached.unlink()  # corrupted, fall through to re-download

    # 5. Auto-download from GitHub Releases
    cached.parent.mkdir(parents=True, exist_ok=True)
    url = GITHUB_RELEASE_URL_TEMPLATE.format(
        version=version, platform=_platform_id())
    try:
        _download_with_progress(url, cached)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download libgansu.so from {url}: {e}\n"
            f"Workaround: download the file manually and either set "
            f"GANSU_LIB to its path or place it at {cached}.") from e

    actual_sha = _sha256_of_file(cached)
    if actual_sha != expected_sha:
        cached.unlink()
        raise RuntimeError(
            f"Downloaded libgansu.so has SHA256 {actual_sha}, expected "
            f"{expected_sha}. The release asset may have been tampered with "
            f"or replaced. Aborting.")
    return str(cached)


_lib = None
_HAS_ENERGY_EVAL_API = False


def _preload_cuda_libs():
    """Preload CUDA shared libraries from nvidia-*-cu12 wheels, if installed.

    When GANSU is installed via pip wheel, libcublas / libcusolver / libnccl
    are provided by the `nvidia-*-cu12` packages under
    `site-packages/nvidia/<lib>/lib/`. These directories are not on the dynamic
    linker's search path, so we dlopen them with RTLD_GLOBAL before loading
    libgansu.so. If the packages are not installed (development mode with a
    system-wide CUDA toolkit), this is a silent no-op.
    """
    import importlib.util

    # Order matters: cusparse before cusolver (cusolver depends on cusparse).
    nvidia_libs = [
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.cublas", "libcublasLt.so.12"),
        ("nvidia.cublas", "libcublas.so.12"),
        ("nvidia.cusparse", "libcusparse.so.12"),
        ("nvidia.cusolver", "libcusolver.so.11"),
        ("nvidia.nccl", "libnccl.so.2"),
    ]
    for pkg, soname in nvidia_libs:
        # find_spec raises ModuleNotFoundError (rather than returning None)
        # when the parent package "nvidia" itself is not installed, e.g. in
        # development environments using a system-wide CUDA toolkit.
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            return
        if spec is None or spec.submodule_search_locations is None:
            continue
        for base in spec.submodule_search_locations:
            so_path = os.path.join(base, "lib", soname)
            if os.path.isfile(so_path):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def _preload_bundled_runtime():
    """Preload the Fortran runtime shipped in the wheel for the statically
    embedded OpenBLAS.

    The wheel-built libgansu.so links OpenBLAS statically but keeps a dynamic
    dependency on libgfortran.so.5 / libquadmath.so.0 (their static archives are
    not reliably position-independent). We dlopen them RTLD_GLOBAL before loading
    libgansu.so so its DT_NEEDED entries resolve against the already-loaded
    objects — no system OpenBLAS / Fortran runtime required. Order matters:
    libgfortran depends on libquadmath, so preload libquadmath first. Silent
    no-op for development builds (system BLAS/LAPACK) where these aren't shipped.
    """
    libdir = Path(__file__).parent / "lib"
    for soname in ("libquadmath.so.0", "libgfortran.so.5"):
        so_path = libdir / soname
        if so_path.is_file():
            try:
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    _preload_cuda_libs()
    _preload_bundled_runtime()
    path = _find_or_fetch_lib()
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

    lib.gansu_get_num_frozen_core.argtypes = [c_handle]
    lib.gansu_get_num_frozen_core.restype = c_int

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

    lib.gansu_get_excited_states.argtypes = [c_handle, c_double_p, c_double_p, c_int]
    lib.gansu_get_excited_states.restype = c_int

    lib.gansu_get_energy_gradient.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_energy_gradient.restype = c_int

    lib.gansu_get_hessian.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_hessian.restype = c_int

    lib.gansu_get_frequencies.argtypes = [c_handle, c_double_p, c_int]
    lib.gansu_get_frequencies.restype = c_int

    lib.gansu_get_dipole.argtypes = [c_handle, c_double_p]
    lib.gansu_get_dipole.restype = c_int

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

    # SCF-free energy-functional evaluation (FMQA interface).
    # Tolerate an older libgansu.so that predates these symbols — import
    # still works, and the Molecule methods raise a clear error when used.
    global _HAS_ENERGY_EVAL_API
    try:
        lib.gansu_prepare.argtypes = [c_handle]
        lib.gansu_prepare.restype = c_int

        lib.gansu_energy_from_mo.argtypes = [c_handle, c_double_p, c_int, c_int, c_double_p]
        lib.gansu_energy_from_mo.restype = c_int

        lib.gansu_energy_from_density.argtypes = [c_handle, c_double_p, c_int, c_double_p]
        lib.gansu_energy_from_density.restype = c_int

        lib.gansu_energy_from_mo_batch.argtypes = [c_handle, c_double_p, c_int, c_int, c_int, c_double_p]
        lib.gansu_energy_from_mo_batch.restype = c_int

        lib.gansu_get_hcore.argtypes = [c_handle, c_double_p, c_int]
        lib.gansu_get_hcore.restype = c_int

        lib.gansu_get_eri.argtypes = [c_handle, c_double_p, c_int]
        lib.gansu_get_eri.restype = c_int
        _HAS_ENERGY_EVAL_API = True
    except AttributeError:
        _HAS_ENERGY_EVAL_API = False

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
    def num_frozen_core(self):
        return self._lib.gansu_get_num_frozen_core(self._h)

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
    def excited_states(self):
        """Excited-state data after a CIS/ADC/EOM/STEOM run, as a dict with
        'energies' (Hartree) and 'oscillator_strengths' numpy arrays."""
        n_max = 256
        e = (ctypes.c_double * n_max)()
        f = (ctypes.c_double * n_max)()
        n = self._lib.gansu_get_excited_states(self._h, e, f, n_max)
        if n < 0:
            raise RuntimeError("Excited-state data not available (run an excited-state method)")
        return {"energies": np.array(e[:n]),
                "oscillator_strengths": np.array(f[:n])}

    @property
    def energy_gradient(self):
        """Analytic energy gradient (nuclear forces) as an (natoms, 3) numpy
        array in Hartree/Bohr. Computed on demand from the converged wavefunction."""
        n = 3 * self.num_atoms
        buf = (ctypes.c_double * n)()
        ret = self._lib.gansu_get_energy_gradient(self._h, buf, n)
        if ret == -2:
            raise RuntimeError("Energy gradient not available for this method")
        if ret < 0:
            raise RuntimeError("Failed to compute energy gradient")
        return np.array(buf[:ret]).reshape(-1, 3)

    @property
    def hessian(self):
        """Analytic Hessian d²E/dR_i dR_j (Hartree/Bohr²) as a (3N, 3N) numpy
        array. Computed on demand."""
        ndim = 3 * self.num_atoms
        n = ndim * ndim
        buf = (ctypes.c_double * n)()
        ret = self._lib.gansu_get_hessian(self._h, buf, n)
        if ret == -2:
            raise RuntimeError("Hessian not available for this method")
        if ret < 0:
            raise RuntimeError("Failed to compute Hessian")
        return np.array(buf[:ret]).reshape(ndim, ndim)

    @property
    def frequencies(self):
        """Harmonic vibrational frequencies (cm⁻¹) as a numpy array (imaginary
        modes negative). Computed on demand (Hessian + mass-weighting + TR projection)."""
        n = 3 * self.num_atoms
        buf = (ctypes.c_double * n)()
        ret = self._lib.gansu_get_frequencies(self._h, buf, n)
        if ret == -2:
            raise RuntimeError("Frequencies not available for this method")
        if ret < 0:
            raise RuntimeError("Failed to compute frequencies")
        return np.array(buf[:ret])

    @property
    def dipole(self):
        """Ground-state SCF dipole moment (atomic units, e·Bohr) as a length-3
        numpy array [mu_x, mu_y, mu_z]. Multiply by 2.5417464157 for Debye. RHF only."""
        xyz = (ctypes.c_double * 3)()
        ret = self._lib.gansu_get_dipole(self._h, xyz)
        if ret == -3:
            raise RuntimeError("Dipole moment is only available for closed-shell RHF")
        if ret != 0:
            raise RuntimeError("Failed to compute dipole moment")
        return np.array(xyz[:3])

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
        # Quiet by default so SCF-free evaluation (energy_of etc., which may
        # lazily build integrals without run()) doesn't spam stdout.
        # run(quiet=...) still controls verbosity for full calculations.
        self._lib.gansu_set(self._h, b"quiet", b"true")
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

    # -- SCF-free energy-functional evaluation (FMQA interface) --

    def _dp(self, arr):
        """numpy array -> POINTER(c_double)."""
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def prepare(self):
        """Prepare integrals for SCF-free energy evaluation (no SCF run).
        Idempotent; called implicitly by energy_of / hcore / eri / overlap."""
        self._require_energy_eval_api()
        if self._lib.gansu_prepare(self._h) != 0:
            raise RuntimeError("gansu_prepare failed")

    def _require_energy_eval_api(self):
        if not _HAS_ENERGY_EVAL_API:
            raise RuntimeError(
                "The loaded libgansu.so predates the SCF-free energy API "
                "(gansu_prepare / gansu_energy_from_mo ...). Rebuild the "
                "library (cd build && make gansu_shared) and make sure the "
                "new .so is the one loaded — e.g. set GANSU_LIB to its path, "
                "or remove a stale bundled copy at python/gansu/lib/libgansu.so "
                "(it takes priority over build/libgansu.so).")

    @property
    def num_basis(self):
        """Number of basis functions (prepares integrals lazily)."""
        self.prepare()
        return self._lib.gansu_get_num_basis(self._h)

    @property
    def num_electrons(self):
        self.prepare()
        return self._lib.gansu_get_num_electrons(self._h)

    @property
    def nocc(self):
        """Number of doubly-occupied orbitals (closed-shell)."""
        return self.num_electrons // 2

    @property
    def nuclear_repulsion_energy(self):
        """Nuclear repulsion energy E_nn in Hartree (no SCF required)."""
        self.prepare()
        return self._lib.gansu_get_nuclear_repulsion_energy(self._h)

    def energy_of(self, C_occ=None, P=None):
        """SCF-free single-point RHF energy evaluation.

        E = sum_pq P_pq h_pq + 1/2 sum_pq P_pq G_pq(P) + E_nn,  G = J - K/2.

        Args:
            C_occ: (n, nocc) numpy array of occupied MO coefficients
                   (column j = j-th occupied orbital). P = 2 C C^T is formed.
                   Orthonormality (C^T S C = I) is the caller's responsibility.
            P: (n, n) numpy array, AO density matrix (Tr(PS) = n_electrons).
            Specify exactly one of the two.

        Returns:
            Total energy in Hartree (E_nn included). No SCF is performed;
            integrals are prepared once on the first call and reused.
        """
        self._require_energy_eval_api()
        if (C_occ is None) == (P is None):
            raise ValueError("specify exactly one of C_occ or P")
        out = ctypes.c_double()
        if C_occ is not None:
            C = np.ascontiguousarray(C_occ, dtype=np.float64)
            if C.ndim != 2:
                raise ValueError(f"C_occ must be 2-D (n, nocc), got shape {C.shape}")
            n, nocc = C.shape
            ret = self._lib.gansu_energy_from_mo(
                self._h, self._dp(C), n, nocc, ctypes.byref(out))
        else:
            Pm = np.ascontiguousarray(P, dtype=np.float64)
            if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
                raise ValueError(f"P must be square (n, n), got shape {Pm.shape}")
            ret = self._lib.gansu_energy_from_density(
                self._h, self._dp(Pm), Pm.shape[0], ctypes.byref(out))
        if ret != 0:
            raise RuntimeError(f"energy evaluation failed (status {ret})")
        return out.value

    def energy_of_batch(self, C_occ_batch):
        """Batched SCF-free evaluation: (batch, n, nocc) -> (batch,) energies."""
        self._require_energy_eval_api()
        arr = np.ascontiguousarray(C_occ_batch, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"C_occ_batch must be 3-D (batch, n, nocc), got shape {arr.shape}")
        batch, n, nocc = arr.shape
        out = np.empty(batch, dtype=np.float64)
        ret = self._lib.gansu_energy_from_mo_batch(
            self._h, self._dp(arr), batch, n, nocc, self._dp(out))
        if ret != 0:
            raise RuntimeError(f"batch energy evaluation failed (status {ret})")
        return out

    @property
    def hcore(self):
        """Core Hamiltonian h = T + V_ne as (n, n) numpy array."""
        n = self.num_basis
        buf = np.empty(n * n, dtype=np.float64)
        if self._lib.gansu_get_hcore(self._h, self._dp(buf), n * n) < 0:
            raise RuntimeError("Failed to get core Hamiltonian")
        return buf.reshape(n, n)

    @property
    def eri(self):
        """Full AO ERI tensor (pq|rs), chemists' notation, as (n, n, n, n)
        numpy array. Stored-ERI method only; refused for large n."""
        n = self.num_basis
        need = n ** 4
        buf = np.empty(need, dtype=np.float64)
        ret = self._lib.gansu_get_eri(self._h, self._dp(buf), need)
        if ret == -2:
            raise RuntimeError(
                "ERI tensor unavailable: requires eri_method=stored and n^4 <= INT_MAX")
        if ret < 0:
            raise RuntimeError("Failed to get ERI tensor")
        return buf.reshape(n, n, n, n)

    @property
    def overlap(self):
        """Overlap matrix S as (n, n) numpy array (no SCF required)."""
        n = self.num_basis
        buf = np.empty(n * n, dtype=np.float64)
        if self._lib.gansu_get_overlap_matrix(self._h, self._dp(buf), n * n) < 0:
            raise RuntimeError("Failed to get overlap matrix")
        return buf.reshape(n, n)

    def scf_reference(self, **kwargs):
        """Run SCF and return the converged total energy (Hartree).
        Convenience for FMQA's GansuEnergy.scf_reference()."""
        return self.run(**kwargs).total_energy

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
