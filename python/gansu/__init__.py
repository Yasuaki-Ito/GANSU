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
        spec = importlib.util.find_spec(pkg)
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


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    _preload_cuda_libs()
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
