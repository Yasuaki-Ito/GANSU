"""Execute GANSU calculations via Python API (libgansu.so) with progress callbacks."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

# ---------------------------------------------------------------------------
#  GANSU Python API import
# ---------------------------------------------------------------------------

_GANSU_PATH = Path(os.environ.get("GANSU_PATH", Path.home() / "GANSU"))
_PYTHON_PATH = _GANSU_PATH / "python"
if _PYTHON_PATH.is_dir() and str(_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PATH))

if "GANSU_LIB" not in os.environ:
    for candidate in [
        _GANSU_PATH / "build" / "libgansu.so",
        _GANSU_PATH / "build" / "libgansu.dylib",
    ]:
        if candidate.exists():
            os.environ["GANSU_LIB"] = str(candidate)
            break

import gansu  # noqa: E402

XYZ_DIR = _GANSU_PATH / "xyz"

_initialized = False

# Element symbols for atomic numbers
_ELEMENTS = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
             "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
             "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
             "Ga", "Ge", "As", "Se", "Br", "Kr"]


def _ensure_init():
    global _initialized
    if not _initialized:
        gansu.init()
        _initialized = True


# ---------------------------------------------------------------------------
#  RunParams (unchanged — frontend compatibility)
# ---------------------------------------------------------------------------

@dataclass
class RunParams:
    xyz_text: str = ""
    xyz_file: str = ""
    xyz_dir: str = "."
    basis: str = "sto-3g"
    method: str = "RHF"
    charge: int = 0
    beta_to_alpha: int = 0
    convergence_method: str = "diis"
    diis_size: int = 8
    diis_include_transform: bool = False
    damping_factor: float = 0.9
    rohf_parameter_name: str = "Roothaan"
    maxiter: int = 100
    convergence_energy_threshold: float = 1e-6
    schwarz_screening_threshold: float = 1e-12
    initial_guess: str = "core"
    post_hf_method: str = "none"
    n_excited_states: int = 5
    spin_type: str = "singlet"
    excited_solver: str = "auto"
    eri_method: str = "stored"
    auxiliary_basis: str = ""
    auxiliary_basis_dir: str = "auxiliary_basis"
    mulliken: bool = False
    mayer: bool = False
    wiberg: bool = False
    export_molden: bool = False
    verbose: bool = False
    timeout: int = 600


# ---------------------------------------------------------------------------
#  XYZ / kwargs helpers
# ---------------------------------------------------------------------------

def _resolve_xyz(params: RunParams) -> tuple[str, str | None]:
    if params.xyz_file:
        base = XYZ_DIR if params.xyz_dir == "." else XYZ_DIR / params.xyz_dir
        return str(base / params.xyz_file), None
    fd, tmp = tempfile.mkstemp(suffix=".xyz", prefix="gansu_")
    with os.fdopen(fd, "w") as f:
        f.write(params.xyz_text)
    return tmp, tmp


def _build_kwargs(params: RunParams) -> dict[str, str]:
    kw: dict[str, str] = {}
    kw["charge"] = str(params.charge)
    if params.beta_to_alpha:
        kw["beta_to_alpha"] = str(params.beta_to_alpha)
    conv_map = {"optimal_damping": "OptimalDamping", "damping": "Damping", "diis": "DIIS"}
    kw["convergence_method"] = conv_map.get(params.convergence_method, params.convergence_method)
    if params.diis_size != 8:
        kw["diis_size"] = str(params.diis_size)
    if params.diis_include_transform:
        kw["diis_include_transform"] = "true"
    if params.damping_factor != 0.9:
        kw["damping_factor"] = str(params.damping_factor)
    if params.method == "ROHF" and params.rohf_parameter_name != "Roothaan":
        kw["rohf_parameter_name"] = params.rohf_parameter_name
    if params.maxiter != 100:
        kw["maxiter"] = str(params.maxiter)
    if params.convergence_energy_threshold != 1e-6:
        kw["convergence_energy_threshold"] = str(params.convergence_energy_threshold)
    if params.schwarz_screening_threshold != 1e-12:
        kw["schwarz_screening_threshold"] = str(params.schwarz_screening_threshold)
    if params.initial_guess != "core":
        kw["initial_guess"] = params.initial_guess
    if params.post_hf_method in ("cis", "adc2", "adc2x", "eom_mp2", "eom_cc2", "eom_ccsd"):
        kw["n_excited_states"] = str(params.n_excited_states)
    if params.spin_type != "singlet":
        kw["spin_type"] = params.spin_type
    if params.excited_solver != "auto":
        solver_map = {"adc2": "adc2_solver", "adc2x": "adc2_solver",
                      "eom_mp2": "eom_mp2_solver", "eom_cc2": "eom_cc2_solver"}
        key = solver_map.get(params.post_hf_method)
        if key:
            kw[key] = params.excited_solver
    eri_map = {"ri": "RI", "direct": "Direct", "direct_ri": "Direct_RI", "stored": "stored"}
    if params.eri_method != "stored":
        kw["eri_method"] = eri_map.get(params.eri_method, params.eri_method)
    if params.auxiliary_basis:
        aux_dir = _GANSU_PATH / ("auxiliary_basis" if params.auxiliary_basis_dir == "auxiliary_basis" else "basis")
        kw["auxiliary_gbsfilename"] = str(aux_dir / f"{params.auxiliary_basis}.gbs")
    if params.mulliken:
        kw["mulliken"] = "1"
    if params.mayer:
        kw["mayer"] = "1"
    if params.wiberg:
        kw["wiberg"] = "1"
    if params.export_molden:
        kw["export_molden"] = "1"
    if params.verbose:
        kw["verbose"] = "1"
    return kw


# ---------------------------------------------------------------------------
#  Build structured result from gansu.Result (replaces parser.py)
# ---------------------------------------------------------------------------

def _build_result_dict(r: gansu.Result, params: RunParams,
                       scf_iterations: list[dict]) -> dict[str, Any]:
    """Build frontend-compatible result dict from gansu.Result."""
    data: dict[str, Any] = {"ok": True}

    # Molecule info
    atoms_list = []
    try:
        for i, (Z, x, y, z) in enumerate(r.atoms):
            elem = _ELEMENTS[Z] if Z < len(_ELEMENTS) else f"Z{Z}"
            atoms_list.append({"index": i, "element": elem, "coords": [x, y, z]})
    except Exception:
        pass
    data["molecule"] = {
        "num_atoms": r.num_atoms,
        "num_electrons": r.num_electrons,
        "atoms": atoms_list,
    }

    # Basis set
    data["basis_set"] = {"num_basis": r.num_basis}

    # SCF iterations (from progress callback)
    data["scf_iterations"] = scf_iterations

    # Summary
    data["summary"] = {
        "method": params.method,
        "total_energy": r.total_energy,
        "iterations": len(scf_iterations),
    }

    # Orbital energies
    try:
        nocc = r.num_electrons // 2
        eps = r.orbital_energies
        data["orbital_energies"] = [
            {"index": i + 1, "occupation": "occ" if i < nocc else "vir", "energy": float(e)}
            for i, e in enumerate(eps)
        ]
    except Exception:
        data["orbital_energies"] = []

    # Post-HF
    if params.post_hf_method != "none":
        corr = r.post_hf_energy
        data["post_hf"] = {
            "method": params.post_hf_method.upper(),
            "correction": corr,
            "total_energy": r.total_energy + corr,
        }

    # Excited states
    report = r.excited_state_report
    if report:
        data["excited_state_report"] = report

    # Mulliken charges
    if params.mulliken:
        try:
            charges = r.mulliken_charges
            data["mulliken"] = [
                {"index": i, "element": atoms_list[i]["element"] if i < len(atoms_list) else "?",
                 "charge": float(c)}
                for i, c in enumerate(charges)
            ]
        except Exception:
            pass

    # Bond orders
    if params.mayer:
        try:
            data["mayer_bond_order"] = r.mayer_bond_order.tolist()
        except Exception:
            pass
    if params.wiberg:
        try:
            data["wiberg_bond_order"] = r.wiberg_bond_order.tolist()
        except Exception:
            pass

    return data


# ---------------------------------------------------------------------------
#  Main execution — Python API with progress callback
# ---------------------------------------------------------------------------

async def run_hf_main(params: RunParams) -> dict[str, Any]:
    """Run GANSU and return structured result dict."""
    _ensure_init()
    xyz_path, tmp = _resolve_xyz(params)

    scf_iters: list[dict] = []
    ccsd_iters: list[dict] = []

    def _on_progress(stage, iter_num, values):
        if stage == "scf" and iter_num > 0:
            scf_iters.append({
                "iteration": iter_num,
                "total_energy": values[0],
                "delta_e": values[1] if len(values) > 1 else 0,
            })
        elif stage == "ccsd":
            ccsd_iters.append({
                "iteration": iter_num,
                "correlation_energy": values[0],
                "delta_e": values[1] if len(values) > 1 else 0,
            })

    try:
        kwargs = _build_kwargs(params)

        def _run():
            mol = gansu.Molecule(xyz_path, basis=params.basis, **kwargs)
            return mol.run(method=params.method, post_hf=params.post_hf_method,
                           quiet=True, on_progress=_on_progress)

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=params.timeout,
        )

        data = _build_result_dict(result, params, scf_iters)
        if ccsd_iters:
            data["ccsd_iterations"] = ccsd_iters
        return data

    except asyncio.TimeoutError:
        return {"ok": False, "error": "Timeout: calculation exceeded time limit"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


async def stream_hf_main(params: RunParams) -> AsyncGenerator[dict[str, Any], None]:
    """Stream calculation progress as dicts, then yield final result."""
    _ensure_init()
    xyz_path, tmp = _resolve_xyz(params)

    progress_queue: asyncio.Queue = asyncio.Queue()
    scf_iters: list[dict] = []

    def _on_progress(stage, iter_num, values):
        if stage == "scf":
            entry = {"stage": "scf", "iteration": iter_num,
                     "total_energy": values[0],
                     "delta_e": values[1] if len(values) > 1 else 0}
            if iter_num > 0:
                scf_iters.append(entry)
            progress_queue.put_nowait(entry)
        elif stage == "ccsd":
            progress_queue.put_nowait({
                "stage": "ccsd", "iteration": iter_num,
                "correlation_energy": values[0],
                "delta_e": values[1] if len(values) > 1 else 0})
        elif stage == "davidson":
            progress_queue.put_nowait({
                "stage": "davidson", "iteration": iter_num,
                "eigenvalues": values[:-1],
                "max_residual": values[-1] if values else 0})
        elif stage == "ccsd_lambda":
            progress_queue.put_nowait({
                "stage": "ccsd_lambda", "iteration": iter_num,
                "residual": values[0] if values else 0})

    result_holder: list = [None, None]  # [result, error]

    def _run():
        try:
            kwargs = _build_kwargs(params)
            mol = gansu.Molecule(xyz_path, basis=params.basis, **kwargs)
            r = mol.run(method=params.method, post_hf=params.post_hf_method,
                        quiet=True, on_progress=_on_progress)
            result_holder[0] = r
        except Exception as e:
            result_holder[1] = str(e)
        finally:
            progress_queue.put_nowait(None)  # sentinel

    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, _run)

    try:
        while True:
            item = await progress_queue.get()
            if item is None:
                break
            yield {"type": "progress", **item}

        await task

        if result_holder[1]:
            yield {"type": "error", "error": result_holder[1]}
        elif result_holder[0]:
            data = _build_result_dict(result_holder[0], params, scf_iters)
            yield {"type": "result", "data": data}
        yield {"type": "done"}

    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def list_sample_dirs() -> list[str]:
    dirs: list[str] = []
    if not XYZ_DIR.exists():
        return dirs
    if any(XYZ_DIR.glob("*.xyz")):
        dirs.append(".")
    for d in sorted(XYZ_DIR.iterdir()):
        if d.is_dir() and any(d.glob("*.xyz")):
            dirs.append(d.name)
    return dirs


def list_samples(subdir: str = ".") -> list[dict[str, str]]:
    samples = []
    if subdir == ".":
        target = XYZ_DIR
    else:
        if "/" in subdir or "\\" in subdir or ".." in subdir:
            return []
        target = XYZ_DIR / subdir
    if target.exists():
        for f in sorted(target.glob("*.xyz")):
            name = f.stem.replace("_", " ").replace("-", " ")
            samples.append({"filename": f.name, "name": name})
    return samples


def list_basis_sets() -> list[str]:
    return gansu.list_basis_sets()


def list_auxiliary_basis_sets() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    aux_dir = _GANSU_PATH / "auxiliary_basis"
    basis_dir = _GANSU_PATH / "basis"
    if aux_dir.exists():
        for f in sorted(aux_dir.glob("*.gbs")):
            result.append({"name": f.stem, "dir": "auxiliary_basis"})
            seen.add(f.stem)
    if basis_dir.exists():
        for f in sorted(basis_dir.glob("*.gbs")):
            if f.stem not in seen:
                result.append({"name": f.stem, "dir": "basis"})
    return result
