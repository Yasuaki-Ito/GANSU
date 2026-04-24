"""FastAPI backend for GANSU-UI — subprocess-based (process isolation for GPU stability)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from parser import parse_output
from runner import (GANSU_BIN, GANSU_PATH, XYZ_DIR, RunParams,
                    list_basis_sets, list_auxiliary_basis_sets,
                    list_sample_dirs, list_samples,
                    run_hf_main, stream_hf_main, run_pes_point)

app = FastAPI(title="GANSU-UI API")


@app.on_event("startup")
async def startup_check():
    print(f"GANSU_PATH: {GANSU_PATH}")
    print(f"GANSU_BIN:  {GANSU_BIN}")
    if not os.path.isfile(GANSU_BIN):
        print(f"WARNING: gansu binary not found at {GANSU_BIN}")
    else:
        print("gansu binary found.")
    print(f"Basis sets: {len(list_basis_sets())} available")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalcRequest(BaseModel):
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
    frozen_core: str = "none"
    eri_method: str = "stored"
    auxiliary_basis: str = ""
    auxiliary_basis_dir: str = "auxiliary_basis"
    mulliken: bool = False
    mayer: bool = False
    wiberg: bool = False
    export_molden: bool = False
    verbose: bool = False
    run_type: str = "energy"
    optimizer: str = "bfgs"
    timeout: int = 600


@app.get("/api/sample_dirs")
def get_sample_dirs():
    return list_sample_dirs()


@app.get("/api/samples")
def get_samples(dir: str = "."):
    return list_samples(dir)


@app.get("/api/samples/{filename}")
def get_sample_content(filename: str, dir: str = "."):
    for part in [filename, dir]:
        if ".." in part or "\\" in part:
            return {"error": "Invalid path"}
    if "/" in filename or not filename.endswith(".xyz"):
        return {"error": "Not found"}
    base = XYZ_DIR if dir == "." else XYZ_DIR / dir
    path = base / filename
    if not path.exists():
        return {"error": "Not found"}
    return {"filename": filename, "content": path.read_text()}


@app.get("/api/basis")
def get_basis():
    return list_basis_sets()


@app.get("/api/auxiliary_basis")
def get_auxiliary_basis():
    return list_auxiliary_basis_sets()


@app.post("/api/run")
async def run_calculation(req: CalcRequest):
    params = RunParams(**req.model_dump())
    stdout, stderr, code, molden = await run_hf_main(params)
    if code != 0:
        return {"ok": False, "error": stderr or f"gansu exited with code {code}", "raw_output": stdout}
    result = parse_output(stdout)
    data = {"ok": True, **result.to_dict()}
    if molden:
        data["molden_content"] = molden
    return data


class PESPointRequest(BaseModel):
    xyz_text: str
    basis: str = "sto-3g"
    method: str = "RHF"
    charge: int = 0
    post_hf_method: str = "none"
    timeout: int = 120
    use_prev_density: bool = False
    convergence_energy_threshold: float = 1e-6


# In-process PES state (density reuse between points)
_pes_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
_pes_density = None  # numpy array or None
_pes_initialized = False


def _ensure_gansu():
    global _pes_initialized
    if not _pes_initialized:
        import sys
        gansu_python = os.path.join(str(GANSU_PATH), "python")
        if gansu_python not in sys.path:
            sys.path.insert(0, gansu_python)
        if "GANSU_LIB" not in os.environ:
            for lib in [GANSU_PATH / "build" / "libgansu.so", GANSU_PATH / "build" / "libgansu.dylib"]:
                if lib.exists():
                    os.environ["GANSU_LIB"] = str(lib)
                    break
        import gansu as _g
        _g.init()
        _pes_initialized = True


@app.post("/api/pes/point")
async def run_pes_single(req: PESPointRequest):
    """Run a single PES point with in-process density reuse."""
    global _pes_density
    import asyncio
    loop = asyncio.get_event_loop()

    def _run():
        global _pes_density, _pes_initialized
        _ensure_gansu()
        import gansu as g
        import tempfile, numpy as np

        # Write xyz to temp file
        fd, tmp = tempfile.mkstemp(suffix=".xyz")
        with os.fdopen(fd, "w") as f:
            f.write(req.xyz_text)

        mol = None
        try:
            mol = g.Molecule(tmp, basis=req.basis, charge=str(req.charge),
                             convergence_energy_threshold=str(req.convergence_energy_threshold),
                             initial_guess='sad')
            if req.use_prev_density and _pes_density is not None:
                mol.set_initial_density(_pes_density)
                import sys
                print(f"[PES] Using previous density (norm={float(np.linalg.norm(_pes_density)):.4f})", file=sys.stderr)

            last_delta_e = [float('inf')]
            def _on_prog(stage, it, vals):
                if stage == 'scf' and len(vals) > 1 and it > 0:
                    last_delta_e[0] = abs(vals[1])

            r = mol.run(method=req.method, post_hf=req.post_hf_method,
                        quiet=True, on_progress=_on_prog)

            converged = last_delta_e[0] < req.convergence_energy_threshold

            if converged:
                try:
                    _pes_density = r.density_matrix.copy()
                except Exception:
                    pass

            return {
                "ok": True,
                "energy": r.total_energy,
                "correction": r.post_hf_energy,
                "converged": converged,
                "delta_e": last_delta_e[0],
            }
        except Exception as e:
            # Error: discard cached density to avoid propagating bad state
            _pes_density = None
            # C API already handles GPU recovery (cudaDeviceReset + re-init)
            # so we keep _pes_initialized = True
            import sys, traceback
            print(f"[PES] Error, reset state: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {"ok": False, "error": str(e)}
        finally:
            # Explicitly destroy molecule handle to release GPU memory
            del mol
            try: os.unlink(tmp)
            except: pass

    result = await loop.run_in_executor(None, _run)
    return result


@app.post("/api/pes/reset")
async def reset_pes_density():
    """Clear cached density for new PES scan."""
    global _pes_density
    _pes_density = None
    return {"ok": True}


@app.post("/api/run/stream")
async def stream_calculation(req: CalcRequest):
    params = RunParams(**req.model_dump())

    async def event_generator():
        full_output: list[str] = []
        stderr_text = ""
        returncode = 0
        molden_content = ""
        async for kind, text, code in stream_hf_main(params):
            if kind == "stdout":
                full_output.append(text)
                yield f"data: {json.dumps({'type': 'line', 'text': text.rstrip()})}\n\n"
            elif kind == "exit":
                stderr_text = text
                returncode = code
            elif kind == "molden":
                molden_content = text
        if returncode != 0:
            error_msg = stderr_text or f"gansu exited with code {returncode}"
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg, 'raw_output': ''.join(full_output)})}\n\n"
        else:
            result = parse_output("".join(full_output))
            data = result.to_dict()
            if molden_content:
                data["molden_content"] = molden_content
            yield f"data: {json.dumps({'type': 'result', 'data': data})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── In-process calculation (no subprocess) ──

# Element number → symbol
_ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn", 35: "Br", 53: "I",
}


class InProcessCalcRequest(BaseModel):
    xyz_text: str = ""
    xyz_file: str = ""
    xyz_dir: str = "."
    basis: str = "sto-3g"
    method: str = "RHF"
    charge: int = 0
    beta_to_alpha: int = 0
    convergence_method: str = "diis"
    diis_size: int = 8
    damping_factor: float = 0.9
    rohf_parameter_name: str = "Roothaan"
    maxiter: int = 100
    convergence_energy_threshold: float = 1e-6
    schwarz_screening_threshold: float = 1e-12
    initial_guess: str = "sad"
    post_hf_method: str = "none"
    n_excited_states: int = 5
    spin_type: str = "singlet"
    excited_solver: str = "auto"
    frozen_core: str = "none"
    eri_method: str = "stored"
    auxiliary_basis: str = ""
    auxiliary_basis_dir: str = "auxiliary_basis"
    mulliken: bool = False
    mayer: bool = False
    wiberg: bool = False
    timeout: int = 600


def _resolve_inprocess_xyz(req: InProcessCalcRequest):
    """Resolve XYZ to a file path. Returns (path, tmp_to_delete_or_None)."""
    import tempfile
    if req.xyz_text:
        fd, tmp = tempfile.mkstemp(suffix=".xyz")
        with os.fdopen(fd, "w") as f:
            f.write(req.xyz_text)
        return tmp, tmp
    elif req.xyz_file:
        base = XYZ_DIR if req.xyz_dir == "." else XYZ_DIR / req.xyz_dir
        return str(base / req.xyz_file), None
    else:
        raise ValueError("No molecule specified (xyz_text or xyz_file required)")


@app.post("/api/run/inprocess")
async def run_inprocess(req: InProcessCalcRequest):
    """Run calculation in-process via Python API (no subprocess)."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _run():
        _ensure_gansu()
        import gansu as g
        import tempfile, numpy as np

        xyz_path, tmp = _resolve_inprocess_xyz(req)

        import re as _re
        mol = None
        try:
            # Build extra params (same as streaming endpoint)
            extra_params = {}
            if req.excited_solver != "auto":
                solver_param = {"adc2": "adc2_solver", "adc2x": "adc2_solver",
                                "eom_mp2": "eom_mp2_solver", "eom_cc2": "eom_cc2_solver"
                                }.get(req.post_hf_method)
                if solver_param:
                    extra_params[solver_param] = req.excited_solver
            if req.post_hf_method in ("cis", "adc2", "adc2x", "eom_mp2", "eom_cc2", "eom_ccsd"):
                extra_params["n_excited_states"] = str(req.n_excited_states)
            if req.spin_type != "singlet":
                extra_params["spin_type"] = req.spin_type
            if req.frozen_core != "none":
                extra_params["frozen_core"] = req.frozen_core
            if req.eri_method != "stored":
                eri_map = {"ri": "RI", "direct": "Direct", "direct_ri": "Direct_RI"}
                extra_params["eri_method"] = eri_map.get(req.eri_method, req.eri_method)

            mol = g.Molecule(
                xyz_path, basis=req.basis, charge=str(req.charge),
                convergence_energy_threshold=str(req.convergence_energy_threshold),
                initial_guess=req.initial_guess,
                **extra_params,
            )

            scf_iterations = []
            def _on_prog(stage, it, vals):
                if stage == "scf":
                    entry = {"iteration": it}
                    if len(vals) > 0: entry["energy"] = vals[0]
                    if len(vals) > 1: entry["delta_e"] = vals[1]
                    if len(vals) > 2: entry["total_energy"] = vals[2]
                    scf_iterations.append(entry)

            r = mol.run(
                method=req.method, post_hf=req.post_hf_method,
                quiet=True, on_progress=_on_prog,
            )

            bohr2ang = 0.529177249
            result = {
                "ok": True,
                "summary": {
                    "total_energy": r.total_energy,
                    "nuclear_repulsion_energy": r.nuclear_repulsion_energy,
                    "electronic_energy": r.total_energy - r.nuclear_repulsion_energy,
                    "iterations": len(scf_iterations),
                    "method": req.method,
                },
                "molecule": {
                    "num_atoms": r.num_atoms,
                    "num_electrons": r.num_electrons,
                    "num_occ": r.num_electrons // 2,
                    "num_vir": r.num_basis - r.num_electrons // 2,
                    "num_frozen": r.num_frozen_core,
                    "frozen_core": req.frozen_core if req.frozen_core != "none" else None,
                },
                "basis_set": {"num_basis": r.num_basis},
                "scf_iterations": scf_iterations,
                "orbital_energies": [],
            }

            if req.post_hf_method != "none":
                result["post_hf"] = {
                    "method": req.post_hf_method.upper(),
                    "correction": r.post_hf_energy,
                    "total_energy": r.total_energy + r.post_hf_energy,
                }

            try:
                eps = r.orbital_energies
                n_occ = r.num_electrons // 2
                for i, e in enumerate(eps):
                    result["orbital_energies"].append({
                        "index": i + 1,
                        "occupation": "occupied" if i < n_occ else "virtual",
                        "energy": float(e),
                    })
            except Exception:
                pass

            if req.mulliken:
                try:
                    charges = r.mulliken_charges
                    atoms_info = r.atoms
                    result["mulliken"] = [
                        {"index": i + 1, "element": _ELEMENT_SYMBOLS.get(atoms_info[i][0], "?"),
                         "charge": float(charges[i])}
                        for i in range(len(charges))
                    ]
                except Exception:
                    pass

            try:
                atoms_info = r.atoms
                result["molecule"]["atoms"] = [
                    {"index": i + 1,
                     "element": _ELEMENT_SYMBOLS.get(a[0], "?"),
                     "coords": [a[1] * bohr2ang, a[2] * bohr2ang, a[3] * bohr2ang]}
                    for i, a in enumerate(atoms_info)
                ]
            except Exception:
                pass

            # Excited states
            try:
                es_report = r.excited_state_report
                if es_report and es_report.strip():
                    es_list = []
                    for line in es_report.split('\n'):
                        m = _re.match(
                            r'\s*(\d+)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+(.*)',
                            line)
                        if m:
                            es_list.append({
                                "state": int(m.group(1)),
                                "energy_ha": float(m.group(2)),
                                "energy_ev": float(m.group(3)),
                                "osc_strength": float(m.group(4)),
                                "transitions": m.group(5).strip(),
                            })
                    if es_list:
                        result["excited_states"] = es_list
                        result["excited_states_method"] = req.post_hf_method.upper()
                        result["excited_states_spin"] = req.spin_type
            except Exception:
                pass

            return result

        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            return {"ok": False, "error": str(e)}
        finally:
            del mol
            if tmp:
                try: os.unlink(tmp)
                except: pass

    result = await loop.run_in_executor(None, _run)
    return result


@app.post("/api/run/inprocess/stream")
async def run_inprocess_stream(req: InProcessCalcRequest):
    """Run calculation in-process with SSE streaming of progress."""
    import asyncio, queue, threading

    progress_queue: queue.Queue = queue.Queue()

    def _run():
        _ensure_gansu()
        import gansu as g
        import numpy as np

        xyz_path, tmp = _resolve_inprocess_xyz(req)

        import time as _time
        mol = None
        t_start = _time.monotonic()
        try:
            # Pass all parameters to gansu
            extra_params = {}
            if req.beta_to_alpha: extra_params["beta_to_alpha"] = str(req.beta_to_alpha)
            if req.convergence_method != "diis": extra_params["convergence_method"] = req.convergence_method
            if req.diis_size != 8: extra_params["diis_size"] = str(req.diis_size)
            if req.damping_factor != 0.9: extra_params["damping_factor"] = str(req.damping_factor)
            if req.method == "ROHF" and req.rohf_parameter_name != "Roothaan":
                extra_params["rohf_parameter_name"] = req.rohf_parameter_name
            if req.maxiter != 100: extra_params["maxiter"] = str(req.maxiter)
            if req.schwarz_screening_threshold != 1e-12:
                extra_params["schwarz_screening_threshold"] = str(req.schwarz_screening_threshold)
            if req.eri_method != "stored":
                eri_map = {"ri": "RI", "direct": "Direct", "direct_ri": "Direct_RI"}
                extra_params["eri_method"] = eri_map.get(req.eri_method, req.eri_method)
            if req.auxiliary_basis:
                from runner import AUX_BASIS_DIR, BASIS_DIR
                aux_dir = AUX_BASIS_DIR if req.auxiliary_basis_dir == "auxiliary_basis" else BASIS_DIR
                extra_params["auxiliary_basis_gbs"] = str(aux_dir / f"{req.auxiliary_basis}.gbs")
            if req.post_hf_method in ("cis", "adc2", "adc2x", "eom_mp2", "eom_cc2", "eom_ccsd"):
                extra_params["n_excited_states"] = str(req.n_excited_states)
            if req.excited_solver != "auto":
                # Map post-HF method to the correct solver parameter name
                solver_param = {"adc2": "adc2_solver", "adc2x": "adc2_solver",
                                "eom_mp2": "eom_mp2_solver", "eom_cc2": "eom_cc2_solver"
                                }.get(req.post_hf_method)
                if solver_param:
                    extra_params[solver_param] = req.excited_solver
            if req.spin_type != "singlet":
                extra_params["spin_type"] = req.spin_type
            if req.frozen_core != "none":
                extra_params["frozen_core"] = req.frozen_core

            mol = g.Molecule(
                xyz_path, basis=req.basis, charge=str(req.charge),
                convergence_energy_threshold=str(req.convergence_energy_threshold),
                initial_guess=req.initial_guess,
                **extra_params,
            )

            scf_iterations = []

            def _on_prog(stage, it, vals):
                entry = {"stage": stage, "iteration": it, "values": vals}
                if stage == "scf":
                    scf_entry = {"iteration": it}
                    if len(vals) > 0: scf_entry["energy"] = vals[0]
                    if len(vals) > 1: scf_entry["delta_e"] = vals[1]
                    if len(vals) > 2: scf_entry["total_energy"] = vals[2]
                    scf_iterations.append(scf_entry)
                progress_queue.put(("progress", entry))

            progress_queue.put(("progress", {"stage": "setup", "iteration": 0, "values": []}))

            r = mol.run(
                method=req.method, post_hf=req.post_hf_method,
                quiet=True, on_progress=_on_prog,
            )

            elapsed_ms = (_time.monotonic() - t_start) * 1000
            bohr2ang = 0.529177249

            # Build comprehensive result matching CalculationResult type
            result = {
                "ok": True,
                "summary": {
                    "total_energy": r.total_energy,
                    "nuclear_repulsion_energy": r.nuclear_repulsion_energy,
                    "electronic_energy": r.total_energy - r.nuclear_repulsion_energy,
                    "iterations": len(scf_iterations),
                    "method": req.method,
                    "convergence_algorithm": req.convergence_method.upper(),
                    "initial_guess": req.initial_guess,
                    "computing_time_ms": elapsed_ms,
                },
                "molecule": {
                    "num_atoms": r.num_atoms,
                    "num_electrons": r.num_electrons,
                    "num_occ": r.num_electrons // 2,
                    "num_vir": r.num_basis - r.num_electrons // 2,
                    "num_frozen": r.num_frozen_core,
                    "frozen_core": req.frozen_core if req.frozen_core != "none" else None,
                },
                "basis_set": {"num_basis": r.num_basis},
                "scf_iterations": scf_iterations,
                "orbital_energies": [],
                "orbital_energies_beta": [],
                "mulliken": [],
                "mayer_bond_order": [],
                "wiberg_bond_order": [],
                "timing": {},
            }

            # Post-HF
            if req.post_hf_method != "none":
                result["post_hf"] = {
                    "method": req.post_hf_method.upper(),
                    "correction": r.post_hf_energy,
                    "total_energy": r.total_energy + r.post_hf_energy,
                }

            # Orbital energies
            try:
                eps = r.orbital_energies
                n_occ = r.num_electrons // 2
                for i, e in enumerate(eps):
                    result["orbital_energies"].append({
                        "index": i + 1,
                        "occupation": "occupied" if i < n_occ else "virtual",
                        "energy": float(e),
                    })
            except Exception:
                pass

            # Atoms with coordinates
            try:
                atoms_info = r.atoms
                result["molecule"]["atoms"] = [
                    {"index": i + 1,
                     "element": _ELEMENT_SYMBOLS.get(a[0], "?"),
                     "coords": [a[1] * bohr2ang, a[2] * bohr2ang, a[3] * bohr2ang]}
                    for i, a in enumerate(atoms_info)
                ]
            except Exception:
                pass

            # Mulliken charges
            if req.mulliken:
                try:
                    charges = r.mulliken_charges
                    atoms_info = r.atoms
                    result["mulliken"] = [
                        {"index": i + 1, "element": _ELEMENT_SYMBOLS.get(atoms_info[i][0], "?"),
                         "charge": float(charges[i])}
                        for i in range(len(charges))
                    ]
                except Exception:
                    pass

            # Mayer bond order
            if req.mayer:
                try:
                    bo = r.mayer_bond_order
                    result["mayer_bond_order"] = bo.tolist()
                except Exception:
                    pass

            # Wiberg bond order
            if req.wiberg:
                try:
                    bo = r.wiberg_bond_order
                    result["wiberg_bond_order"] = bo.tolist()
                except Exception:
                    pass

            # Excited states
            try:
                es_report = r.excited_state_report
                if es_report and es_report.strip():
                    result["excited_states_report"] = es_report
                    # Parse excited states from report text
                    import re
                    es_list = []
                    for line in es_report.split('\n'):
                        m = re.match(
                            r'\s*(\d+)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+(.*)',
                            line)
                        if m:
                            es_list.append({
                                "state": int(m.group(1)),
                                "energy_ha": float(m.group(2)),
                                "energy_ev": float(m.group(3)),
                                "osc_strength": float(m.group(4)),
                                "transitions": m.group(5).strip(),
                            })
                    if es_list:
                        result["excited_states"] = es_list
                        result["excited_states_method"] = req.post_hf_method.upper()
                        result["excited_states_spin"] = req.spin_type
            except Exception:
                pass

            # Energy difference from last SCF iteration
            if scf_iterations:
                last = scf_iterations[-1]
                if "delta_e" in last:
                    result["summary"]["energy_difference"] = last["delta_e"]

            progress_queue.put(("result", result))

        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            progress_queue.put(("error", {"ok": False, "error": str(e)}))
        finally:
            del mol
            if tmp:
                try: os.unlink(tmp)
                except: pass
            progress_queue.put(("done", None))

    async def event_generator():
        import queue as _q
        loop = asyncio.get_event_loop()
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            # Wait for at least one event (short timeout for responsiveness)
            try:
                kind, data = await loop.run_in_executor(
                    None, lambda: progress_queue.get(timeout=0.02))
            except _q.Empty:
                if not thread.is_alive():
                    break
                continue

            # Drain all queued events and yield each immediately
            batch = [(kind, data)]
            while not progress_queue.empty():
                try:
                    batch.append(progress_queue.get_nowait())
                except _q.Empty:
                    break

            done = False
            for kind, data in batch:
                if kind == "progress":
                    yield f"data: {json.dumps({'type': 'progress', **data})}\n\n"
                elif kind == "result":
                    yield f"data: {json.dumps({'type': 'result', 'data': data})}\n\n"
                elif kind == "error":
                    yield f"data: {json.dumps({'type': 'error', 'error': data['error']})}\n\n"
                elif kind == "done":
                    yield "data: {\"type\": \"done\"}\n\n"
                    done = True
            if done:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve frontend static files
_HERE = Path(__file__).resolve().parent
for _base in [_HERE.parent, _HERE]:
    _dist = _base / "frontend" / "dist"
    if _dist.is_dir():
        app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
        break
