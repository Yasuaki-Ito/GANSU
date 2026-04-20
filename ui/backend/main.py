"""FastAPI backend for GANSU-UI — Python API backend (no subprocess, no parser)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from runner import (RunParams, list_basis_sets, list_auxiliary_basis_sets,
                    list_sample_dirs, list_samples, run_hf_main, stream_hf_main, XYZ_DIR)

app = FastAPI(title="GANSU-UI API")


@app.on_event("startup")
async def startup_check():
    try:
        import gansu
        basis = gansu.list_basis_sets()
        print(f"GANSU-UI ready: {len(basis)} basis sets available")
    except Exception as e:
        print(f"WARNING: gansu not available: {e}")


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
    eri_method: str = "stored"
    auxiliary_basis: str = ""
    auxiliary_basis_dir: str = "auxiliary_basis"
    mulliken: bool = False
    mayer: bool = False
    wiberg: bool = False
    export_molden: bool = False
    verbose: bool = False
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
    return await run_hf_main(params)


@app.post("/api/run/stream")
async def stream_calculation(req: CalcRequest):
    params = RunParams(**req.model_dump())

    async def event_generator():
        async for item in stream_hf_main(params):
            yield f"data: {json.dumps(item)}\n\n"

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
