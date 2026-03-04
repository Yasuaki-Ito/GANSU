"""FastAPI backend for GANSU-UI."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from parser import parse_output
from runner import HF_MAIN, GANSU_PATH, XYZ_DIR, RunParams, list_basis_sets, list_auxiliary_basis_sets, list_sample_dirs, list_samples, run_hf_main, stream_hf_main

app = FastAPI(title="GANSU-UI API")


@app.on_event("startup")
async def startup_check():
    print(f"GANSU_PATH: {GANSU_PATH}")
    print(f"HF_MAIN:    {HF_MAIN}")
    if not os.path.isfile(HF_MAIN):
        print(f"WARNING: HF_main not found at {HF_MAIN}")
    else:
        print("HF_main found.")

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
    """Return the content of a sample XYZ file."""
    # Security: reject path traversal
    for part in [filename, dir]:
        if ".." in part or "\\" in part:
            return {"error": "Invalid path"}
    if "/" in filename:
        return {"error": "Invalid filename"}
    if not filename.endswith(".xyz"):
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
        return {"ok": False, "error": stderr or f"HF_main exited with code {code}", "raw_output": stdout}
    result = parse_output(stdout)
    data = {"ok": True, **result.to_dict()}
    if molden:
        data["molden_content"] = molden
    return data


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
            error_msg = stderr_text or f"HF_main exited with code {returncode}"
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


# Serve frontend static files (npm run build -> frontend/dist/)
# Local: backend/main.py -> ../../frontend/dist
# Remote: gansu-ui/main.py -> ../frontend/dist
_HERE = Path(__file__).resolve().parent
for _base in [_HERE.parent, _HERE]:
    _dist = _base / "frontend" / "dist"
    if _dist.is_dir():
        app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
        break
