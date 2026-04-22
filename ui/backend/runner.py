"""Execute GANSU calculations via subprocess (process isolation for GPU stability)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

# Path configuration
GANSU_PATH = Path(os.environ.get("GANSU_PATH", Path.home() / "GANSU"))
GANSU_BIN = os.environ.get("GANSU_BIN", str(GANSU_PATH / "build" / "gansu"))
# Fallback to HF_main if gansu binary not found
if not os.path.isfile(GANSU_BIN):
    alt = str(GANSU_PATH / "build" / "HF_main")
    if os.path.isfile(alt):
        GANSU_BIN = alt
XYZ_DIR = GANSU_PATH / "xyz"
BASIS_DIR = GANSU_PATH / "basis"
AUX_BASIS_DIR = GANSU_PATH / "auxiliary_basis"


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
    run_type: str = "energy"
    optimizer: str = "bfgs"
    timeout: int = 600


def build_command(params: RunParams, xyz_path: str) -> list[str]:
    """Build gansu command-line arguments."""
    cmd = [GANSU_BIN]
    cmd.extend(["-x", xyz_path])
    cmd.extend(["-g", params.basis])  # gansu resolves basis name automatically
    cmd.extend(["-m", params.method])
    cmd.extend(["-c", str(params.charge)])

    if params.beta_to_alpha:
        cmd.extend(["--beta_to_alpha", str(params.beta_to_alpha)])
    if params.convergence_method != "diis":
        conv_map = {"optimal_damping": "OptimalDamping", "damping": "Damping", "diis": "DIIS"}
        cmd.extend(["--convergence_method", conv_map.get(params.convergence_method, params.convergence_method)])
    if params.diis_size != 8:
        cmd.extend(["--diis_size", str(params.diis_size)])
    if params.diis_include_transform:
        cmd.extend(["--diis_include_transform", "true"])
    if params.damping_factor != 0.9:
        cmd.extend(["--damping_factor", str(params.damping_factor)])
    if params.method == "ROHF" and params.rohf_parameter_name != "Roothaan":
        cmd.extend(["--rohf_parameter_name", params.rohf_parameter_name])
    if params.maxiter != 100:
        cmd.extend(["--maxiter", str(params.maxiter)])
    if params.convergence_energy_threshold != 1e-6:
        cmd.extend(["--convergence_energy_threshold", str(params.convergence_energy_threshold)])
    if params.schwarz_screening_threshold != 1e-12:
        cmd.extend(["--schwarz_screening_threshold", str(params.schwarz_screening_threshold)])
    if params.initial_guess != "core":
        cmd.extend(["--initial_guess", params.initial_guess])
    if params.post_hf_method != "none":
        cmd.extend(["--post_hf_method", params.post_hf_method])
    if params.post_hf_method in ("cis", "adc2", "adc2x", "eom_mp2", "eom_cc2", "eom_ccsd"):
        cmd.extend(["--n_excited_states", str(params.n_excited_states)])
    if params.spin_type != "singlet":
        cmd.extend(["--spin_type", params.spin_type])
    if params.excited_solver != "auto":
        solver_param_map = {
            "adc2": "adc2_solver", "adc2x": "adc2_solver",
            "eom_mp2": "eom_mp2_solver", "eom_cc2": "eom_cc2_solver",
        }
        solver_key = solver_param_map.get(params.post_hf_method)
        if solver_key:
            cmd.extend([f"--{solver_key}", params.excited_solver])
    if params.eri_method != "stored":
        eri_map = {"ri": "RI", "direct": "Direct", "direct_ri": "Direct_RI", "stored": "stored"}
        cmd.extend(["--eri_method", eri_map.get(params.eri_method, params.eri_method)])
    if params.auxiliary_basis:
        aux_dir = AUX_BASIS_DIR if params.auxiliary_basis_dir == "auxiliary_basis" else BASIS_DIR
        cmd.extend(["-ag", str(aux_dir / f"{params.auxiliary_basis}.gbs")])
    if params.run_type != "energy":
        cmd.extend(["-r", params.run_type])
    if params.run_type == "optimize" and params.optimizer != "bfgs":
        cmd.extend(["--optimizer", params.optimizer])
    if params.mulliken:
        cmd.extend(["--mulliken", "1"])
    if params.mayer:
        cmd.extend(["--mayer", "1"])
    if params.wiberg:
        cmd.extend(["--wiberg", "1"])
    if params.export_molden:
        cmd.extend(["--export_molden", "1"])
    if params.verbose:
        cmd.extend(["-v", "1"])

    return cmd


async def run_pes_point(params: RunParams, density_file: str | None = None) -> tuple[str, str, int, str]:
    """Run a single PES point with optional density chaining.
    Uses --save_density to write density for the next point,
    and --load_density to read from previous point."""
    xyz_path, tmp = _resolve_xyz(params)
    workdir = tempfile.mkdtemp(prefix="gansu_pes_")
    save_path = os.path.join(workdir, "density.bin")
    try:
        cmd = build_command(params, xyz_path)
        cmd.extend(["--save_density", save_path])
        if density_file and os.path.isfile(density_file):
            cmd.extend(["--load_density", density_file])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=params.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", "Timeout", -1, ""

        # Copy saved density for chaining
        new_density = ""
        if os.path.isfile(save_path):
            import shutil
            persistent = tempfile.mktemp(suffix=".bin", prefix="gansu_density_")
            shutil.copy2(save_path, persistent)
            new_density = persistent

        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            proc.returncode or 0,
            new_density,
        )
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _resolve_xyz(params: RunParams) -> tuple[str, str | None]:
    if params.xyz_file:
        base = XYZ_DIR if params.xyz_dir == "." else XYZ_DIR / params.xyz_dir
        return str(base / params.xyz_file), None
    fd, tmp = tempfile.mkstemp(suffix=".xyz", prefix="gansu_")
    with os.fdopen(fd, "w") as f:
        f.write(params.xyz_text)
    return tmp, tmp


async def run_hf_main(params: RunParams) -> tuple[str, str, int, str]:
    """Run gansu and return (stdout, stderr, returncode, molden_content)."""
    xyz_path, tmp = _resolve_xyz(params)
    workdir = tempfile.mkdtemp(prefix="gansu_run_")
    try:
        cmd = build_command(params, xyz_path)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=params.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", "Timeout: calculation exceeded time limit", -1, ""

        molden = ""
        molden_path = Path(workdir) / "output.molden"
        if molden_path.exists():
            molden = molden_path.read_text(errors="replace")

        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            proc.returncode or 0,
            molden,
        )
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


async def stream_hf_main(params: RunParams) -> AsyncGenerator[tuple[str, str, int], None]:
    """Stream gansu stdout line by line."""
    xyz_path, tmp = _resolve_xyz(params)
    workdir = tempfile.mkdtemp(prefix="gansu_run_")
    try:
        cmd = build_command(params, xyz_path)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("stdout", line.decode("utf-8", errors="replace"), 0)
        await proc.wait()
        stderr = await proc.stderr.read()
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        yield ("exit", stderr_text, proc.returncode or 0)
        molden_path = Path(workdir) / "output.molden"
        if molden_path.exists():
            yield ("molden", molden_path.read_text(errors="replace"), 0)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


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
    if BASIS_DIR.exists():
        return sorted({f.stem for f in BASIS_DIR.glob("*.gbs")})
    return []


def list_auxiliary_basis_sets() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    if AUX_BASIS_DIR.exists():
        for f in sorted(AUX_BASIS_DIR.glob("*.gbs")):
            result.append({"name": f.stem, "dir": "auxiliary_basis"})
            seen.add(f.stem)
    if BASIS_DIR.exists():
        for f in sorted(BASIS_DIR.glob("*.gbs")):
            if f.stem not in seen:
                result.append({"name": f.stem, "dir": "basis"})
    return result
