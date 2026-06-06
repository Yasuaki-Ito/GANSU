# GANSU size-scaling + max-size + ORCA comparison benchmark

Measures, per method, on **H200×4**:
1. **How large a molecule each method can handle** (max size before OOM/timeout).
2. **Wall time vs molecule size** (size scaling).
3. **GANSU (GPU) vs ORCA (CPU) wall time** at production settings.

Methods: `rihf` (distributed RI-HF), `rimp2` (RI-MP2), `dlpno_ccsd` (DLPNO-CCSD ground
state — yields both DLPNO-MP2 and CCSD phase times), `dlpno_steom` (STEOM-DLPNO-CCSD,
frozen-core, env-free auto-scaling).

All GANSU runs use **spherical harmonics** (`--use_spherical 1`, 5D/7F) and the **SAD**
initial guess — cc-pVDZ is defined spherical and ORCA uses spherical for cc-pVnZ by
default, so both choices keep the GANSU-vs-ORCA comparison apples-to-apples (no ORCA
keyword needed; ORCA is spherical by default for cc-pVDZ).

Size series + per-method molecule lists live in `series.tsv` (edit freely).
Both runners **stop at the first failure for a method** — once a size is too big to
run, no larger molecule is attempted (no wasted runs).

## 1. GANSU (run from `build/`)

```bash
cd build
# one method, on 4 GPUs (per-job cap 120 min):
bash ../bench/run_gansu.sh dlpno_ccsd  4 120
bash ../bench/run_gansu.sh dlpno_steom 4 180
bash ../bench/run_gansu.sh rihf        4 120
bash ../bench/run_gansu.sh rimp2       4 120
# optional single-GPU reference column (GPU-count benefit):
bash ../bench/run_gansu.sh dlpno_ccsd  1 120
```
Logs → `bench/logs/`, per-method results → `bench/results_<method>_g<N>.tsv`.
Parse everything to a tidy CSV (phase breakdown: ri_Bbuild/scf/dlpno_mp2/dlpno_ccsd_t2/cis/ip/ea/steom):
```bash
python3 ../bench/parse_gansu.py > ../bench/gansu_bench.csv
```

## 2. ORCA (CPU reference)

```bash
python3 bench/make_orca_inputs.py 64 3000      # NPROCS=64, 3 GB/rank → bench/orca/inp/*.inp
ORCA=/opt/orca6 bash bench/run_orca.sh dlpno_ccsd  240
ORCA=/opt/orca6 bash bench/run_orca.sh dlpno_steom 360
ORCA=/opt/orca6 bash bench/run_orca.sh rihf        240
ORCA=/opt/orca6 bash bench/run_orca.sh rimp2       240
```
Headers used (cc-pVDZ; frozen core = ORCA default = matches GANSU `--frozen_core auto`):
`RHF RIJCOSX` / `RI-MP2 cc-pVDZ/C` / `DLPNO-CCSD cc-pVDZ/C` / `STEOM-DLPNO-CCSD cc-pVDZ/C nroots 5`.

## 3. Comparison table

```bash
python3 bench/parse_orca.py                         > bench/orca_bench.csv
python3 bench/parse_orca.py bench/gansu_bench.csv    > bench/compare.csv
```
`compare.csv`: method, molecule, natoms, gansu_wall_s, orca_wall_s, speedup (orca/gansu),
and status of each — so the **max size** per package = last `OK` row before OOM.

## SCF initial guess (important for fair SCF timing)
GANSU defaults to the `core` guess → many, run-dependent SCF iterations, so the total
SCF wall is dominated by the iteration count rather than the per-iteration GPU cost.
The runner therefore uses **`--initial_guess sad`** (override with `GUESS=core ...`),
which converges in few, stable iterations and is comparable to ORCA's default
(PAtom/PModel). The parser additionally reports **`scf_iters`** and
**`fock_per_iter_ms`** (= `compute_fock_matrix` total / calls) — the latter is the
**iteration-count-independent** RI-HF GPU metric; quote it alongside total SCF wall.
(SAD needs the `basis/<basis>.sad` atomic-density cache; `script/generate_sad_cache.py`
creates it if missing.)

## Notes / caveats (for the paper)
- **Hardware asymmetry is the point**: GANSU = 4×H200 GPU, ORCA = N-core CPU. State both.
- Algorithms are production-equivalent, not byte-identical (ORCA HF = RIJCOSX vs GANSU full
  RI-JK; DLPNO/STEOM presets default vs GANSU `normal`). For accuracy parity, align thresholds.
- `dlpno_steom` uses the env-free auto-scale (NSLAB/device-balancing auto-enabled at
  pentacene+); at total_dim>12000 the STEOM final diag uses iterative Davidson — add
  `GANSU_STEOM_DENSE_DIAG=1` for deterministic roots if a size shows spurious values.
- `series.tsv` order = increasing size; trim/extend the per-method `tags` to control reach.
