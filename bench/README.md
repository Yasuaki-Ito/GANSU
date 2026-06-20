# GANSU size-scaling + max-size + ORCA comparison benchmark

Measures, per method, on **H200×4**:
1. **How large a molecule each method can handle** (max size before OOM/timeout).
2. **Wall time vs molecule size** (size scaling).
3. **GANSU (GPU) vs ORCA (CPU) wall time** at production settings.

Methods: `rihf` (distributed RI-HF), `rimp2` (RI-MP2), `dlpno_ccsd` (DLPNO-CCSD ground
state — yields both DLPNO-MP2 and CCSD phase times), `dlpno_ccsd_t` (DLPNO-CCSD(T),
perturbative triples on top of the ground CCSD — O(N^7), steeper size ceiling),
`dlpno_steom` (STEOM-DLPNO-CCSD, frozen-core, env-free auto-scaling).
All DLPNO methods run `--frozen_core auto` (matches ORCA's default frozen core).

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

## 3. Comparison table (wall time)

```bash
python3 bench/parse_orca.py                         > bench/orca_bench.csv
python3 bench/parse_orca.py bench/gansu_bench.csv    > bench/compare.csv
```
`compare.csv`: method, molecule, natoms, gansu_wall_s, orca_wall_s, speedup (orca/gansu),
and status of each — so the **max size** per package = last `OK` row before OOM.

## 4. STEOM accuracy table (excitation energies, eV)

The *accuracy* counterpart to the wall-time table: pairs the STEOM excited-state
energies GANSU and ORCA print for the same molecule, root by root. Run the same
`dlpno_steom` jobs as above (§1, §2) so both `bench/logs/dlpno_steom_g*_*.log` and
`bench/orca/inp/dlpno_steom__*.out` exist, then:
```bash
python3 bench/compare_steom_roots.py > bench/steom_roots.csv   # MAD/MAX summary → stderr
```
`steom_roots.csv`: molecule, natoms, state, gansu_k, gansu_eV, orca_iroot, orca_eV,
diff_eV, abs_diff_meV. Roots are paired by ascending-energy position (state 1 = lowest).
**Caveat**: near-degenerate states (D2h acenes — see the known STEOM root jitter) can
reorder between packages, so an index pairing may misalign one pair; eyeball the table
before trusting a single large `diff`. For deterministic GANSU roots add
`GANSU_STEOM_DENSE_DIAG=1` (see §Notes). Keep thresholds aligned (GANSU `normal` preset
vs ORCA default) for a fair accuracy comparison.

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

---

## Python driver (recommended) — `bench.py`

One script replaces `run_gansu.sh` + `run_orca.sh` + `make_orca_inputs.py` +
`parse_*.py`. Each (molecule, method) is an isolated subprocess (clean GPU memory,
OOM/crash isolation), wall-timed and parsed into one CSV + a live table.

```bash
# generate the local-chromophore size series (xyz/bench_steom) once:
python3 ../bench/make_local_series.py        # -> series_{aldehyde,ketone,amide,nitrile,alkylbenzene}.tsv

cd build
# GANSU-only timing/scaling:
python3 ../bench/bench.py --family aldehyde --methods dlpno_ccsd,dlpno_steom --num_gpus 4

# GANSU vs ORCA (adds orca subprocess + speedup column):
python3 ../bench/bench.py --family aldehyde --methods dlpno_ccsd \
    --num_gpus 4 --orca /opt/orca6/orca --orca-nprocs 64 --timeout 240

# any series file / explicit molecules:
python3 ../bench/bench.py --series ../bench/series_alkylbenzene.tsv --methods dlpno_steom --num_gpus 4
python3 ../bench/bench.py --xyz ../xyz/Naphthalene.xyz ../xyz/Pentacene.xyz --methods dlpno_ccsd
```

Output: `bench/bench_<seriestag>.csv` (method, molecule, natoms, gansu_s, gansu_status,
gansu_E, orca_s, orca_rt, orca_status, speedup=orca/gansu). Processes increasing size;
stops at the first OOM/timeout/crash per package (`--no-stop` to run all). `dlpno_steom`
GANSU energy column = lowest excitation (eV); others = correlation/total energy (Ha).
Native EOM is default-ON in GANSU now, so no env is needed.
