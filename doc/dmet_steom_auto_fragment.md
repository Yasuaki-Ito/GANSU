# CIS-guided automatic fragmentation for DMET-STEOM

Excitation-driven, real-space fragment selection for DMET-STEOM. Instead of
specifying the chromophore by hand (`--dmet_fragments "{...}"`), a preprocessing
CIS computes where each excitation lives and the chromophore atoms are selected
automatically from the natural-transition-orbital (NTO) per-atom weights.

Enable with `--dmet_steom_auto_fragment 1` on the `--post_hf_method dmet_steom`
path. RI is required (`--eri_method ri -ag <aux>`), as for all DMET-STEOM. An
explicit `--dmet_fragments` always takes priority (auto is skipped).

## Quick start

```bash
# Auto-select the chromophore and run DMET-STEOM
./gansu -x ../xyz/large_molecular/Doxorubicin.xyz -g cc-pvdz \
        --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
        --post_hf_method dmet_steom --n_excited_states 5 \
        --dmet_steom_auto_fragment 1 --dmet_cluster_solver dlpno \
        --frozen_core auto --num_gpus 4 --initial_guess sad
```

## How it works

1. **Preprocess CIS-NTO** on the whole system (hoisted before fragment selection).
2. **Score every atom** by its occupation-weighted hole+particle NTO Löwdin
   population — where the excitation's density sits.
3. **Greedily select** atoms above the per-atom floor until the cumulative
   coverage target is met (capped by the cluster orbital budget).
4. **Self-verify** with the bath-sufficiency gauge; optionally **expand** the
   fragment (Phase B) if the bare Schmidt bath does not capture the excitation.
5. Solve DMET-STEOM on the selected cluster (the rest of the molecule enters via
   the Schmidt bath, exactly as with a manual fragment).

The cluster is three layers: **fragment atoms + Schmidt bath + NTO-augmented
bath**. Selecting the atoms sets *where*; the bath and NTO augmentation supply
the environment entanglement and the particle-space adequacy.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `dmet_steom_auto_fragment` | 0 | Enable auto extraction |
| `dmet_steom_auto_coverage` | 0.92 | Cumulative per-atom NTO coverage target |
| `dmet_steom_auto_atom_floor` | 0.01 | Per-atom score floor (below → not a candidate) |
| `dmet_steom_auto_budget` | 0 | Cluster orbital budget (est. n_emb); 0 = auto (canonical 460 / dlpno 700) |
| `dmet_steom_auto_include_h` | 0 | Attach bonded H to selected heavy atoms (default off: env C–H σ is covered by the bath) |
| `dmet_steom_auto_n_cis` | 0 | CIS states for the extraction; 0 = auto |
| `dmet_steom_auto_focus_states` | 0 | Build the NTO from only the lowest N CIS roots (avoids n→π* contamination of the fragment); 0 = average all |
| `dmet_steom_auto_max_expand` | 1 | Max gauge-triggered fragment-expansion rounds (Phase B) |
| `dmet_steom_auto_json` | "" | Write per-state per-atom localization JSON (for the grouping driver) |
| `dmet_steom_auto_xyz` | "" | Write the selected fragment geometry as `.xyz` |

## Reading the output

- **`selected N atom(s) (coverage=…)`** — the chosen chromophore and how much of
  the excitation it captures.
- **`[auto-frag size] → RIGHT-SIZED / BUDGET-CAPPED / OVER-SELECTED`** — whether a
  large cluster is *required* by the excitation (accept it), was *capped by budget*
  (under-converged; raise the budget / use `--dmet_cluster_solver dlpno`), or is
  *over-selected* (tighten coverage/floor to shrink with little accuracy loss).
- **`bath … (uncaptured=…; virtual-space tail uncaptured=…)`** — the
  bath-sufficiency gauge. A high tail relative to the active value flags a
  truncated virtual (particle) space; both low with a still-poor energy indicates
  the mean-field embedding limit of a fully delocalized excitation (that molecule
  is outside DMET's domain — don't fragment it).
- **Warnings** — floor-sensitive selection, delocalized (coverage unreachable),
  or `>3 disconnected regions` (a mix of spatially distinct excitations → use the
  grouping driver below). Two regions (donor…acceptor) is a *note*, not a warning.

## Multi-state grouping (multiple chromophores / job splitting)

When a molecule has spatially distinct excitations, one state-averaged fragment
scatters. Emit the per-state JSON and let the driver split the states into
region-specific fragments, each a separate job:

```bash
./gansu … --dmet_steom_auto_json states.json --dmet_steom_auto_max_expand 0
python3 ../script/dmet_steom_group_states.py --json states.json \
        --outdir groups --sim 0.5 --coverage 0.92 \
        --gansu-args "-g cc-pvdz --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
                      --frozen_core auto --num_gpus 4 --n_excited_states 5"
```

The driver clusters the states by cosine similarity of their per-atom vectors,
writes one `fragment_k.xyz` per group, and emits `groups/jobs.sh` (one DMET-STEOM
job per group). It also reports **charge-transfer states** — where the hole and
particle sit on disjoint regions — as `donor … -> acceptor …`.

## Worked examples

- **Doxorubicin (localized chromophore)**: auto selects the anthraquinone
  (matches the hand-tuned fragment); lowest excitation within ~0.05 eV of the
  manual-fragment reference.
- **Naphthalene (delocalized ππ*)**: auto selects all 10 carbons and is flagged
  RIGHT-SIZED — the excitation spans the whole conjugated backbone, so DMET
  fragmentation buys little (the honest signal).
- **Reichardt's dye (charge transfer)**: auto selects both the phenolate donor
  and the pyridinium acceptor; the driver labels the CT states donor→acceptor.

## Cluster excited-state solver: STEOM or ADC(2)

The embedded cluster can be solved by either STEOM-CCSD (default) or **ADC(2)**,
selected with `--dmet_excited_method {steom|adc2}`. The auto-fragment / bath /
gauge machinery is shared; only the final cluster solver differs.

```bash
# DMET + ADC(2) on the auto-extracted chromophore (cheaper than STEOM, triplets OK)
./gansu -x mol.xyz -g cc-pvdz --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
        --post_hf_method dmet_steom --dmet_excited_method adc2 \
        --dmet_steom_auto_fragment 1 --n_excited_states 5
```

ADC(2) on the cluster uses the physical (un-level-shifted) cluster ε. When the
cluster ERIs come from RI (`--eri_method ri`), the four ADC(2) sub-blocks
(ovov/vvov/ooov/oovv) are pulled from the cluster `B_mo` via `mo_eri_block_into`
(**RI-block path**) instead of materialising the dense `n_emb⁴` MO-ERI — which is
247 GB at `n_emb≈427` and OOMs. Verified: whole-molecule DMET-ADC(2) equals
standalone ADC(2) bit-for-bit on both the dense and RI-block paths.

Solver mode follows `--adc2_solver`
(auto/schur_static/schur_omega/schur_davidson/full); triplets via
`--spin_type triplet`. **`auto` is size-aware**: the ADC(2) doubles block is
diagonal, so it folds into a symmetric singles×singles `M_eff(ω)` when the full
Davidson subspace won't fit (`full` needs `O(singles²)` explicit doubles vectors —
~1.6 TB at `n_emb=427`). Tiering: `full` when it fits; `schur_omega` (exact
ω-iterated dense fold) up to ~10⁴ singles; **`schur_davidson`** beyond — a
matrix-free symmetric Davidson on `M_eff(ω=0)` via the operator's kernel `apply()`
(no dense `M_eff`, no `O(singles³)` tridiagonalisation; solves buffer roots to
avoid root-skipping in near-degenerate manifolds; schur_static accuracy,
~0.005–0.02 Ha; `GANSU_ADC2_OMEGA_REFINE=1` adds warm-started per-root ω
self-consistency). The dense Schur solvers use a partial eigensolver (`syevdx`,
lowest `n_states`). Reference scale: dox group-0 (`n_emb=427`, singles=29640)
runs end-to-end in ~63 min on 4×H200 under `schur_davidson`; the dense
diagonalisation route did not finish in 2 h. Debug: `GANSU_ADC2_DEBUG=1` prints
per-stage asum probes of the M11 build (catches non-finite blocks immediately).

## Limitations

- **RI only.** `focus_states`, the JSON, and `hole/part` scores need the stashed
  CIS amplitudes (RI path). Without them those features no-op with a message; the
  basic auto extraction still runs.
- **Delocalized excitations.** If the excitation spans the whole conjugated
  system (naphthalene), DMET fragmentation cannot improve on full STEOM — the
  gauge/size-class report this rather than hiding it.
- **Charge-transfer energies.** The CT *chromophore* is detected reliably, but CT
  excitation energies inherit the usual CIS/STEOM CT difficulties.

## Regression

After rebuilding the CIS-NTO / auto path, run
`bash ../script/dmet_steom_auto_regression.sh` from the build directory
(naphthalene, ~min): checks default-off byte-identity, the 10-carbon selection,
and the gauge / size-classification output against fixed anchors.
