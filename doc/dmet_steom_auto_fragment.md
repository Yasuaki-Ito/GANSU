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
