#!/usr/bin/env python3
"""Systematic SO enumeration to read off the g_phhp EA s-route formula.

g_phhp = Mc (cross-spin 1h1p, <i_a^a|G|j_b^b>). EA route = Mc(Hbar+[S_ea,Hbar]) - Mc(base),
linear in s_EA, all in the det oracle's SO convention (no SO/spatial confound). Key spin
fact (steom_so_fit): for a beta root E, sea[E,J,A,B] is nonzero only with J alpha, (A,B)
mixed => g_phhp is a CROSS-SPIN (direct-Coulomb) object, structurally unlike the campaign's
same-spin u_bkje (which is why it was wrong).

We enumerate candidate SO contractions sea_so * X (X in {direct-D, antisym-g, f}) with the
sea root E fixed to the output b-slot, slice the alpha-bra/beta-ket block [i,a,j,b], and
lstsq-fit the target. A low residual with clean coefficients = the formula.
Run: wsl python3 script/steom_so_fit2.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, itertools
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, occ_so, vir_so, so_index)
from steom_so_derive import det_singles_block, build_direct, build_so_integrals


def run(atom, active=None):
    data = (get_active_data(atom=atom, basis="sto-3g", active=active) if active
            else get_active_data(atom=atom, basis="sto-3g", ncore=0))
    nocc, nvir, nact = data["nocc"], data["nvir"], data["nact"]
    nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    S_ea = build_S(data, dets, index, zIP, sEA)
    comm = S_ea @ Hbar - Hbar @ S_ea
    _, Mc_base, _ = det_singles_block(data, dets, index, Hbar)
    _, Mc_lin, _ = det_singles_block(data, dets, index, Hbar + comm)
    target = Mc_lin - Mc_base                       # [i,a,j,b], cross-spin

    g, f = build_so_integrals(data)
    D = build_direct(data)
    sea = np.zeros((nso, nso, nso, nso))            # sea[E,J,A,B]
    for e in sEA:
        sea[e] = sEA[e]

    # SO index sets by spin/space
    oa = [so_index(i, 0, nact) for i in range(nocc)]   # alpha occ (== i)
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]  # alpha vir
    ob = [so_index(j, 1, nact) for j in range(nocc)]
    vb = [so_index(b + nocc, 1, nact) for b in range(nvir)]

    # cross-spin slice of a full SO 4-tensor T[P,Q,R,S] -> [i,a,j,b] (i_a,a_a,j_b,b_b)
    def slc(T):
        return T[np.ix_(oa, va, ob, vb)]

    # Build candidates: contract sea (root axis 0 = E) with a 2-electron integral X
    # (D or g) over 2 of sea's {J,A,B} axes, leaving output [outi,outa,outj, E].
    # einsum: sea 'e' + 3 letters for (J,A,B); X 4 letters; shared = contracted.
    # Output order fixed to 'p q r e' then we relabel to (i,a,j,b).
    cand = {}
    seaax = ['j', 'a', 'b']          # labels for sea's J,A,B axes
    xax = ['p', 'q', 'r', 's']
    for Xname, X in [("D", D), ("g", g)]:
        # choose which 2 sea axes contract, and which 2 X axes they pair to
        for sea_contract in itertools.combinations(range(3), 2):
            sea_free = [k for k in range(3) if k not in sea_contract][0]
            for xc in itertools.permutations(range(4), 2):   # X axes paired to the 2 sea-contracted
                # assign shared dummy letters
                lab_sea = list(seaax)
                lab_x = list(xax)
                for si, xi in zip(sea_contract, xc):
                    lab_x[xi] = lab_sea[si]                  # share letter
                # output = X's 2 free axes + sea's free axis; einsum output 'e'+...
                x_free = [k for k in range(4) if k not in xc]
                out = [lab_x[x_free[0]], lab_x[x_free[1]], lab_sea[sea_free], 'e']
                es = f"e{''.join(lab_sea)},{''.join(lab_x)}->{''.join(out)}"
                try:
                    T = np.einsum(es, sea, X, optimize=True)
                except Exception:
                    continue
                # T is [out0,out1,out2,E]; map to [i,a,j,b] by trying output-axis->(i,a,j)
                for perm in itertools.permutations(range(3)):
                    Tp = np.transpose(T, (perm[0], perm[1], perm[2], 3))
                    c = slc(Tp)
                    if np.linalg.norm(c) < 1e-9:
                        continue
                    key = f"{Xname}:{es}:perm{perm}"
                    cand[key] = c.ravel()
    # f-based: sea contract 1 axis with f, leaves 2 sea + 1 f = 3 outputs
    for fc in itertools.product(range(3), range(2)):
        si, fi = fc
        lab_sea = list(seaax); lab_f = ['p', 'q']
        lab_f[fi] = lab_sea[si]
        sea_free = [k for k in range(3) if k != si]
        f_free = [k for k in range(2) if k != fi][0]
        out = [lab_f[f_free], lab_sea[sea_free[0]], lab_sea[sea_free[1]], 'e']
        es = f"e{''.join(lab_sea)},{''.join(lab_f)}->{''.join(out)}"
        T = np.einsum(es, sea, f, optimize=True)
        for perm in itertools.permutations(range(3)):
            Tp = np.transpose(T, (perm[0], perm[1], perm[2], 3))
            c = slc(Tp)
            if np.linalg.norm(c) < 1e-9:
                continue
            cand[f"f:{es}:perm{perm}"] = c.ravel()

    # dedup identical columns (many perms give the same tensor)
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v, 10))
        if key not in uniq:
            uniq[key] = n
    names = list(uniq.values())
    tv = target.ravel(); tn = np.linalg.norm(tv)
    print(f"H6 EA-route SO enum: {len(cand)} raw -> {len(names)} unique, ||target||={tn:.4f}")
    # single best candidate
    best = min(names, key=lambda n: np.linalg.norm(target.ravel() - (cand[n]@tv/(cand[n]@cand[n]))*cand[n]))
    cb = cand[best]@tv/(cand[best]@cand[best])
    print(f"  BEST single: rel-resid={np.linalg.norm(tv-cb*cand[best])/tn:.3e}  c={cb:+.4f}")
    print(f"    {best}")
    # greedy sparse formula (print coefficients each step)
    chosen = []; res = tv.copy()
    for step in range(5):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n]@res/(cand[n]@cand[n]))*cand[n]))
        if n in chosen:
            break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tv, rcond=None)
        res = tv - Ac @ co
        print(f"  step {step+1}: rel-resid={np.linalg.norm(res)/tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"      {cc:+.4f}  {m}")
        if np.linalg.norm(res)/tn < 5e-3:
            break


def main():
    print("===== H6 rect ladder =====")
    run("; ".join(f"H {2.0*(k%2)} {1.4*(k//2)} 0" for k in range(6)))
    print("\n===== hexatriene pi-CAS(6,6) =====")
    import steom_cas_verify as V
    atom = V.polyene(6); active, _ = V.detect_pi(atom, "sto-3g", 3, 3)
    run(atom, active=active)


if __name__ == "__main__":
    main()
