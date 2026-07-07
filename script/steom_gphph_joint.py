#!/usr/bin/env python3
"""g_phph IP route (projection, dressed): JOINT multi-system lstsq.

The single-system dressed fit is exact (resid ~1e-13) but greedy can't isolate a
sparse formula (collinearity).  A PHYSICAL term has the SAME coefficient in every
system; a collinear artifact is only degenerate in one.  So stack H2O(4,2),
hexatriene(3,3), asymmetric-H6(?,?) with ONE shared coeff per candidate einsum
string -> the physical sparse set survives, artifacts get killed.

Run: wsl python3 script/steom_gphph_joint.py
"""
import os, sys, tempfile, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_cas_verify as V
import steom_cfour_weff as CW
from steom_fockspace_ref import get_active_data
import steom_gphph_spatial_dressed as SD


def cand_dict(sip, bar_h, target, nocc, nvir):
    """Return {name: vec} and target vec [m,a,c,i]. name = einsum-pattern string
    (dim-independent) so it matches across systems."""
    W = {"Wooov": bar_h["Wooov"], "Wvovv": bar_h["Wvovv"], "Wovvo": bar_h["Wovvo"],
         "Wovov": bar_h["Wovov"], "Wovoo": bar_h["Wovoo"], "Woooo": bar_h["Woooo"],
         "eri_ovov": bar_h["eri_ovov"], "Fov": bar_h["Fov"]}
    if "Wvvvo" in bar_h:
        W["Wvvvo"] = bar_h["Wvvvo"]
    o, v = nocc, nvir
    C = {}
    def add(es, arrs, name):
        try:
            T = np.einsum(es, *arrs, optimize=True)
        except Exception:
            return
        if T.ndim != 4:
            return
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nocc, nvir, nvir, nocc):
                continue
            vv = Tp.reshape(-1)
            if np.linalg.norm(vv) < 1e-9:
                continue
            C[f"{name}:{es}:p{perm}"] = vv
    sip_axes = [(0, o), (1, o), (2, v)]        # s_ip[m][i,k,c] axes 0=i,1=k,2=c
    for wn, Wt in W.items():
        if wn == "Fov":
            continue
        wsh = Wt.shape
        for (s1, k1), (s2, k2) in itertools.combinations(sip_axes, 2):
            for w1, w2 in itertools.permutations(range(len(wsh)), 2):
                if wsh[w1] != k1 or wsh[w2] != k2:
                    continue
                sl = ['m', 'A', 'B', 'C']; wl = [chr(ord('P') + t) for t in range(len(wsh))]
                sl[1 + s1] = 'X'; wl[w1] = 'X'; sl[1 + s2] = 'Y'; wl[w2] = 'Y'
                free = [c for c in sl[1:] if c not in ('X', 'Y')] + \
                       [wl[t] for t in range(len(wsh)) if t not in (w1, w2)]
                es = f"{''.join(sl)},{''.join(wl)}->m{''.join(free)}"
                add(es, [sip, Wt], wn)
    # 2-index (1-body-like) intermediates: Fov(o,v), Loo(o,o), Lvv(v,v)
    twobody = [("Fov", W["Fov"])]
    for nm in ("Loo", "Lvv"):
        if nm in bar_h:
            twobody.append((nm, bar_h[nm]))
    for nm, M in twobody:
        for (s1, k1) in sip_axes:
            for w1 in range(2):
                if M.shape[w1] != k1:
                    continue
                sl = ['m', 'A', 'B', 'C']; wl = ['P', 'Q']
                sl[1 + s1] = 'X'; wl[w1] = 'X'
                free = [c for c in sl[1:] if c != 'X'] + [wl[t] for t in range(2) if t != w1]
                es = f"{''.join(sl)},{''.join(wl)}->m{''.join(free)}"
                add(es, [sip, M], nm)
    # NO within-system dedup: names are deterministic across systems so they match;
    # the joint fit across differing dims breaks the within-system collinearity.
    tv = target.transpose(1, 0, 2, 3).reshape(-1)
    return C, tv


def load_system(xyz=None, atom=None, active=None, ncore=0):
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
        d = CW.load(xyz, "sto-3g", ncore)
    elif active is not None:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
        xyzf = os.path.join(tempfile.gettempdir(), "gj.xyz")
        lines = [a.strip() for a in atom.split(";")]
        open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
        d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    else:  # atom + ncore (no CAS window)
        data = get_active_data(atom=atom, basis="sto-3g", ncore=ncore)
        xyzf = os.path.join(tempfile.gettempdir(), "gj.xyz")
        lines = [a.strip() for a in atom.split(";")]
        open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
        d = CW.load(xyzf, "sto-3g", ncore, atom=atom)
    nocc, nvir = data["nocc"], data["nvir"]
    base, opp_IP, sip_sp = SD.oracle_gphph_routes(data, atom, active)
    sip = np.stack([sip_sp[m] for m in range(nocc)], 0)
    C, tv = cand_dict(sip, d["bar"], opp_IP, nocc, nvir)
    return C, tv


def main():
    # ALL C1-asymmetric (no symmetry => no exact IP-root degeneracy => clean s_ip,
    # so a UNIVERSAL formula must fit jointly) AND nocc != nvir (identical candidate
    # string sets).  Symmetric molecules (NH3 C3v) mix degenerate IP roots and
    # contaminate the joint => avoid.
    systems = []
    defs = [
        ("H2O distort A FC1 (4,2)", dict(atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=1)),
        ("H2O distort B FC1 (4,2)", dict(atom="O 0 0 0; H 1.05 0.10 0.22; H -0.50 0.75 0.13", ncore=1)),
        ("H2O distort C FC1 (4,2)", dict(atom="O 0 0 0; H 0.88 0.45 -0.20; H -0.20 0.96 0.30", ncore=1)),
        ("H2O distort A full (5,2)", dict(atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=0)),
        ("H2O distort B full (5,2)", dict(atom="O 0 0 0; H 1.05 0.10 0.22; H -0.50 0.75 0.13", ncore=0)),
        ("NH3 distort-C1 FC1 (4,3)", dict(atom="N 0 0 0; H 0.95 0.05 0.30; H -0.45 0.83 0.28; H -0.52 -0.78 0.35", ncore=1)),
    ]
    for lbl, kw in defs:
        print(f"loading {lbl}...")
        try:
            systems.append(load_system(**kw))
        except Exception as e:
            print(f"   SKIP ({e})")

    # union of candidate names present in ALL systems (physical must appear in all)
    common = set(systems[0][0])
    for C, _ in systems[1:]:
        common &= set(C)
    common = sorted(common)
    print(f"\ncommon candidates across all systems: {len(common)}")
    # build joint design
    rows = []; tgt = []
    for C, tv in systems:
        A = np.stack([C[n] for n in common], 1)
        rows.append(A); tgt.append(tv)
    Ajoint = np.concatenate(rows, 0); tjoint = np.concatenate(tgt, 0)
    tn = np.linalg.norm(tjoint)
    co, *_ = np.linalg.lstsq(Ajoint, tjoint, rcond=None)
    resid = np.linalg.norm(Ajoint @ co - tjoint) / tn
    print(f"JOINT full lstsq resid={resid:.3e}  (ncand={len(common)})")
    # rank by |coeff|*meancolnorm
    order = sorted(range(len(common)), key=lambda i: -abs(co[i]) * np.linalg.norm(Ajoint[:, i]))
    print("  top joint candidates (coeff):")
    for i in order[:14]:
        print(f"    {co[i]:+.4f}  {common[i]}")
    # greedy sparse on joint
    print("\n  greedy sparse (joint):")
    chosen = []; res = tjoint.copy()
    for step in range(8):
        best = min(range(len(common)),
                   key=lambda i: np.linalg.norm(res - (Ajoint[:, i] @ res / (Ajoint[:, i] @ Ajoint[:, i] + 1e-30)) * Ajoint[:, i]))
        if best in chosen: break
        chosen.append(best)
        Ac = Ajoint[:, chosen]
        cc, *_ = np.linalg.lstsq(Ac, tjoint, rcond=None)
        res = tjoint - Ac @ cc
        print(f"    step{step+1}: resid={np.linalg.norm(res)/tn:.3e}  " +
              "  ".join(f"{c:+.3f}[{common[j]}]" for c, j in zip(cc, chosen)))
        if np.linalg.norm(res) / tn < 5e-3: break


if __name__ == "__main__":
    main()
