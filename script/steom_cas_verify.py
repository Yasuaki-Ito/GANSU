#!/usr/bin/env python3
"""EE-base fix on REAL pi-conjugated config-mixed systems (naphthalene analogs).

H4/H6 are STEOM-degenerate (S=0) or s-route-polluted. This tests the base fix on
all-trans polyene pi-CAS(2n,2n) - the actual acene-La/Lb physics - comparing:
   oracle full plain STEOM  (= ORCA-equivalent target)
   GANSU analytic  [shipped base]  vs  [EE base]
on the LOW valence roots. Run: wsl python3 script/steom_cas_verify.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile
import numpy as np
sys.path.insert(0, "script")
from scipy.linalg import expm
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
import steom_cfour_weff as CW
from pyscf_steom_feff_reference import build_g_canonical_full
Ha = 27.211386245988


def polyene(n, distort=0.0):
    """all-trans C_n H_{n+2} planar polyene in xy-plane (zigzag). distort>0 adds a fixed
    asymmetric per-atom ramp to lift accidental symmetry degeneracies (clean root-following)."""
    dx, dy = 1.24, 0.70
    at = []
    cx = []
    for k in range(n):
        x = k * dx + distort * k * 0.03          # asymmetric bond-length ramp
        y = dy * (k % 2) + distort * (0.05 if k == 0 else 0.0)
        at.append(f"C {x:.4f} {y:.4f} 0.0"); cx.append((x, y))
    # H: each carbon gets one in-plane H pointing away from the chain (outward y), termini get 2
    for k in range(n):
        x, y = cx[k]
        yo = 1.08 if (k % 2 == 0) else -1.08   # outward
        at.append(f"H {x:.4f} {y + yo:.4f} 0.0")
    # terminal extra H (along -x / +x)
    x0, y0 = cx[0]; at.append(f"H {x0-1.02:.4f} {y0:.4f} 0.0")
    xn, yn = cx[-1]; at.append(f"H {xn+1.02:.4f} {yn:.4f} 0.0")
    return "; ".join(at)


def detect_pi(atom, basis, n_pi_occ, n_pi_vir):
    from pyscf import gto, scf
    mol = gto.M(atom=atom, basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
    labels = mol.cart_labels()
    pz = [k for k, lb in enumerate(labels) if lb.strip().endswith(" z") or lb.strip().endswith("pz")]
    C = mf.mo_coeff; S = mf.get_ovlp(); SC = S @ C
    wpz = np.einsum("ki,ki->i", C[pz, :], SC[pz, :])
    nocc = mol.nelectron // 2
    occ = sorted(range(nocc), key=lambda p: -wpz[p])[:n_pi_occ]
    vir = sorted(range(nocc, C.shape[1]), key=lambda p: -wpz[p])[:n_pi_vir]
    return sorted(occ + vir), wpz


def _eig(G, shift=0.0):
    w, vr = np.linalg.eig(G)
    o = np.argsort(w.real)
    return (w[o].real - shift) * Ha, vr[:, o]


def oracle_plain(atom, basis, active):
    data = get_active_data(atom=atom, basis=basis, active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    ns = sum(np.linalg.norm(v) for v in sIP.values()) + sum(np.linalg.norm(v) for v in sEA.values())
    S = build_S(data, dets, index, sIP, sEA)
    Gs, _ = project_1h1p(data, dets, index, expm(S) @ Hbar @ expm(-S))
    e, V = _eig(Gs, E_N)
    return e, V, nocc, nvir, ns


def gansu_analytic(atom, basis, active, want_vec=False):
    xyzf = os.path.join(tempfile.gettempdir(), "cas.xyz")
    lines = [a.strip() for a in atom.split(";")]
    open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
    d = CW.load(xyzf, basis, 0, atom=atom, active=active)
    Ga, *_ = build_g_canonical_full(d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    e, V = _eig(Ga)
    return (e, V) if want_vec else e


def _pr(v):
    p = np.abs(v) ** 2; s = p.sum()
    return float(1.0 / np.sum((p / s) ** 2)) if s > 0 else 0.0


def _follow(vref, V):
    ov = np.abs(vref.conj() @ V) / (np.linalg.norm(vref) * np.linalg.norm(V, axis=0) + 1e-30)
    return int(np.argmax(ov))


def run(label, n, n_pi):
    atom = polyene(n, distort=0.0)
    active, wpz = detect_pi(atom, "sto-3g", n_pi, n_pi)
    print(f"\n=== {label}  pi-CAS({2*n_pi},{2*n_pi}) active={active} pz={np.round(wpz[active],2)}")
    e_or, V_or, nocc, nvir, ns = oracle_plain(atom, "sto-3g", active)
    print(f"    ||sIP+sEA||={ns:.3f}")
    # GANSU variants (root-followed to oracle): shipped / +EE-base / +EE-base+EA-route
    def variant(**env):
        for k, v in env.items(): os.environ[k] = v
        e, V = gansu_analytic(atom, "sto-3g", active, want_vec=True)
        for k in env: os.environ.pop(k, None)
        return e, V
    e_sh, V_sh = variant()
    e_ee, V_ee = variant(STEOM_EE_BASE="1")
    e_ea, V_ea = variant(STEOM_EE_BASE="1", STEOM_TEST_GPHHP_EA="2")
    print(f"    {'root':>4} {'PR':>5} {'oracle':>8} | {'shipped':>8} {'+EEbase':>8} {'+EE+EA':>8} "
          f"| {'g_sh':>7} {'g_ee':>7} {'g_all':>7}")
    for i in range(len(e_or)):
        vo = V_or[:, i]; pr = _pr(vo)
        js, je, ja = _follow(vo, V_sh), _follow(vo, V_ee), _follow(vo, V_ea)
        mix = "*" if pr >= 1.5 else " "
        print(f"    {i:>4}{mix} {pr:5.2f} {e_or[i]:8.3f} | {e_sh[js]:8.3f} {e_ee[je]:8.3f} {e_ea[ja]:8.3f} "
              f"| {e_sh[js]-e_or[i]:+7.3f} {e_ee[je]-e_or[i]:+7.3f} {e_ea[ja]-e_or[i]:+7.3f}")


def main():
    run("hexatriene", 6, 3)    # pi-CAS(6,6) - config-mixed valence + S!=0 (naphthalene analog)
    if "but" in sys.argv:
        run("butadiene", 4, 2)


if __name__ == "__main__":
    main()
