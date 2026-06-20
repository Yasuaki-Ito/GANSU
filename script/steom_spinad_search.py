#!/usr/bin/env python3
"""
Experimentally determine the correct spin-adaptation (.spinad / tilde) pattern
of the STEOM W^eff phhp (g_phhp) and phph (g_phph) dressing intermediates by
LINEAR LEAST-SQUARES against the EXACT effective Hamiltonian elements.

Rationale
---------
At complete active space STEOM(G) == EOM-EE-CCSD, so the singlet/triplet EOM
roots give the EXACT g_phph / g_phhp tensors element-wise on the (i!=j, a!=b)
off-diagonal block (where F^eff does not enter):
    g_phph[a,j,b,i] = -G_t_exact[ia,jb]
    g_phhp[b,j,i,a] = (G_s_exact - G_t_exact)/2 [ia,jb]

The dressing  (exact - base)  is LINEAR in a set of candidate contraction
tensors T_k (each = one bar-H intermediate contracted with an s amplitude in a
specific index pattern).  Index analysis shows the S^IP route output [b,j,c]
can come ONLY from {Fov, Wovoo, Wvovv}, and the S^EA route ONLY from
{Fov, Wvovv, Wooov} -- the same three families already in the code.  So the
true formula is a clean-coefficient combination of the DIRECT and HOLE/PARTICLE
SWAP variants of those families.  Stacking many off-diagonal elements over
several small molecules and solving  M c = target  reveals the coefficients;
clean rationals (0, +-1, +-2, +-1/3 ...) with ~0 residual = the formula.

Usage:
  wsl python3 script/steom_spinad_search.py            # default molecule set
  wsl python3 script/steom_spinad_search.py phhp        # which block
"""
import sys
import numpy as np
sys.path.insert(0, "script")
from pyscf_steom_feff_reference import read_xyz, build_bar_h
from steom_full_active_test import principal_roots_ip, principal_roots_ea
from pyscf_steom_feff_reference import build_normalized_s

np.set_printoptions(precision=5, suppress=True, linewidth=160)

# molecules: (xyz, basis, ncore) chosen so every correlated occ has a clean
# principal IP (well-conditioned R1mat) -> STEOM==EOM exactly.
DEFAULT_MOLS = [
    ("xyz/H2O.xyz", "sto-3g", 2),
    ("xyz/CH4.xyz", "sto-3g", 1),
    ("xyz/HF.xyz",  "sto-3g", 1),
]


def setup(xyz, basis, ncore):
    """Return everything needed for one molecule: bar_h, s_IP, s_EA, exact
    g_phph/g_phhp targets, base tensors, and dims."""
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd
    mol = gto.M(atom=read_xyz(xyz), basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    mycc = cc.CCSD(mf, frozen=ncore); mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    t1, t2 = mycc.t1, mycc.t2
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc
    mo_c = mf.mo_coeff[:, ncore:]; moe_c = mf.mo_energy[ncore:]
    eri_mo = ao2mo.kernel(mol, mo_c, compact=False).reshape(nmo, nmo, nmo, nmo)
    f_oo = np.diag(moe_c[:nocc]); f_vv = np.diag(moe_c[nocc:])
    bar_h = build_bar_h(eri_mo, t1, t2, f_oo, f_vv, nocc, nvir)
    dim = nocc * nvir

    # exact singlet + triplet EOM effective singles Hamiltonians
    eom = eom_rccsd.EOMEESinglet(mycc)
    e_s, v_s = eom.kernel(nroots=dim)
    e_s = np.atleast_1d(np.asarray(e_s))
    R1 = np.zeros((dim, dim))
    for n in range(dim):
        r1, _ = eom.vector_to_amplitudes(v_s[n]); R1[:, n] = np.asarray(r1).reshape(dim)
    G_s = R1 @ np.diag(e_s[:dim]) @ np.linalg.inv(R1)

    eomT = eom_rccsd.EOMEETriplet(mycc)
    e_t, v_t = eomT.kernel(nroots=dim)
    e_t = np.atleast_1d(np.asarray(e_t))
    R1t = np.zeros((dim, dim))
    for n in range(dim):
        amp = eomT.vector_to_amplitudes(v_t[n])
        r1t = amp[0] if isinstance(amp, (tuple, list)) else amp
        R1t[:, n] = np.asarray(r1t).reshape(dim)
    G_t = R1t @ np.diag(e_t[:dim]) @ np.linalg.inv(R1t)

    # exact tensors:  exact_phph[a,j,b,i] = -G_t[ia,jb];
    #                 exact_phhp[b,j,i,a] = (G_s-G_t)/2 [ia,jb]
    exact_phph = np.zeros((nvir, nocc, nvir, nocc))
    exact_phhp = np.zeros((nvir, nocc, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    exact_phph[a, j, b, i] = -G_t[r, c]
                    exact_phhp[b, j, i, a] = (G_s[r, c] - G_t[r, c]) / 2.0

    # principal roots -> normalized s
    r1_ip, r2_ip, _, occ_idx, _ = principal_roots_ip(mycc, nocc, nvir)
    r1_ea, r2_ea, _, vir_idx, _ = principal_roots_ea(mycc, nocc, nvir)
    s_IP, s_EA = build_normalized_s(r2_ip, r2_ea, r1_ip, r1_ea,
                                    occ_idx, vir_idx, nocc, nvir)
    return dict(bar_h=bar_h, s_IP=s_IP, s_EA=s_EA, nocc=nocc, nvir=nvir,
                occ_idx=occ_idx, vir_idx=vir_idx,
                exact_phph=exact_phph, exact_phhp=exact_phhp)


# ----------------------------------------------------------------------
# Candidate contraction generators for g_phhp[b,k,j,c]  (base = Wovvo[k,b,c,j])
# Each generator returns a full [nvir,nocc,nocc,nvir] tensor, scattered over the
# active-root slots exactly like build_g_canonical_full.
# ----------------------------------------------------------------------
def scatter_IP(per_m_bjc, occ_idx, nocc, nvir):
    """per_m_bjc[m] is [b,j,c]; place at g_phhp[:, occ_idx[m], :, :]."""
    out = np.zeros((nvir, nocc, nocc, nvir))
    for m, k in enumerate(occ_idx):
        out[:, k, :, :] += per_m_bjc[m]
    return out


def scatter_EA(per_e_bkj, vir_idx, nocc, nvir):
    """per_e_bkj[e] is [b,k,j]; place at g_phhp[:, :, :, vir_idx[e]]."""
    out = np.zeros((nvir, nocc, nocc, nvir))
    for e, c in enumerate(vir_idx):
        out[:, :, :, c] += per_e_bkj[e]
    return out


def phhp_candidates(d):
    """Dict name -> full [b,k,j,c] tensor, for the g_phhp dressing."""
    bar = d["bar_h"]; s_IP = d["s_IP"]; s_EA = d["s_EA"]
    nocc = d["nocc"]; nvir = d["nvir"]; occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    Fov = bar["Fov"]; Wovoo = bar["Wovoo"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]
    n_act_occ = len(s_IP); n_act_vir = len(s_EA)
    cand = {}

    # ---- S^IP route:  s = s_IP[m] = [k,j,b] (occ,occ,vir) -> output [b,j,c] ----
    def IP(builder):
        return scatter_IP([builder(s_IP[m]) for m in range(n_act_occ)], occ_idx, nocc, nvir)
    # Fov family (contract one occ of s with Fov occ; Fov supplies c)
    cand["IP_F_d"]  = IP(lambda s: -np.einsum("kc,kjb->bjc", Fov, s))   # direct (existing)
    cand["IP_F_x"]  = IP(lambda s: -np.einsum("kc,jkb->bjc", Fov, s))   # swap occ
    # Wovoo family (contract two occ of s; Wovoo supplies c and j)
    cand["IP_O_d"]  = IP(lambda s:  np.einsum("kclj,klb->bjc", Wovoo, s))  # existing
    cand["IP_O_x"]  = IP(lambda s:  np.einsum("kclj,lkb->bjc", Wovoo, s))  # swap occ
    # Wvovv family (contract one occ + one vir of s; Wvovv supplies b,c)
    cand["IP_V_d"]  = IP(lambda s: -np.einsum("bkdc,kjd->bjc", Wvovv, s))  # existing
    cand["IP_V_x"]  = IP(lambda s: -np.einsum("bkdc,jkd->bjc", Wvovv, s))  # swap occ
    cand["IP_Vb_d"] = IP(lambda s: -np.einsum("bkcd,kjd->bjc", Wvovv, s))  # swap vir(c,d)
    cand["IP_Vb_x"] = IP(lambda s: -np.einsum("bkcd,jkd->bjc", Wvovv, s))
    # Wooov family (contract two occ of s; Wooov supplies c and j) -- MISSING before
    cand["IP_W_d"]  = IP(lambda s:  np.einsum("kljc,klb->bjc", Wooov, s))
    cand["IP_W_x"]  = IP(lambda s:  np.einsum("kljc,lkb->bjc", Wooov, s))
    cand["IP_Wb_d"] = IP(lambda s:  np.einsum("lkjc,klb->bjc", Wooov, s))
    cand["IP_Wb_x"] = IP(lambda s:  np.einsum("lkjc,lkb->bjc", Wooov, s))

    # ---- S^EA route:  s = s_EA[e] = [j,d,b] (occ,vir,vir) -> output [b,k,j] ----
    def EA(builder):
        return scatter_EA([builder(s_EA[e]) for e in range(n_act_vir)], vir_idx, nocc, nvir)
    # Fov family (contract one vir of s; Fov supplies k)
    cand["EA_F_d"]  = EA(lambda s:  np.einsum("kd,jdb->bkj", Fov, s))   # existing
    cand["EA_F_x"]  = EA(lambda s:  np.einsum("kd,jbd->bkj", Fov, s))   # swap vir
    # Wvovv family (contract two vir of s; Wvovv supplies b and c=e slot... here k)
    cand["EA_V_d"]  = EA(lambda s:  np.einsum("bkdc,jcd->bkj", Wvovv, s))  # existing
    cand["EA_V_x"]  = EA(lambda s:  np.einsum("bkdc,jdc->bkj", Wvovv, s))  # swap vir
    cand["EA_Vb_d"] = EA(lambda s:  np.einsum("bkcd,jcd->bkj", Wvovv, s))
    cand["EA_Vb_x"] = EA(lambda s:  np.einsum("bkcd,jdc->bkj", Wvovv, s))
    # Wooov family (contract one occ + one vir; Wooov supplies k and j... )
    cand["EA_O_d"]  = EA(lambda s: -np.einsum("lkjd,ldb->bkj", Wooov, s))  # existing
    cand["EA_O_x"]  = EA(lambda s: -np.einsum("lkjd,lbd->bkj", Wooov, s))  # swap vir
    # Wovoo family (contract occ+vir of s; Wovoo[k,b,l,j] o v o o supplies k,j) -- MISSING
    Wovoo = bar["Wovoo"]
    cand["EA_P_d"]  = EA(lambda s:  np.einsum("kdlj,ldb->bkj", Wovoo, s))  # contract l(occ),d(vir)
    cand["EA_P_x"]  = EA(lambda s:  np.einsum("kdlj,lbd->bkj", Wovoo, s))  # swap vir
    cand["EA_Pb_d"] = EA(lambda s:  np.einsum("jdlk,ldb->bkj", Wovoo, s))  # swap k<->j out
    cand["EA_Pb_x"] = EA(lambda s:  np.einsum("jdlk,lbd->bkj", Wovoo, s))

    # ---- CROSS route (S^IP . S^EA), placed at [b, occ_idx[m], :, vir_idx[e]] ----
    Wovvo = bar["Wovvo"]; eri_ovov = bar["eri_ovov"]
    # hp intermediates (linear in one s)
    u_ma = np.zeros((n_act_occ, nvir))
    for m in range(n_act_occ):
        st = 2.0 * s_IP[m] - s_IP[m].transpose(1, 0, 2)
        u_ma[m] = -np.einsum("kdal,kld->a", Wovvo, st)
    u_ie = np.zeros((nocc, n_act_vir))
    for e in range(n_act_vir):
        st = 2.0 * s_EA[e] - s_EA[e].transpose(0, 2, 1)
        u_ie[:, e] = np.einsum("idcl,lcd->i", Wovvo, st)
    u_mlid = np.zeros((n_act_occ, nocc, nocc, nvir))
    for m in range(n_act_occ):
        st = 2.0 * s_IP[m] - s_IP[m].transpose(1, 0, 2)
        u_mlid[m] = np.einsum("jbld,ijb->lid", eri_ovov, st) \
                  - np.einsum("lbjd,ijb->lid", eri_ovov, s_IP[m])
    u_klie = np.zeros((nocc, nocc, nocc, n_act_vir))
    for e in range(n_act_vir):
        u_klie[:, :, :, e] = np.einsum("kalb,iab->kli", eri_ovov, s_EA[e])

    def CROSS(builder):
        out = np.zeros((nvir, nocc, nocc, nvir))
        for m in range(n_act_occ):
            for e in range(n_act_vir):
                out[:, occ_idx[m], :, vir_idx[e]] += builder(m, e)
        return out
    cand["X_ma"]  = CROSS(lambda m, e: np.einsum("d,jdb->bj", u_ma[m], s_EA[e]))
    cand["X_ie"]  = CROSS(lambda m, e: np.einsum("k,kjb->bj", u_ie[:, e], s_IP[m]))
    cand["X_kli"] = CROSS(lambda m, e: np.einsum("klj,klb->bj", u_klie[:, :, :, e], s_IP[m]))
    cand["X_mli"] = CROSS(lambda m, e: np.einsum("ljd,ldb->bj", u_mlid[m], s_EA[e]))
    return cand


def base_phhp(d):
    return np.einsum("kbcj->bkjc", d["bar_h"]["Wovvo"]).copy()


def mask_phhp(nocc, nvir):
    """boolean [b,k,j,c]: True where exact target valid (i!=j,a!=b).
    G index: row (i,a)=(j_tensor=j, c=a), col (j,b)=(k_tensor=k, b).
    i!=j -> j != k ;  a!=b -> c != b."""
    m = np.zeros((nvir, nocc, nocc, nvir), bool)
    for b in range(nvir):
        for k in range(nocc):
            for j in range(nocc):
                for c in range(nvir):
                    if j != k and c != b:
                        m[b, k, j, c] = True
    return m


def run(block="phhp", mols=DEFAULT_MOLS):
    assert block == "phhp", "phph TODO"
    # build stacked design matrix + target across molecules
    names = None
    M_rows = []; t_rows = []
    per_mol = []
    for (xyz, basis, ncore) in mols:
        d = setup(xyz, basis, ncore)
        cand = phhp_candidates(d)
        if names is None:
            names = list(cand.keys())
        msk = mask_phhp(d["nocc"], d["nvir"])
        tgt = (d["exact_phhp"] - base_phhp(d))[msk]
        Mi = np.stack([cand[n][msk] for n in names], axis=1)
        M_rows.append(Mi); t_rows.append(tgt)
        per_mol.append((xyz, d, cand, msk))
        print(f"{xyz} {basis} ncore={ncore}: nocc={d['nocc']} nvir={d['nvir']} "
              f"#elems={msk.sum()}")
    M = np.vstack(M_rows); t = np.concatenate(t_rows)
    print(f"\ndesign matrix M = {M.shape}, target = {t.shape}")

    # least squares
    c, res, rank, sv = np.linalg.lstsq(M, t, rcond=None)
    pred = M @ c
    resid = np.linalg.norm(pred - t)
    print(f"\nLSQ rank={rank}/{M.shape[1]}  residual ‖Mc-t‖ = {resid:.3e}  "
          f"(‖t‖={np.linalg.norm(t):.3e})")
    print("coefficients:")
    for n, ci in zip(names, c):
        flag = "  <-- " + ("~0" if abs(ci) < 1e-3 else f"~{round(ci*3)/3:+.4g}?") if abs(ci) > 1e-3 else ""
        print(f"  {n:10s} {ci:+.5f}{flag}")
    return names, c, resid, per_mol


if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "phhp"
    run(block)
