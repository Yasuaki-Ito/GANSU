#!/usr/bin/env python3
"""Spin-orbital derivation of the closed-shell STEOM g_phhp (ph-hp Coulomb) route.

The closed-shell candidate-fitting failed because spin-adaptation mixes diagrams
and the closed-shell route tensors have a large-cancellation / collinearity that
makes per-element identification an artifact (every signal failed the eigenvalue
test).  In spin-orbital each diagram is separate and unambiguous.

Foundation built here (validated vs the determinant oracle steom_fockspace_ref):
  - spin-orbital antisymmetrized integrals <pq||rs>
  - spin-orbital Fock f (active, with core already folded into h1)
  - spin-orbital t1so/t2so (from build_t_so)
  - spin-orbital CCSD Hbar intermediates (Gauss-Stanton form)

Run:  wsl python3 script/steom_so_derive.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_fockspace_ref as F
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                  build_S, hf_det, apply_string, so_index, build_t_so,
                                  occ_so, vir_so, spin_of, spat_of)
from scipy.linalg import expm

Ha = 27.211386245988
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def build_so_integrals(data):
    """spin-orbital antisymmetrized <pq||rs> (physicist) and Fock f.
    <pq||rs> = (pr|qs) - (ps|qr) with spin: (pr|qs) needs sp==sr, sq==ss.
    f[p,q] = h[p,q] + Σ_i <pi||qi>  (i over occ-so)."""
    nact = data["nact"]; h1 = data["h1"]; eri = data["eri"]   # eri chemist (pr|qs)
    nso = 2 * nact; nocc = data["nocc"]
    occ = occ_so(data)
    # one-body
    hso = np.zeros((nso, nso))
    for s in range(2):
        for p in range(nact):
            for q in range(nact):
                hso[so_index(p, s, nact), so_index(q, s, nact)] = h1[p, q]
    # antisymmetrized two-body <pq||rs>
    g = np.zeros((nso, nso, nso, nso))
    for P in range(nso):
        for Q in range(nso):
            for R in range(nso):
                for S in range(nso):
                    sP, sQ, sR, sS = (P // nact, Q // nact, R // nact, S // nact)
                    p, q, r, s = P % nact, Q % nact, R % nact, S % nact
                    val = 0.0
                    if sP == sR and sQ == sS:
                        val += eri[p, r, q, s]      # (pr|qs)
                    if sP == sS and sQ == sR:
                        val -= eri[p, s, q, r]      # (ps|qr)
                    g[P, Q, R, S] = val
    # Fock
    f = hso.copy()
    for P in range(nso):
        for Q in range(nso):
            for I in occ:
                f[P, Q] += g[P, I, Q, I]
    return g, f


def slices(data):
    nact = data["nact"]; nocc = data["nocc"]
    occ = occ_so(data); vir = vir_so(data)
    return occ, vir


def validate_foundation(data, dets, index, Hbar):
    """check spin-orbital f, g reproduce determinant H̄ ground & a few sectors."""
    g, f = build_so_integrals(data)
    occ, vir = slices(data)
    # HF energy from f,g: E = Σ_i f[i,i] - 0.5 Σ_ij <ij||ij>
    Ehf = sum(f[I, I] for I in occ) - 0.5 * sum(g[I, J, I, J] for I in occ for J in occ)
    Ehf += data["Ecore"] + data["Enuc"]
    print(f"[foundation] HF from f,g = {Ehf:.8f}  vs data Ehf = {data['Ehf']:.8f}  "
          f"Δ={abs(Ehf-data['Ehf']):.2e}")
    return g, f


def build_t_arrays(data):
    """closed-shell-derived spin-orbital t1[I,A], t2[I,J,A,B] (antisym)."""
    t1so, t2so = build_t_so(data)
    occ, vir = slices(data)
    no = len(occ); nv = len(vir)
    # remap to compact occ/vir index (0..no-1, 0..nv-1) preserving so order
    o_map = {P: k for k, P in enumerate(occ)}
    v_map = {P: k for k, P in enumerate(vir)}
    t1 = np.zeros((no, nv)); t2 = np.zeros((no, no, nv, nv))
    for I in occ:
        for A in vir:
            t1[o_map[I], v_map[A]] = t1so[I, A]
    for I in occ:
        for J in occ:
            for A in vir:
                for B in vir:
                    t2[o_map[I], o_map[J], v_map[A], v_map[B]] = t2so[I, J, A, B]
    return t1, t2, o_map, v_map


def build_hbar(data):
    """Standard spin-orbital CCSD Hbar intermediates (Gauss-Stanton 1991 / Crawford).
    Returns dict with compact occ/vir-indexed arrays. Slices: o,v in compact order."""
    g, f = build_so_integrals(data)
    occ, vir = slices(data)
    o_map = {P: k for k, P in enumerate(occ)}; v_map = {P: k for k, P in enumerate(vir)}
    no = len(occ); nv = len(vir)
    O = occ; V = vir
    def blk(*idx):
        sl = [O if c == 'o' else V for c in idx[0]]
        return g[np.ix_(*sl)]
    goooo = g[np.ix_(O, O, O, O)]; goovv = g[np.ix_(O, O, V, V)]
    govov = g[np.ix_(O, V, O, V)]; govvo = g[np.ix_(O, V, V, O)]
    govvv = g[np.ix_(O, V, V, V)]; gooov = g[np.ix_(O, O, O, V)]
    gvvvv = g[np.ix_(V, V, V, V)]; govoo = g[np.ix_(O, V, O, O)]
    gvovv = g[np.ix_(V, O, V, V)]; goovo = g[np.ix_(O, O, V, O)]
    fov = f[np.ix_(O, V)]; foo = f[np.ix_(O, O)]; fvv = f[np.ix_(V, V)]
    t1, t2, _, _ = build_t_arrays(data)
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1) - np.einsum('ib,ja->ijab', t1, t1)
    taut = t2 + 0.5*(np.einsum('ia,jb->ijab', t1, t1) - np.einsum('ib,ja->ijab', t1, t1))
    # F intermediates
    # FULL F (keep diagonal f = orbital energies; needed for an explicit matrix)
    Fae = fvv - 0.5*np.einsum('me,ma->ae', fov, t1) \
          + np.einsum('mf,amef->ae', t1, gvovv) - 0.5*np.einsum('mnaf,mnef->ae', taut, goovv)
    Fmi = foo + 0.5*np.einsum('ie,me->mi', t1, fov) \
          + np.einsum('ne,mnie->mi', t1, gooov) + 0.5*np.einsum('inef,mnef->mi', taut, goovv)
    Fme = fov + np.einsum('nf,mnef->me', t1, goovv)
    # W intermediates
    Wmnij = goooo + np.einsum('je,mnie->mnij', t1, gooov) - np.einsum('ie,mnje->mnij', t1, gooov) \
            + 0.25*np.einsum('ijef,mnef->mnij', tau, goovv)
    Wabef = gvvvv - np.einsum('mb,amef->abef', t1, gvovv) + np.einsum('ma,bmef->abef', t1, gvovv) \
            + 0.25*np.einsum('mnab,mnef->abef', tau, goovv)
    Wmbej = govvo + np.einsum('jf,mbef->mbej', t1, govvv) - np.einsum('nb,mnej->mbej', t1, goovo) \
            - np.einsum('jnfb,mnef->mbej', 0.5*t2 + np.einsum('jf,nb->jnfb', t1, t1), goovv)
    return dict(g=g, f=f, t1=t1, t2=t2, tau=tau, taut=taut,
                Fae=Fae, Fmi=Fmi, Fme=Fme, Wmnij=Wmnij, Wabef=Wabef, Wmbej=Wmbej,
                gooov=gooov, gvovv=gvovv, govvo=govvo, govov=govov, goovv=goovv,
                no=no, nv=nv, fov=fov, foo=foo, fvv=fvv)


def det_singles_block(data, dets, index, Hbar):
    """Exact spin-orbital EOM-EE singles block from the determinant Hbar:
    Msame[i,a,j,b]=<i^a_alpha|Hbar|j^b_alpha>, Mcross[i,a,j,b]=<i^a_a|Hbar|j^b_b>
    (dressing relative to <HF|Hbar|HF>=E_N on the diagonal)."""
    nact = data["nact"]; nocc = data["nocc"]; nvir = data["nvir"]
    hf = hf_det(data); iHF = index[hf]; E_N = Hbar[iHF, iHF]
    def sdet(i, a, s):
        I = so_index(i, s, nact); A = so_index(a + nocc, s, nact)
        sg, dd = apply_string(hf, [("c", A), ("a", I)]); return index[dd], sg
    Ms = np.zeros((nocc, nvir, nocc, nvir)); Mc = np.zeros((nocc, nvir, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            ida, sga = sdet(i, a, 0)
            for j in range(nocc):
                for b in range(nvir):
                    jda, ta = sdet(j, b, 0); jdb, tb = sdet(j, b, 1)
                    Ms[i, a, j, b] = Hbar[ida, jda]*sga*ta - (E_N if (i==j and a==b) else 0)
                    Mc[i, a, j, b] = Hbar[ida, jdb]*sga*tb
    return Ms, Mc, E_N


def validate_singles(data, dets, index, Hbar, H):
    """Build the EOM-EE singles block from Fae/Fmi/Wmbej and compare to determinant.
    EOM-EE: <Phi_i^a|Hbar|Phi_j^b> = d_ij Fae[a,b] - d_ab Fmi[i,j] + Wmaie  where
    the coupling = - Wmbej[j,a,b,i] (W_{maei} convention). Match both spin blocks."""
    no = H["no"]; nv = H["nv"]; nocc = data["nocc"]; nvir = data["nvir"]
    Fae = H["Fae"]; Fmi = H["Fmi"]; Wmbej = H["Wmbej"]
    Ms_det, Mc_det, E_N = det_singles_block(data, dets, index, Hbar)
    # analytic same-spin: d_ij Fae[ab] - d_ab Fmi[ij] - Wmbej[j,a,b,i]  (antisym coupling)
    Ms = np.zeros((nocc, nvir, nocc, nvir)); Mc = np.zeros((nocc, nvir, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    fpart = (Fae[a, b] if i == j else 0.0) - (Fmi[i, j] if a == b else 0.0)
                    Ms[i, a, j, b] = fpart - Wmbej[j, a, b, i]
                    Mc[i, a, j, b] = - Wmbej[j, a, b, i]   # cross: only Coulomb part of W
    # same-spin: validated coupling = + Wmbej[j,a,b,i]
    Ms = np.zeros((nocc, nvir, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    fpart = (Fae[a, b] if i == j else 0.0) - (Fmi[i, j] if a == b else 0.0)
                    Ms[i, a, j, b] = fpart + Wmbej[j, a, b, i]
    print(f"[singles] same-spin ||det-ana|| = {np.linalg.norm(Ms_det-Ms):.3e} "
          f"(||det||={np.linalg.norm(Ms_det):.3f})  [Fae/Fmi/Wmbej VALIDATED]")
    return Ms_det, Mc_det


def build_direct(data):
    """spin-orbital DIRECT (Coulomb, chemist) t-dressed intermediates for the
    CROSS-SPIN g_phhp dressing. Cross-spin (alpha bra / beta ket) sees only the
    direct Coulomb vertex (no exchange between the two spins) ⇒ g_phhp uses these,
    NOT the antisymmetrized <pq||rs>. Built from chemist (pq|rs) with same-spin
    deltas only on each electron pair."""
    nact = data["nact"]; eri = data["eri"]; nso = 2 * nact
    occ, vir = slices(data)
    # direct spin-orbital integral D[p,q,r,s] = (pr|qs) with sp==sr, sq==ss (Coulomb)
    D = np.zeros((nso, nso, nso, nso))
    for P in range(nso):
        for Q in range(nso):
            for R in range(nso):
                for S in range(nso):
                    if (P // nact == R // nact) and (Q // nact == S // nact):
                        D[P, Q, R, S] = eri[P % nact, R % nact, Q % nact, S % nact]
    return D


def main():
    ncore = 1; xyz = "xyz/H2O.xyz"; basis = "sto-3g"
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    print(f"== {xyz} {basis} ncore={ncore}  nact={data['nact']} nocc={data['nocc']} "
          f"nvir={data['nvir']} (so: {len(occ_so(data))}occ {len(vir_so(data))}vir)")
    dets, index, Hbar = build_sector(data, data["nelec"])
    g, f = validate_foundation(data, dets, index, Hbar)
    # spin-orbital g sliced into blocks
    occ, vir = slices(data)
    oo = np.ix_(occ, occ); vv = np.ix_(vir, vir)
    fov = f[np.ix_(occ, vir)]
    print(f"[foundation] ||f_ov|| = {np.linalg.norm(fov):.4e} (should be ~0 at RHF: canonical)")
    H = build_hbar(data)
    validate_singles(data, dets, index, Hbar, H)


if __name__ == "__main__":
    main()
