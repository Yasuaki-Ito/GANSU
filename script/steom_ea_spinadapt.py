#!/usr/bin/env python3
"""Spin-adapt the g_phhp EA route to a SPATIAL formula using the CONFIRMED
PySCF RHF EOMEA r2 convention (spatial2spin_ea, eom_rccsd.py:639):

    spatial r2[j,a,b]  ==  SO block  sea[j_beta, a_alpha, b_beta]   (bab, DIRECT)
    same-spin (aaa)    ==  r2[j,a,b] - r2[j,b,a]                     (particle antisym)

The earlier spatial fit (steom_spatial_fit) failed because it fit GANSU's
RENORMALIZED s_EA against the oracle target — two different normalizations.
Here we extract the amplitude straight from the oracle's own sea in the exact
spatial convention (s_sp), so the fit is convention-consistent and must be sparse
if the SO structure (2 direct-Coulomb terms) is right.

Run: wsl python3 script/steom_ea_spinadapt.py
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


def get_target_and_sea(data):
    """cross-spin g_phhp EA-route target Mc_lin - Mc_base  [i,a,j,b] (i_a,a_a,j_b,b_b)."""
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    S_ea = build_S(data, dets, index, zIP, sEA)
    comm = S_ea @ Hbar - Hbar @ S_ea
    _, Mc_base, _ = det_singles_block(data, dets, index, Hbar)
    _, Mc_lin, _ = det_singles_block(data, dets, index, Hbar + comm)
    target = Mc_lin - Mc_base
    return target, sEA


def extract_spatial_amp(sEA, data):
    """s_sp[e][j,a,b] = sea[e_alpha][ j_beta, a_alpha, b_beta ]  (the bab block).
    By the confirmed convention this IS the closed-shell spatial r2 the RHF
    EOMEA solver would store for spatial root e."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    s_sp = np.zeros((nvir, nocc, nvir, nvir))
    for v in range(nvir):
        eA = so_index(v + nocc, 0, nact)              # alpha vir root
        for j in range(nocc):
            Jb = so_index(j, 1, nact)                 # beta occ hole
            for a in range(nvir):
                Aa = so_index(a + nocc, 0, nact)      # alpha vir particle
                for b in range(nvir):
                    Bb = so_index(b + nocc, 1, nact)  # beta vir particle
                    s_sp[v, j, a, b] = sEA[eA][Jb, Aa, Bb]
    return s_sp


def verify_convention(sEA, s_sp, data):
    """Check the aaa block of the alpha-root equals s_sp - s_sp.transpose(a,b)."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    err = 0.0; nrm = 0.0
    for v in range(nvir):
        eA = so_index(v + nocc, 0, nact)
        for j in range(nocc):
            Ja = so_index(j, 0, nact)                 # alpha occ hole (aaa)
            for a in range(nvir):
                Aa = so_index(a + nocc, 0, nact)
                for b in range(nvir):
                    Ba = so_index(b + nocc, 0, nact)
                    ref = s_sp[v, j, a, b] - s_sp[v, j, b, a]
                    err += (sEA[eA][Ja, Aa, Ba] - ref) ** 2
                    nrm += ref ** 2
    print(f"  [convention] ||aaa - (s_sp - s_sp^T)|| = {np.sqrt(err):.3e}  "
          f"(||aaa||={np.sqrt(nrm):.3f})")


def spatial_candidates(s_sp, eri, nocc, nvir):
    """Enumerate spatial contractions of the EA amplitude with chemist eri (AB|CD).
    Spin-integrating an SO term produces BOTH the direct amplitude s_sp[e][j,a,b]
    and its particle-swap s_sp[e][j,b,a] (the two spin channels), so we enumerate
    over both amplitude variants."""
    o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    amps = {
        "s":  s_sp,                                # direct (bab spin block)
        "sT": s_sp.transpose(0, 1, 3, 2),          # particle-swap  s[e][j,b,a]
    }
    blocks = {
        "ovvv": eri[o, v, v, v],   # (i a | b c)
        "ooov": eri[o, o, o, v],   # (i j | k a)
        "oovv": eri[o, o, v, v],   # (i j | a b)
        "ovov": eri[o, v, o, v],   # (i a | j b)
        "vvvv": eri[v, v, v, v],
        "oooo": eri[o, o, o, o],
    }
    cand = {}
    seaax = ['j', 'a', 'b']          # amplitude axes (occ,vir,vir)
    for Aname, S in amps.items():
        for Bname, B in blocks.items():
            nleg = 4
            for sea_contract in itertools.combinations(range(3), 2):
                sea_free = [k for k in range(3) if k not in sea_contract][0]
                for xc in itertools.permutations(range(nleg), 2):
                    lab_sea = list(seaax); lab_x = list("pqrs")
                    for si, xi in zip(sea_contract, xc):
                        lab_x[xi] = lab_sea[si]
                    x_free = [k for k in range(nleg) if k not in xc]
                    out = [lab_x[x_free[0]], lab_x[x_free[1]], lab_sea[sea_free], 'e']
                    es = f"e{''.join(lab_sea)},{''.join(lab_x)}->{''.join(out)}"
                    try:
                        T = np.einsum(es, S, B, optimize=True)
                    except Exception:
                        continue
                    for perm in itertools.permutations(range(3)):
                        Tp = np.transpose(T, (perm[0], perm[1], perm[2], 3))
                        if Tp.shape != (nocc, nvir, nocc, nvir):
                            continue
                        if np.linalg.norm(Tp) < 1e-9:
                            continue
                        cand[f"{Aname}|{Bname}:{es}:perm{perm}"] = Tp.ravel()
    return cand


def fit(target, cand, tag):
    tv = target.ravel(); tn = np.linalg.norm(tv)
    # dedup identical columns
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v / (np.linalg.norm(v) + 1e-30), 9))
        if key not in uniq:
            uniq[key] = n
    names = list(uniq.values())
    print(f"  [{tag}] {len(cand)} raw -> {len(names)} unique, ||target||={tn:.4f}")
    best = min(names, key=lambda n: np.linalg.norm(tv - (cand[n] @ tv / (cand[n] @ cand[n])) * cand[n]))
    cb = cand[best] @ tv / (cand[best] @ cand[best])
    print(f"    BEST single: rel-resid={np.linalg.norm(tv - cb * cand[best]) / tn:.3e}  c={cb:+.4f}  {best}")
    chosen = []; res = tv.copy()
    for step in range(4):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n] @ res / (cand[n] @ cand[n])) * cand[n]))
        if n in chosen:
            break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tv, rcond=None)
        res = tv - Ac @ co
        print(f"    step {step + 1}: rel-resid={np.linalg.norm(res) / tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"        {cc:+.4f}  {m}")
        if np.linalg.norm(res) / tn < 5e-3:
            break


def test_current_guess(target, s_sp, eri, nocc, nvir):
    """The current STEOM_EA_ROUTE spatial guess, but with the CORRECT amplitude s_sp:
       Term1  s[i,c,d]*eri_ovvv[j,d,c,a] ;  Term2  s[k,a,c]*eri_ooov[k,i,j,c]
    (eri_ovvv/eri_ooov are physicist bar_h blocks; recompute from chemist here)."""
    o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    # bar_h eri_ovvv = <ja|dc>? build_g uses eri_ovvv[j,d,c,a]; below we just try a
    # few chemist contractions matching the intended (vir-vir) and (occ-vir) sums.
    tv = target.ravel(); tn = np.linalg.norm(tv)
    ovvv = eri[o, v, v, v]      # (i a|b c) chemist
    ooov = eri[o, o, o, v]      # (i j|k a) chemist
    # Term1 candidate: sum_cd s[e][i,c,d] (a c|d j)_chem  -> [i,a,j,e]
    #   (ac|dj) = eri[a+? ...]; use chemist eri[va, vc, vd, oj]
    T1 = np.einsum("eicd,acdj->iaje", s_sp,
                   eri[v, v, v, o], optimize=True)   # (a c|d j)
    # Term2 candidate: sum_kc s[e][k,c,a] (k i|c j)_chem -> [i,a,j,e]
    T2 = np.einsum("ekca,kicj->iaje", s_sp,
                   eri[o, o, v, o], optimize=True)   # (k i|c j)
    for lbl, arr in [("T1 (ac|dj)", T1), ("T2 (ki|cj)", T2)]:
        c = arr.ravel() @ tv / (arr.ravel() @ arr.ravel())
        r = np.linalg.norm(tv - c * arr.ravel()) / tn
        print(f"  [guess] {lbl}: single-c={c:+.4f} rel-resid={r:.3e}")
    A = np.stack([T1.ravel(), T2.ravel()], 1)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    r = np.linalg.norm(tv - A @ co) / tn
    print(f"  [guess] T1+T2 lstsq: c=({co[0]:+.4f},{co[1]:+.4f})  rel-resid={r:.3e}")


def compare_so_terms(sEA, s_sp, eri, data):
    """Build the EXACT SO Term1/Term2 (the steom_so_fit2 winners), slice to the
    cross block [i,a,j,b], and compare element-wise to the Kramers-reduced spatial
    T1/T2.  This localizes any error in the hand reduction."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    nso = 2 * nact
    D = build_direct(data)
    sea = np.zeros((nso, nso, nso, nso))
    for e in sEA:
        sea[e] = sEA[e]
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    ob = [so_index(j, 1, nact) for j in range(nocc)]
    vb = [so_index(b + nocc, 1, nact) for b in range(nvir)]
    def slc(T):
        return T[np.ix_(oa, va, ob, vb)]
    # Term1: D:ejab,abrs->rsje:perm(2,0,1)
    T1so = np.einsum("ejab,abrs->rsje", sea, D, optimize=True)
    T1so = np.transpose(T1so, (2, 0, 1, 3))
    T1so = slc(T1so)                                   # [i,a,j,b]
    # Term2: D:ejab,jars->rsbe:perm(0,2,1)
    T2so = np.einsum("ejab,jars->rsbe", sea, D, optimize=True)
    T2so = np.transpose(T2so, (0, 2, 1, 3))
    T2so = slc(T2so)                                   # [i,a,j,b]
    o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    T1sp = -np.einsum("bicd,cadj->iajb", s_sp, eri[v, v, v, o], optimize=True)
    T2sp = +np.einsum("bkca,kicj->iajb", s_sp, eri[o, o, v, o], optimize=True)
    for lbl, so, sp in [("T1", T1so, T1sp), ("T2", T2so, T2sp)]:
        d = np.linalg.norm(so - sp); n = np.linalg.norm(so)
        # also try best scalar & sign
        c = (so.ravel() @ sp.ravel()) / (sp.ravel() @ sp.ravel() + 1e-30)
        print(f"  [SO-vs-sp] {lbl}: ||SO||={n:.4f} ||sp||={np.linalg.norm(sp):.4f} "
              f"||SO-sp||={d:.3e}  best-scale(sp->SO)={c:+.4f}")
    # direct block-identity check for the beta-root blocks used by each term
    bbeta = [so_index(b + nocc, 1, nact) for b in range(nvir)]
    # T2 block: sea[b_beta][k_alpha, c_beta, a_alpha] =? +s_sp[b][k,c,a]
    blk2 = np.zeros((nvir, nocc, nvir, nvir))
    # T1 block: sea[b_beta][i_alpha, c_alpha, d_beta] =? -s_sp[b][i,c,d]
    blk1 = np.zeros((nvir, nocc, nvir, nvir))
    for b in range(nvir):
        Bb = bbeta[b]
        for i in range(nocc):
            for c in range(nvir):
                for d in range(nvir):
                    blk1[b, i, c, d] = sEA[Bb][so_index(i, 0, nact),
                                              so_index(c + nocc, 0, nact),
                                              so_index(d + nocc, 1, nact)]
                    blk2[b, i, c, d] = sEA[Bb][so_index(i, 0, nact),
                                              so_index(c + nocc, 1, nact),
                                              so_index(d + nocc, 0, nact)]
    # ---- decisive test: does a CLEAN convention-rebuilt sea reproduce Term1? ----
    sea_r = build_sea_recon(s_sp, data)
    dmix = np.linalg.norm(sea - sea_r)
    print(f"  [recon] ||oracle_sea - convention_sea|| = {dmix:.3e}  (||sea||={np.linalg.norm(sea):.3f}) "
          f"<- nonzero => oracle eig mixed degenerate a/b roots")
    T1r = np.einsum("ejab,abrs->rsje", sea_r, D, optimize=True)
    T1r = slc(np.transpose(T1r, (2, 0, 1, 3)))
    T2r = np.einsum("ejab,jars->rsbe", sea_r, D, optimize=True)
    T2r = slc(np.transpose(T2r, (0, 2, 1, 3)))
    print(f"  [recon] Term1: ||T1_recon - T1sp||={np.linalg.norm(T1r - T1sp):.3e}  "
          f"||T1_recon - T1_oracleSO||={np.linalg.norm(T1r - T1so):.3e}")
    print(f"  [recon] Term2: ||T2_recon - T2sp||={np.linalg.norm(T2r - T2sp):.3e}  "
          f"||T2_recon - T2_oracleSO||={np.linalg.norm(T2r - T2so):.3e}")
    print(f"  [block] T1 sea[b_b][i_a,c_a,d_b] vs -s_sp[b][i,c,d]: "
          f"||d||={np.linalg.norm(blk1 + s_sp):.3e}  ||blk||={np.linalg.norm(blk1):.3e}")
    print(f"  [block] T2 sea[b_b][k_a,c_b,a_a] vs +s_sp[b][k,a,c]: "
          f"||d||={np.linalg.norm(blk2 - s_sp.transpose(0, 1, 3, 2)):.3e}  ||blk||={np.linalg.norm(blk2):.3e}")
    # also: is the T1 SO block antisymmetric combo of s_sp?  test -(s - s^T) variants
    for nm, candblk in [("-s_sp", -s_sp),
                        ("-(s_sp - s_sp^T on c,d)", -(s_sp - s_sp.transpose(0, 1, 3, 2)))]:
        print(f"      T1 block vs {nm}: ||d||={np.linalg.norm(blk1 - candblk):.3e}")


def build_sea_recon(s_sp, data):
    """Reconstruct the full SO sea[E,J,A,B] from spatial s_sp via the CONFIRMED
    convention (spatial2spin_ea) for alpha-roots and its Kramers flip for beta-roots."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    nso = 2 * nact
    sea = np.zeros((nso, nso, nso, nso))
    def O(i, s): return so_index(i, s, nact)
    def V(a, s): return so_index(a + nocc, s, nact)
    for e in range(nvir):
        s = s_sp[e]                                  # [j,a,b]
        sa = s - s.transpose(0, 2, 1)                # antisym (a,b)
        Ea = V(e, 0); Eb = V(e, 1)
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    # alpha-root e_alpha
                    sea[Ea, O(j, 0), V(a, 0), V(b, 0)] = sa[j, a, b]      # aaa
                    sea[Ea, O(j, 1), V(a, 0), V(b, 1)] = s[j, a, b]       # bab
                    sea[Ea, O(j, 1), V(a, 1), V(b, 0)] = -s[j, a, b]      # bba
                    # beta-root e_beta (Kramers flip)
                    sea[Eb, O(j, 1), V(a, 1), V(b, 1)] = sa[j, a, b]      # bbb
                    sea[Eb, O(j, 0), V(a, 1), V(b, 0)] = s[j, a, b]       # aba
                    sea[Eb, O(j, 0), V(a, 0), V(b, 1)] = -s[j, a, b]      # aab
    return sea


def test_derived(target, s_sp, eri, nocc, nvir):
    """The spatial formula derived analytically from the SO Term1/Term2 by
    Kramers-reducing the beta-root sea blocks to the spatial amplitude s_sp:
        Mc[i,a,j,b] = -Sum_cd s_sp[b][i,c,d]*(ca|dj)   (Term1, bab->bba = -s_sp)
                      +Sum_kc s_sp[b][k,c,a]*(ki|cj)    (Term2, bab      = +s_sp)
    chemist eri[A,B,C,D]=(AB|CD).  s_sp[e][i,c,d], root e == output b."""
    o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    T1 = -np.einsum("bicd,cadj->iajb", s_sp, eri[v, v, v, o], optimize=True)  # (ca|dj)
    T2 = +np.einsum("bkca,kicj->iajb", s_sp, eri[o, o, v, o], optimize=True)  # (ki|cj)
    tv = target.ravel(); tn = np.linalg.norm(tv)
    for lbl, arr in [("T1", T1), ("T2", T2), ("T1+T2", T1 + T2)]:
        d = (T1 + T2).ravel() if lbl == "T1+T2" else arr.ravel()
        c = d @ tv / (d @ d)
        print(f"  [DERIVED] {lbl:6s}: single-c={c:+.4f}  rel-resid(c=1)={np.linalg.norm(tv - d) / tn:.3e}"
              f"  rel-resid(best-c)={np.linalg.norm(tv - c * d) / tn:.3e}")
    A = np.stack([T1.ravel(), T2.ravel()], 1)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    print(f"  [DERIVED] free lstsq coeffs = ({co[0]:+.4f}, {co[1]:+.4f})  "
          f"rel-resid={np.linalg.norm(tv - A @ co) / tn:.3e}")


def target_from_clean_sea(data, s_sp, full=False):
    """Rebuild the EA-route contribution to the cross block using the CLEAN convention
    sea (per-root), so target and formula share GANSU's gauge.
      full=False: LINEAR commutator [S_ea,Hbar]  (matches a linear closed form)
      full=True:  FULL transform expm(S_ea) Hbar expm(-S_ea)  (linear + S_ea^2 quadratic,
                  = what cross_locate measured / the plain-STEOM S_ea route)."""
    from scipy.linalg import expm
    sea_r = build_sea_recon(s_sp, data)
    sEA_clean = {E: sea_r[E] for E in vir_so(data)}
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    S_ea = build_S(data, dets, index, zIP, sEA_clean)
    if full:
        Hd = expm(S_ea) @ Hbar @ expm(-S_ea)
    else:
        Hd = Hbar + (S_ea @ Hbar - Hbar @ S_ea)
    _, Mc_base, _ = det_singles_block(data, dets, index, Hbar)
    _, Mc_ea, _ = det_singles_block(data, dets, index, Hd)
    return Mc_ea - Mc_base


def so_enum_clean(data, s_sp, label):
    """SO systematic enumeration (steom_so_fit2 style) but with the CLEAN convention
    sea and the CLEAN-gauge target — the gauge GANSU actually uses. Finds the COMPLETE
    linear-in-s_EA g_phhp EA route (the oracle-gauge 2-term fit is incomplete here)."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sea_r = build_sea_recon(s_sp, data)
    sEA_clean = {E: sea_r[E] for E in vir_so(data)}
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    S_ea = build_S(data, dets, index, zIP, sEA_clean)
    comm = S_ea @ Hbar - Hbar @ S_ea
    _, Mc_base, _ = det_singles_block(data, dets, index, Hbar)
    _, Mc_lin, _ = det_singles_block(data, dets, index, Hbar + comm)
    target = Mc_lin - Mc_base                          # [i,a,j,b]
    g, f = build_so_integrals(data)
    D = build_direct(data)
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    ob = [so_index(j, 1, nact) for j in range(nocc)]
    vb = [so_index(b + nocc, 1, nact) for b in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, ob, vb)]
    cand = {}
    seaax = ['j', 'a', 'b']
    for Xname, X in [("D", D), ("g", g)]:
        for sc in itertools.combinations(range(3), 2):
            sf = [k for k in range(3) if k not in sc][0]
            for xc in itertools.permutations(range(4), 2):
                lab_s = list(seaax); lab_x = list("pqrs")
                for si, xi in zip(sc, xc):
                    lab_x[xi] = lab_s[si]
                xf = [k for k in range(4) if k not in xc]
                out = [lab_x[xf[0]], lab_x[xf[1]], lab_s[sf], 'e']
                es = f"e{''.join(lab_s)},{''.join(lab_x)}->{''.join(out)}"
                try:
                    T = np.einsum(es, sea_r, X, optimize=True)
                except Exception:
                    continue
                for perm in itertools.permutations(range(3)):
                    Tp = np.transpose(T, (perm[0], perm[1], perm[2], 3))
                    c = slc(Tp)
                    if np.linalg.norm(c) < 1e-9:
                        continue
                    cand[f"{Xname}:{es}:perm{perm}"] = c.ravel()
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v / (np.linalg.norm(v) + 1e-30), 9))
        uniq.setdefault(key, n)
    names = list(uniq.values())
    tv = target.ravel(); tn = np.linalg.norm(tv)
    print(f"  [{label}] clean SO enum: {len(cand)}->{len(names)} uniq, ||target||={tn:.4f}")
    chosen = []; res = tv.copy()
    for step in range(6):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n] @ res / (cand[n] @ cand[n])) * cand[n]))
        if n in chosen:
            break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tv, rcond=None)
        res = tv - Ac @ co
        print(f"    step {step + 1}: rel-resid={np.linalg.norm(res) / tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"        {cc:+.4f}  {m}")
        if np.linalg.norm(res) / tn < 3e-3:
            break


def run(atom, active=None, label=""):
    print(f"\n===== {label} =====")
    data = (get_active_data(atom=atom, basis="sto-3g", active=active) if active
            else get_active_data(atom=atom, basis="sto-3g", ncore=0))
    nocc, nvir = data["nocc"], data["nvir"]
    eri = data["eri"]                         # chemist (pq|rs)
    target, sEA = get_target_and_sea(data)
    s_sp = extract_spatial_amp(sEA, data)
    verify_convention(sEA, s_sp, data)
    print(f"  nocc={nocc} nvir={nvir}  ||s_sp||={np.linalg.norm(s_sp):.4f}  "
          f"||target||={np.linalg.norm(target):.4f}")
    test_current_guess(target, s_sp, eri, nocc, nvir)
    test_derived(target, s_sp, eri, nocc, nvir)
    compare_so_terms(sEA, s_sp, eri, data)
    # gauge-consistent target (clean convention sea) vs the derived formula
    o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    T1 = -np.einsum("bicd,cadj->iajb", s_sp, eri[v, v, v, o], optimize=True)
    T2 = +np.einsum("bkca,kicj->iajb", s_sp, eri[o, o, v, o], optimize=True)
    tgt_clean = target_from_clean_sea(data, s_sp)
    dd = tgt_clean - (T1 + T2)
    print(f"  [GAUGE] ||clean-target||={np.linalg.norm(tgt_clean):.4f}  "
          f"||clean-target - (T1+T2)||={np.linalg.norm(dd):.3e}  "
          f"rel={np.linalg.norm(dd) / np.linalg.norm(tgt_clean):.3e}")
    # complete clean-gauge SO enumeration (the gauge GANSU uses)
    so_enum_clean(data, s_sp, label)


def main():
    run("; ".join(f"H {2.0 * (k % 2)} {1.4 * (k // 2)} 0" for k in range(6)),
        label="H6 rect ladder")
    if "h8" in sys.argv:
        run("; ".join(f"H {2.0 * (k % 2)} {1.4 * (k // 2)} 0" for k in range(8)),
            label="H8 rect ladder")
    try:
        import steom_cas_verify as V
        atom = V.polyene(6); active, _ = V.detect_pi(atom, "sto-3g", 3, 3)
        run(atom, active=active, label="hexatriene pi-CAS(6,6)")
    except Exception as e:
        print(f"  (hexatriene skipped: {e})")


if __name__ == "__main__":
    main()
