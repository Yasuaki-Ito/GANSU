#!/usr/bin/env python3
"""SO Wick derivation of the g_phph (same-spin Ms) route, step 1: build the exact
dressed 3-body Hbar vertex W3[ijk,abc] = <Phi_ijk^abc|Hbar|HF> from the det oracle
(= (T)-triples driver = t2.v, connected since CCSD singles/doubles residuals=0),
and confirm it is nonzero.  Then the route's 3-body part = s_ip fully contracted
into W3.  Enumerating s_ip x {F, W2, W3} in SO (each diagram unambiguous) should
fit the det Ms route cleanly where spatial fitting was collinear.

Run: wsl python3 script/steom_gphph_sowick.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, hf_det, apply_string,
                                 so_index, occ_so, vir_so)


def build_W3(data, dets, index, Hbar):
    """W3[I,J,K,A,B,C] = <Phi_{IJK}^{ABC}|Hbar|HF>, antisymmetrized in (I,J,K) & (A,B,C).
    |Phi_ijk^abc> = a†_A a†_B a†_C a_K a_J a_I |HF>.  SO indices."""
    nso = 2 * data["nact"]
    hf = hf_det(data); iHF = index[hf]
    occ = occ_so(data); vir = vir_so(data)
    W3 = np.zeros((nso, nso, nso, nso, nso, nso))
    for I, J, K in itertools.combinations(occ, 3):
        for A, B, C in itertools.combinations(vir, 3):
            sg, d = apply_string(hf, [("c", A), ("c", B), ("c", C), ("a", K), ("a", J), ("a", I)])
            if sg == 0:
                continue
            val = Hbar[index[d], iHF] * sg
            if abs(val) < 1e-13:
                continue
            # scatter with full antisymmetry
            for pi, (i, j, k) in zip(_perm_signs(), itertools.permutations((I, J, K))):
                for pa, (a, b, c) in zip(_perm_signs(), itertools.permutations((A, B, C))):
                    W3[i, j, k, a, b, c] = pi * pa * val
    return W3


def _perm_signs():
    # signs for the 6 permutations of 3 objects (parity)
    return [1, -1, -1, 1, 1, -1]  # matches itertools.permutations order for (0,1,2)


def ms_route_and_cands(data, dets, index, Hbar, W3):
    """det Ms same-spin route (sip_sp gauge) + SO candidates s x {f, g, W3}."""
    import steom_ip_route_derive as IPD
    from steom_so_derive import det_singles_block, build_so_integrals
    from steom_fockspace_ref import build_S, solve_ip
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data)
    SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso)) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    comm = S_ip @ Hbar - Hbar @ S_ip
    Ms0, _, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, _, _ = det_singles_block(data, dets, index, Hbar + comm)
    Ms = Ms1 - Ms0                          # [i,a,j,b] same-spin route target
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP_clean:
        sip[m] = SIP_clean[m]
    g, f = build_so_integrals(data)
    # same-spin slice indices
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, oa, va)].ravel()
    cand = {}
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso):
                continue
            v = slc(Tp)
            if np.linalg.norm(v) < 1e-9:
                continue
            cand[f"{tag}:p{perm}"] = v
    # s x g (2-body, contract 2 of s's 3 amp axes with 2 of g's 4) ; root e external
    for sc in itertools.combinations(range(3), 2):
        sf = [k for k in range(3) if k not in sc][0]
        for xc in itertools.permutations(range(4), 2):
            lab_s = ['p', 'q', 'r']; lab_x = list("wxyz")
            for si, xi in zip(sc, xc):
                lab_x[xi] = lab_s[si]
            xf = [k for k in range(4) if k not in xc]
            out = [lab_x[xf[0]], lab_x[xf[1]], lab_s[sf], 'e']
            es = f"e{''.join(lab_s)},{''.join(lab_x)}->{''.join(out)}"
            try:
                add(np.einsum(es, sip, g, optimize=True), f"g:{es}")
            except Exception:
                pass
    # s x f (1-body)
    for si in range(3):
        for fi in range(2):
            lab_s = ['p', 'q', 'r']; lab_f = ['w', 'x']; lab_f[fi] = lab_s[si]
            sfree = [k for k in range(3) if k != si]; ffree = 1 - fi
            out = [lab_f[ffree], lab_s[sfree[0]], lab_s[sfree[1]], 'e']
            es = f"e{''.join(lab_s)},{''.join(lab_f)}->{''.join(out)}"
            add(np.einsum(es, sip, f, optimize=True), f"f:{es}")
    # s x W3: contract ANY 3 of s's 4 legs {e(=m root), p, q, r} = {m,I,J,B} with 3 of
    # W3's 6 legs; the remaining s-leg + 3 W3-free legs = external [i,a,j,b].  The
    # root-contracted cases (leg e contracted, an amp leg external) were MISSED before.
    slegs = ['e', 'p', 'q', 'r']                        # sip axes 0,1,2,3 = m,I,J,B
    for s3 in itertools.combinations(range(4), 3):       # which 3 s-legs contract
        sfree_leg = [k for k in range(4) if k not in s3][0]
        for w3c in itertools.permutations(range(6), 3):  # target W3 legs for the 3 s-legs
            sl = list(slegs); wl = list("UVWXYZ")
            for si, wi in zip(s3, w3c):
                wl[wi] = sl[si]
            wfree = [k for k in range(6) if k not in w3c]
            out = [sl[sfree_leg], wl[wfree[0]], wl[wfree[1]], wl[wfree[2]]]
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
            try:
                add(np.einsum(es, sip, W3, optimize=True), f"W3:{es}")
            except Exception:
                pass
    return Ms, cand


def _fit(Ms, cand, subset_pred, label):
    names = [n for n in cand if subset_pred(n)]
    if not names:
        print(f"  [{label}] no candidates"); return
    A = np.stack([cand[n] for n in names], 1)
    tv = Ms.ravel(); tn = np.linalg.norm(tv)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    print(f"  [{label}] ncand={len(names)}  FULL-lstsq resid={np.linalg.norm(A@co-tv)/tn:.3e}")


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nocc, nvir, nact = data["nocc"], data["nvir"], data["nact"]
    print(f"H2O FC1: nact={nact} nocc={nocc} nvir={nvir} nso={2*nact} "
          f"(occ-so={2*nocc}, vir-so={2*nvir})")
    dets, index, Hbar = build_sector(data, data["nelec"])
    W3 = build_W3(data, dets, index, Hbar)
    print(f"||W3|| = {np.linalg.norm(W3):.6f}  (nonzero => 3-body Hbar / (T) driver present)")
    Ms, cand = ms_route_and_cands(data, dets, index, Hbar, W3)
    print(f"||Ms route||={np.linalg.norm(Ms):.4f}  total SO candidates={len(cand)}")
    _fit(Ms, cand, lambda n: n.startswith(("g:", "f:")), "2-body only (s x g,f)")
    _fit(Ms, cand, lambda n: n.startswith("W3:"), "3-body only (s x W3)")
    _fit(Ms, cand, lambda n: True, "2-body + 3-body (s x g,f,W3)")
    return
    # sanity vs the (T) formula W3 = P(i/jk)P(a/bc)[t_jk^ae <ei||bc> - t_im^bc <ma||jk>]
    from steom_so_derive import build_so_integrals
    from steom_fockspace_ref import build_t_so
    g, f = build_so_integrals(data)   # g = <pq||rs> physicist antisym
    _, t2 = build_t_so(data)          # t2[I,J,A,B]
    occ = occ_so(data); vir = vir_so(data)
    O = occ; V = vir
    goovv = g[np.ix_(O, O, V, V)]; t2oovv = t2[np.ix_(O, O, V, V)]
    gvvvo = g[np.ix_(V, V, V, O)]; govoo = g[np.ix_(O, V, O, O)]
    # <ei||bc> with e vir -> use g[V,O? ] careful: physicist <ei||bc>=g[e,i,b,c]? our g is <pq||rs>
    # term1: t_jk^ae <ei||bc> ; term2: -t_im^bc <ma||jk>
    no, nv = len(O), len(V)
    # map to compact
    t2c = t2[np.ix_(O, O, V, V)]
    geibc = g[np.ix_(V, O, V, V)]      # <ei||bc> e vir, i occ -> [e,i,b,c]
    gmajk = g[np.ix_(O, V, O, O)]      # <ma||jk> [m,a,j,k]
    T1 = np.einsum("jkae,eibc->ijkabc", t2c, geibc, optimize=True)   # t_jk^ae <ei||bc>
    T2 = -np.einsum("imbc,majk->ijkabc", t2c, gmajk, optimize=True)  # -t_im^bc <ma||jk>
    def Pijk_abc(X):
        # antisymmetrize i/jk and a/bc  (P(i/jk)=1-(ij)-(ik), P(a/bc)=1-(ab)-(ac))
        Y = X - X.transpose(1, 0, 2, 3, 4, 5) - X.transpose(2, 1, 0, 3, 4, 5)
        Z = Y - Y.transpose(0, 1, 2, 4, 3, 5) - Y.transpose(0, 1, 2, 5, 4, 3)
        return Z
    W3f = Pijk_abc(T1 + T2)
    # compare compact block of W3 (occ,occ,occ,vir,vir,vir) to W3f
    W3blk = W3[np.ix_(O, O, O, V, V, V)]
    r = np.linalg.norm(W3blk - W3f) / (np.linalg.norm(W3blk) + 1e-30)
    print(f"(T)-formula check: ||W3_oracle - W3_formula|| rel = {r:.3e}  "
          f"(||W3f||={np.linalg.norm(W3f):.4f})")


if __name__ == "__main__":
    main()
