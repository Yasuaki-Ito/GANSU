#!/usr/bin/env python3
"""DLPNO-CCSD singles (T1) term audit vs canonical CCSD — full-space oracle.

Goal: explain why GANSU_DLPNO_CCSD_SINGLES=1 (increments 1+2+E) WORSENS
DLPNO-STEOM (+0.2 eV) even though the ground-state energy improves.

Method (convention-safe, no DLPNO machinery involved):
  1. PySCF canonical CCSD (cart=True, same xyz as GANSU) -> t1*, t2*.
  2. Transcribe GANSU's canonical T1 residual (src/eri_stored.cu:7128-7155,
     host reference; v(p,q,r,s)=<pq|rs> physicist, w=2v-v_swap,
     Dia=eps_i-eps_a) and VERIFY the fixed-point identity val/Dia == t1*.
  3. Solve the T1 equation with t2 FROZEN at t2* for term subsets:
       impl  : what dlpno_ccsd.cu solve_t1 implements today
               (S4 ring w_voov*t1, ovvv*t2 [inc1+S2a], ooov*t2 [S2b], f_ia)
       +C    : + Fac*t1 - Fki*t1 with tau-dressed Fac/Fki  (missing C)
       +A    : + Fkc*(2 t2 - t2^T + t1 t1)                 (missing A)
       +B    : + t1t1 parts of ovvv/ooov sources           (missing B1/B2)
     ("+B" == full canonical eq -> must reproduce t1* to solver tol; this
      validates the transcription AND the increment inventory.)
  4. Report, per variant: max|dT1|, ||dT1||/||t1*||, and whether the variant
     is closer to t1* than T1=0 is (the STEOM on/off question).
  5. T2-side: decompose the canonical T2 residual's T1-couplings at
     (t1*,t2*) into implemented {vvvv*t1t1, vvov*t1, vooo*t1} vs missing
     (Wklij t1-dressing+tau, Wabcd t1-dressing, Wakic/Wakci t1-dressing,
      Lac/Lki t1-dressing, quadratic vvov/vooo) and estimate the induced
     T2 error ||missing/Dijab|| / ||t2*||.

Run:  wsl python3 script/dlpno_singles_term_audit.py [xyzfile basis]
Default: xyz/Formaldehyde.xyz and xyz/C2H4.xyz, cc-pVDZ, cart=True, no FC.
"""
import sys
import numpy as np
from pyscf import gto, scf, cc, ao2mo

np.set_printoptions(precision=4, suppress=True)


def load_xyz(path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].split()[0])
    atoms = [l.split()[:4] for l in lines[2:2 + n]]
    return "\n".join(" ".join(a) for a in atoms)


def canonical_ccsd(xyz, basis):
    mol = gto.M(atom=xyz, basis=basis, cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    mycc.max_cycle = 200
    mycc.kernel()
    assert mycc.converged
    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    # full MO ERI, chemist (pq|rs)
    eri_chem = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff), nmo)
    # physicist <pq|rs> = (pr|qs)
    v = eri_chem.transpose(0, 2, 1, 3)
    return mf, mycc, v, nocc, nmo


def build_blocks(v, eps, nocc, nmo):
    o = slice(0, nocc)
    vs = slice(nocc, nmo)
    b = {}
    b["voov"] = v[vs, o, o, vs]        # <ak|ic>
    b["vovo"] = v[vs, o, vs, o]        # <ak|ci>
    b["vovv"] = v[vs, o, vs, vs]       # <ak|cd>
    b["ooov"] = v[o, o, o, vs]         # <kl|ic>
    b["oovv"] = v[o, o, vs, vs]        # <kl|cd>
    b["eps_o"] = eps[:nocc]
    b["eps_v"] = eps[nocc:]
    # blocks for the "misread" (as-implemented) variants
    b["voov_c"] = v[vs, o, o, vs]      # <ai|kc> indexed [a,i,k,c]
    b["ovvv_c"] = v[o, vs, vs, vs]     # <kd|ac> indexed [k,d,a,c]
    b["ooov_c"] = v[o, o, o, vs]       # <ki|lc> indexed [k,i,l,c]
    return b


def w_of(vb, swap_axes):
    """w = 2*v - v with the last two 'electron-2' labels swapped.
    GANSU w(p,q,r,s) = 2<pq|rs> - <pq|sr>."""
    return 2.0 * vb - vb.transpose(swap_axes)


def t1_val(b, t1, t2, terms):
    """GANSU canonical T1 'val' (eri_stored.cu:7128-7155), term-switchable.
    t2[i,j,a,b]; returns val[i,a]."""
    no, nv = t1.shape
    val = np.zeros((no, nv))
    w_oovv = 2.0 * b["oovv"] - b["oovv"].transpose(0, 1, 3, 2)   # w(k,l,C,D)

    if terms.get("fvv_dress"):
        # Fac[a,c] = -sum_kld w(k,l,C,D)*(t2[k,l,a,d] + t1[k,a]t1[l,d])
        tau = t2 + (np.einsum("ka,ld->klad", t1, t1) if terms.get("tau_in_f", True) else 0.0)
        Fac = -np.einsum("klcd,klad->ac", w_oovv, tau)
        val += np.einsum("ac,ic->ia", Fac, t1)
    if terms.get("foo_dress"):
        # Fki[k,i] = sum_lcd w(k,l,C,D)*(t2[i,l,c,d] + t1[i,c]t1[l,d])
        tau = t2 + (np.einsum("ic,ld->ilcd", t1, t1) if terms.get("tau_in_f", True) else 0.0)
        Fki = np.einsum("klcd,ilcd->ki", w_oovv, tau)
        val -= np.einsum("ki,ka->ia", Fki, t1)
    if terms.get("fkc"):
        # Fkc[k,c] = sum_ld w(k,l,C,D) t1[l,d]
        # term: sum_kc Fkc * (2 t2[k,i,c,a] - t2[i,k,c,a] + t1[i,c] t1[k,a])
        Fkc = np.einsum("klcd,ld->kc", w_oovv, t1)
        val += 2.0 * np.einsum("kc,kica->ia", Fkc, t2)
        val -= np.einsum("kc,ikca->ia", Fkc, t2)
        val += np.einsum("kc,ic,ka->ia", Fkc, t1, t1)
    if terms.get("ring"):
        # + sum_kc w(A,k,i,C) t1[k,c] ; w = 2<ak|ic> - <ak|ci>
        wv = 2.0 * b["voov"] - b["vovo"].transpose(0, 1, 3, 2)   # [a,k,i,c]
        val += np.einsum("akic,kc->ia", wv, t1)
    if terms.get("ovvv_t2"):
        # + sum_kcd w(A,k,C,D) t2[i,k,c,d] ; w = 2<ak|cd> - <ak|dc>
        wv = 2.0 * b["vovv"] - b["vovv"].transpose(0, 1, 3, 2)
        val += np.einsum("akcd,ikcd->ia", wv, t2)
    if terms.get("ovvv_t1t1"):
        wv = 2.0 * b["vovv"] - b["vovv"].transpose(0, 1, 3, 2)
        val += np.einsum("akcd,ic,kd->ia", wv, t1, t1)
    if terms.get("ooov_t2"):
        # - sum_klc w(k,l,i,C) t2[k,l,a,c] ; w = 2<kl|ic> - <kl|ci>
        wo = 2.0 * b["ooov"] - v_oovo_as_ooov(b)
        val -= np.einsum("klic,klac->ia", wo, t2)
    if terms.get("ooov_t1t1"):
        wo = 2.0 * b["ooov"] - v_oovo_as_ooov(b)
        val -= np.einsum("klic,ka,lc->ia", wo, t1, t1)

    # ---- "misread" variants: what dlpno_ccsd.cu ACTUALLY contracts. ----
    # The phase24 blocks are chemist (pq|rs) with the physicist index string
    # transliterated, e.g. W_voov_diag[a,k,c] = chem(ak|ic) instead of the
    # canonical 2*chem(ai|kc) - chem(ac|ki). Since chem(pq|rs) = chem(pq|sr),
    # each 2W-W' pair collapses to a single Coulomb-type integral.
    # chem(pq|rs) = <pr|qs>  =>  express via physicist v blocks.
    if terms.get("ring_mis"):
        # chem(ak|ic) = <ai|kc>  -> v[a,i,k,c]
        val += np.einsum("aikc,kc->ia", b["voov_c"], t1)
    if terms.get("ovvv_mis"):
        # chem(ka|dc) = <kd|ac>  -> v[k,d,a,c];  sum_kcd * t2[i,k,c,d]
        val += np.einsum("kdac,ikcd->ia", b["ovvv_c"], t2)
    if terms.get("ooov_mis"):
        # chem(kl|ic) = <ki|lc>  -> v[k,i,l,c];  -sum_klc * t2[k,l,a,c]
        val -= np.einsum("kilc,klac->ia", b["ooov_c"], t2)
    return val


def v_oovo_as_ooov(b):
    """<kl|ci> reordered to [k,l,i,c]:  <kl|ci> = <lk|ic> (bra-ket particle
    swap symmetry of real integrals: <pq|rs> = <qp|sr>)."""
    return b["ooov"].transpose(1, 0, 2, 3)  # <lk|ic> indexed [k,l,i,c] -> swap k,l


def solve_t1_fixed_t2(b, t2, terms, t1_init=None, tol=1e-12, maxiter=2000, damp=0.5):
    no, nv = t2.shape[0], t2.shape[2]
    Dia = b["eps_o"][:, None] - b["eps_v"][None, :]
    t1 = np.zeros((no, nv)) if t1_init is None else t1_init.copy()
    for it in range(maxiter):
        t1_new = t1_val(b, t1, t2, terms) / Dia
        dmax = np.abs(t1_new - t1).max()
        t1 = damp * t1_new + (1.0 - damp) * t1
        if dmax < tol:
            return t1, it, True
    return t1, maxiter, False


def report(tag, t1x, t1_ref):
    d = t1x - t1_ref
    n_ref = np.linalg.norm(t1_ref)
    print(f"  {tag:<28s} max|dT1|={np.abs(d).max():.3e}  "
          f"||dT1||/||T1*||={np.linalg.norm(d)/n_ref:.3f}  "
          f"(T1=0 baseline: 1.000)")


def t2_residual_terms(b, v, t1, t2, nocc, nmo):
    """Canonical T2 residual (eri_stored.cu:7050-7211) decomposed into
    T1-free part + individual T1-coupling terms. Returns dict of raw[i,j,a,b]
    contributions (pre-symmetrization)."""
    o = slice(0, nocc)
    vs = slice(nocc, nmo)
    oovv = b["oovv"]
    w_oovv = 2.0 * oovv - oovv.transpose(0, 1, 3, 2)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    C = {}

    # ---- Wklij pieces ----
    v_oooo = v[o, o, o, o]
    Wk_bare_t2 = v_oooo + np.einsum("klcd,ijcd->klij", oovv, t2)
    # v(l,k,C,i)*t1[j,c] + v(k,l,C,j)*t1[i,c]
    Wk_t1lin = (np.einsum("lkci,jc->klij", v[o, o, vs, o], t1)
                + np.einsum("klcj,ic->klij", v[o, o, vs, o], t1))
    Wk_t1t1 = np.einsum("klcd,ic,jd->klij", oovv, t1, t1)
    C["oooo_t1free"] = 0.5 * np.einsum("klij,klab->ijab", Wk_bare_t2, t2)
    # implemented in DLPNO: NONE of the t1 pieces of the oooo ladder
    C["MISS oooo Wdress_t1lin"] = 0.5 * np.einsum("klij,klab->ijab", Wk_t1lin, tau)
    C["MISS oooo Wdress_t1t1"] = 0.5 * np.einsum("klij,klab->ijab", Wk_t1t1, tau)
    C["MISS oooo tau_t1t1"] = 0.5 * np.einsum("klij,ka,lb->ijab", Wk_bare_t2, t1, t1)

    # ---- Wabcd pieces ----
    v_vvvv = v[vs, vs, vs, vs]
    Wab_t1 = -(np.einsum("kadc,kb->abcd", v[o, vs, vs, vs], t1)
               + np.einsum("kbcd,ka->abcd", v[o, vs, vs, vs], t1))
    C["vvvv_t1free"] = 0.5 * np.einsum("abcd,ijcd->ijab", v_vvvv, t2)
    C["IMPL vvvv tau_t1t1"] = 0.5 * np.einsum("abcd,ic,jd->ijab", v_vvvv, t1, t1)
    C["MISS vvvv Wdress_t1"] = 0.5 * np.einsum("abcd,ijcd->ijab", Wab_t1, tau)

    # ---- Lac / Lki ----
    Fki = np.einsum("klcd,ilcd->ki", w_oovv, t2 + np.einsum("ic,ld->ilcd", t1, t1))
    Fac = -np.einsum("klcd,klad->ac", w_oovv, t2 + np.einsum("ka,ld->klad", t1, t1))
    Lki_t1 = np.einsum("lkci,lc->ki", 2.0 * v[o, o, vs, o], t1) \
             - np.einsum("lkic,lc->ki", v[o, o, o, vs], t1)
    Lac_t1 = np.einsum("kadc,kd->ac", 2.0 * v[o, vs, vs, vs], t1) \
             - np.einsum("kacd,kd->ac", v[o, vs, vs, vs], t1)
    # t2-dressed L parts: how much is in DLPNO already? CCD-validated ->
    # count the pure-t2 Fki/Fac as t1free, all t1-borne parts as missing.
    Fki_t2only = np.einsum("klcd,ilcd->ki", w_oovv, t2)
    Fac_t2only = -np.einsum("klcd,klad->ac", w_oovv, t2)
    C["L_t1free"] = (np.einsum("ac,ijcb->ijab", Fac_t2only, t2)
                     - np.einsum("ki,kjab->ijab", Fki_t2only, t2))
    C["MISS L_t1parts"] = (np.einsum("ac,ijcb->ijab", (Fac - Fac_t2only) + Lac_t1, t2)
                           - np.einsum("ki,kjab->ijab", (Fki - Fki_t2only) + Lki_t1, t2))

    # ---- vvov / vooo linear + quadratic ----
    C["IMPL vvov_lin"] = np.einsum("abic,jc->ijab", v[vs, vs, o, vs], t1)
    C["IMPL vooo_lin"] = -np.einsum("akij,kb->ijab", v[vs, o, o, o], t1)
    C["MISS vvov_quad"] = -np.einsum("kbic,ka,jc->ijab", v[o, vs, o, vs], t1, t1)
    C["MISS vooo_quad"] = -np.einsum("akic,jc,kb->ijab", v[vs, o, o, vs], t1, t1)

    # ---- Wakic / Wakci ph ladders ----
    Wakic_bare_t2 = (v[vs, o, o, vs]
                     - 0.5 * np.einsum("lkdc,ilda->akic", oovv, t2)
                     + 0.5 * np.einsum("lkdc,ilad->akic", w_oovv, t2))
    Wakic_t1 = (-np.einsum("klci,la->akic", v[o, o, vs, o], t1)
                + np.einsum("kacd,id->akic", v[o, vs, vs, vs], t1)
                - np.einsum("lkdc,id,la->akic", oovv, t1, t1))
    Wakci_bare_t2 = (v[vs, o, vs, o]
                     - 0.5 * np.einsum("lkcd,ilda->akci", oovv, t2))
    Wakci_t1 = (-np.einsum("lkci,la->akci", v[o, o, vs, o], t1)
                + np.einsum("kadc,id->akci", v[o, vs, vs, vs], t1)
                - np.einsum("lkcd,id,la->akci", oovv, t1, t1))

    def ph_contract(Wic, Wci):
        return (2.0 * np.einsum("akic,kjcb->ijab", Wic, t2)
                - np.einsum("akci,kjcb->ijab", Wci, t2)
                - np.einsum("akic,kjbc->ijab", Wic, t2)
                - np.einsum("bkci,kjac->ijab", Wci, t2))

    C["ph_t1free"] = ph_contract(Wakic_bare_t2, Wakci_bare_t2)
    C["MISS ph_Wdress_t1"] = ph_contract(Wakic_t1, Wakci_t1)

    C["drv"] = 0.5 * v[o, o, vs, vs].transpose(0, 1, 2, 3)  # 0.5 <ij|ab>
    return C


def main():
    if len(sys.argv) >= 3:
        cases = [(sys.argv[1], sys.argv[2])]
    else:
        cases = [("xyz/Formaldehyde.xyz", "cc-pvdz"),
                 ("xyz/C2H4.xyz", "cc-pvdz")]

    for xyzf, basis in cases:
        print(f"\n=== {xyzf}  {basis}  cart=True  frozen=0 ===")
        xyz = load_xyz(xyzf)
        mf, mycc, v, nocc, nmo = canonical_ccsd(xyz, basis)
        eps = mf.mo_energy
        b = build_blocks(v, eps, nocc, nmo)
        t1s, t2s = mycc.t1, mycc.t2
        print(f"  nocc={nocc} nvir={nmo-nocc}  E_corr={mycc.e_corr:.10f}  "
              f"max|T1*|={np.abs(t1s).max():.4e}  ||T1*||={np.linalg.norm(t1s):.4e}")

        FULL = dict(fvv_dress=1, foo_dress=1, fkc=1, ring=1,
                    ovvv_t2=1, ovvv_t1t1=1, ooov_t2=1, ooov_t1t1=1)
        # 0) transcription check: fixed-point identity at (t1*, t2*)
        Dia = b["eps_o"][:, None] - b["eps_v"][None, :]
        resid = t1_val(b, t1s, t2s, FULL) / Dia - t1s
        print(f"  [check] canonical T1 fixed-point identity: max|res| = "
              f"{np.abs(resid).max():.3e}  (must be ~conv_tol)")

        IMPL = dict(ring=1, ovvv_t2=1, ooov_t2=1)
        variants = [
            ("impl (current DLPNO)", IMPL),
            ("impl+C (F dressings)", {**IMPL, "fvv_dress": 1, "foo_dress": 1}),
            ("impl+C+A (Fkc term)", {**IMPL, "fvv_dress": 1, "foo_dress": 1, "fkc": 1}),
            ("impl+C+A+B (=full)", FULL),
        ]
        print("  --- T1 fixed point with t2 frozen at t2* ---")
        for tag, terms in variants:
            t1x, it, ok = solve_t1_fixed_t2(b, t2s, terms)
            report(tag + ("" if ok else " [NOCONV]"), t1x, t1s)

        # C-only and A-only for ranking
        for tag, terms in [("impl+A only", {**IMPL, "fkc": 1}),
                           ("impl+B only", {**IMPL, "ovvv_t1t1": 1, "ooov_t1t1": 1})]:
            t1x, it, ok = solve_t1_fixed_t2(b, t2s, terms)
            report(tag + ("" if ok else " [NOCONV]"), t1x, t1s)

        # ---- as-coded (chemist-misread) variant: must reproduce the GANSU
        # zero-truncation Gate0 signature (cos ~ 0.26 for CH2O). ----
        MIS = dict(ring_mis=1, ovvv_mis=1, ooov_mis=1)
        t1m, it, ok = solve_t1_fixed_t2(b, t2s, MIS)
        nm, ns = np.linalg.norm(t1m), np.linalg.norm(t1s)
        cos = (t1m * t1s).sum() / (nm * ns)
        print(f"  impl-MISREAD (as-coded){'' if ok else ' [NOCONV]'}:  "
              f"||T1_mis||={nm:.4e}  ||T1*||={ns:.4e}  cos={cos:+.4f}")

        # ---- T2-side T1-coupling decomposition at (t1*, t2*) ----
        print("  --- T2 residual T1-couplings at (t1*,t2*): induced dT2 estimate ---")
        Cm = t2_residual_terms(b, v, t1s, t2s, nocc, nmo)
        Dijab = (b["eps_o"][:, None, None, None] + b["eps_o"][None, :, None, None]
                 - b["eps_v"][None, None, :, None] - b["eps_v"][None, None, None, :])
        n_t2 = np.linalg.norm(t2s)
        for k in sorted(Cm.keys()):
            if not (k.startswith("MISS") or k.startswith("IMPL")):
                continue
            r = Cm[k]
            rs = (r + r.transpose(1, 0, 3, 2)) / Dijab
            print(f"    {k:<28s} ||dT2_est||/||T2*|| = {np.linalg.norm(rs)/n_t2:.4f}"
                  f"   max = {np.abs(rs).max():.3e}")


if __name__ == "__main__":
    main()
