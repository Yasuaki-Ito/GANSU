#!/usr/bin/env python3
"""
Faithful port of the CFOUR-style STEOM W^eff (gmaei/gmaie) working equations
(megansimons/steom_ccsd-ct, see AQUA/STEOM_WEFF_CFOUR_REF.md), built directly
from RAW r2 amplitudes + explicit X(MI)/X(EA), with spinad = 2T - T.swap(0,1).

Verified term-by-term against the EXACT g_phhp/g_phph from singlet/triplet
EOM-EE (which equal STEOM at complete active space).

Index dictionary (CFOUR <-> PySCF bar_h, all 0-based):
  fem[E,M]      = Fov[M,E]
  wamef[E,F,A,M]= Wvovv[A,M,E,F]
  wmnie[M,N,I,E]= Wooov[M,N,I,E]
  wmnef[A,B,I,J]= eri_ovov[I,A,J,B]
  sabej[A,B,E,J]= r2_ea[E][J,A,B]      (E = active-vir root)
  smbij[I,J,M,B]= r2_ip[M][J,I,B]      (M = active-occ root; note J,I transpose)
  xea[E,A]=X_EA[E,A]   xmi[M,I]=X_IP[M,I]
  gmaei[A,I,B,J] -> g_phhp[B,J,I,A]    gmaie[A,I,B,J] -> g_phph[A,J,B,I]
"""
import sys
import numpy as np
sys.path.insert(0, "script")
from pyscf_steom_feff_reference import read_xyz, build_bar_h, build_x_matrices
from steom_full_active_test import principal_roots_ip, principal_roots_ea

np.set_printoptions(precision=5, suppress=True, linewidth=170)


def spinad(T):
    """2T - T.swapaxes(0,1): spin-adapt on leading same-type pair."""
    return 2.0 * T - T.swapaxes(0, 1)


def load(xyz, basis, ncore, atom=None, active=None):
    """atom overrides xyz (raw PySCF atom string); active = explicit active MO list
    (CAS window: freeze every occ/vir not in active), mirrors
    steom_fockspace_ref.get_active_data so the oracle and GANSU-analytic share a space."""
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd
    geom = atom if atom is not None else read_xyz(xyz)
    mol = gto.M(atom=geom, basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    nmo_tot = mf.mo_coeff.shape[1]; nocc_tot = mol.nelectron // 2
    if active is None:
        active = list(range(ncore, nmo_tot))
    active = sorted(active)
    frozen = [p for p in range(nmo_tot) if p not in active]
    mycc = cc.CCSD(mf, frozen=(frozen if frozen else 0))
    mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    t1, t2 = mycc.t1, mycc.t2
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc
    mo_c = mf.mo_coeff[:, active]; moe = mf.mo_energy[active]
    eri = ao2mo.kernel(mol, mo_c, compact=False).reshape(nmo, nmo, nmo, nmo)
    bar = build_bar_h(eri, t1, t2, np.diag(moe[:nocc]), np.diag(moe[nocc:]), nocc, nvir)
    # EE-consistent singles-block intermediates for the STEOM base fix (2026-07-06):
    # the correct s=0 base is 2 woVvO + woVVo (PySCF eom_rccsd make_ee), NOT the IP-side
    # 2 Wovvo - Wovov. Stash them so build_g_canonical_full can use them under STEOM_EE_BASE.
    _eeimds = eom_rccsd.EOMEESinglet(mycc).make_imds()
    bar["woVvO"] = np.asarray(_eeimds.woVvO)   # [m,b,e,j]
    bar["woVVo"] = np.asarray(_eeimds.woVVo)   # [m,b,e,j]
    # EE-convention 1-body intermediates (exact base F for the singles-singles
    # block; differ from IP-side Loo/Lvv by t1-dressing convention).
    bar["Foo_ee"] = np.asarray(_eeimds.Foo)
    bar["Fvv_ee"] = np.asarray(_eeimds.Fvv)
    dim = nocc * nvir

    # exact singlet/triplet effective singles H -> exact g_phph / g_phhp
    eom = eom_rccsd.EOMEESinglet(mycc); e_s, v_s = eom.kernel(nroots=dim)
    e_s = np.atleast_1d(e_s)
    R1 = np.zeros((dim, dim))
    for n in range(dim):
        r1, _ = eom.vector_to_amplitudes(v_s[n]); R1[:, n] = np.asarray(r1).reshape(dim)
    G_s = R1 @ np.diag(e_s[:dim]) @ np.linalg.inv(R1)
    eomT = eom_rccsd.EOMEETriplet(mycc); e_t, v_t = eomT.kernel(nroots=dim)
    e_t = np.atleast_1d(e_t)
    R1t = np.zeros((dim, dim))
    for n in range(dim):
        amp = eomT.vector_to_amplitudes(v_t[n])
        r1t = amp[0] if isinstance(amp, (tuple, list)) else amp
        R1t[:, n] = np.asarray(r1t).reshape(dim)
    G_t = R1t @ np.diag(e_t[:dim]) @ np.linalg.inv(R1t)

    exact_phhp = np.zeros((nvir, nocc, nocc, nvir))   # [b,j,i,a]
    exact_phph = np.zeros((nvir, nocc, nvir, nocc))   # [a,j,b,i]
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    exact_phhp[b, j, i, a] = (G_s[r, c] - G_t[r, c]) / 2.0
                    exact_phph[a, j, b, i] = -G_t[r, c]

    r1_ip, r2_ip, w_ip, occ_idx, _ = principal_roots_ip(mycc, nocc, nvir)
    r1_ea, r2_ea, w_ea, vir_idx, _ = principal_roots_ea(mycc, nocc, nvir)
    X_IP, X_EA = build_x_matrices(r1_ip, r1_ea, occ_idx, vir_idx)
    return dict(bar=bar, nocc=nocc, nvir=nvir, dim=dim, t1=t1, t2=t2,
                r1_ip=r1_ip, r1_ea=r1_ea,
                r2_ip=r2_ip, r2_ea=r2_ea, X_IP=X_IP, X_EA=X_EA,
                occ_idx=occ_idx, vir_idx=vir_idx, G_s=G_s, G_t=G_t,
                exact_phhp=exact_phhp, exact_phph=exact_phph,
                e_s=e_s[:dim], e_t=e_t[:dim])


def cfour_tensors(d):
    """Build CFOUR-indexed tensors from PySCF bar_h + raw r2."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    nE = len(d["r2_ea"]); nM = len(d["r2_ip"])
    fem = bar["Fov"].T.copy()                                  # [E,M]
    wamef = np.einsum("AMEF->EFAM", bar["Wvovv"]).copy()       # [E,F,A,M]
    wmnie = bar["Wooov"].copy()                                # [M,N,I,E]
    wmnef = np.einsum("IAJB->ABIJ", bar["eri_ovov"]).copy()    # [A,B,I,J]
    sabej = np.zeros((nvir, nvir, nE, nocc))
    for E in range(nE):
        sabej[:, :, E, :] = np.einsum("JAB->ABJ", d["r2_ea"][E])
    smbij = np.zeros((nocc, nocc, nM, nvir))
    for M in range(nM):
        smbij[:, :, M, :] = np.einsum("JIB->IJB", d["r2_ip"][M])  # smbij[I,J,M,B]=r2[J,I,B]
    return dict(fem=fem, wamef=wamef, wmnie=wmnie, wmnef=wmnef,
                sabej=sabej, smbij=smbij)


# ---------------------------------------------------------------- single routes
def build_ujaei(T):
    """ujaei_final, ujaie_final  [E,I,B,J]."""
    fem = T["fem"]; wamef = T["wamef"]; wmnie = T["wmnie"]; sabej = T["sabej"]
    # A: raw sabej
    A = np.einsum("FBEJ,FI->EIBJ", sabej, fem)
    A += np.einsum("GFEJ,FGBI->EIBJ", sabej, wamef)
    A -= np.einsum("FBEN,NIJF->EIBJ", sabej, wmnie)
    # B: spinad'd sabej (swap A,B i.e. axes 0,1) ; wmnie spinad in term3
    ss = spinad(sabej)
    wn = spinad(wmnie)
    B = np.einsum("BFEJ,FI->EIBJ", ss, fem)
    B += np.einsum("GFEJ,GFBI->EIBJ", ss, wamef)
    B += np.einsum("BFEN,INJF->EIBJ", ss, wn)
    ujaei = 1.5 * A - 0.5 * B
    ujaie = (1.0 / 3.0) * ujaei + (2.0 / 3.0) * B   # = 0.5A + 0.5B
    return ujaei, ujaie


def build_umabi(T):
    """umabi_final, umaib_final  [A,M,B,J]  (single-route, no cross)."""
    fem = T["fem"]; wamef = T["wamef"]; wmnie = T["wmnie"]; smbij = T["smbij"]
    A = -np.einsum("NJMB,AN->AMBJ", smbij, fem)
    A += np.einsum("ONMB,NOJA->AMBJ", smbij, wmnie)
    A -= np.einsum("NJMF,FABN->AMBJ", smbij, wamef)
    ss = spinad(smbij)
    wf = spinad(wamef)
    B = -np.einsum("JNMB,AN->AMBJ", ss, fem)
    B += np.einsum("ONMB,ONJA->AMBJ", ss, wmnie)
    B += np.einsum("JNMF,AFBN->AMBJ", ss, wf)
    umabi = 1.5 * A - 0.5 * B
    umaib = (1.0 / 3.0) * umabi + (2.0 / 3.0) * B
    return umabi, umaib


def build_uei_uam_uijke(T):
    wmnef = T["wmnef"]; sabej = T["sabej"]; smbij = T["smbij"]
    wsp = spinad(wmnef)
    uei = np.einsum("GFIM,GFEM->EI", wsp, sabej)                # [E,I]
    uam = -np.einsum("AEON,ONME->AM", wsp, smbij)              # [A,M]
    uijke = np.einsum("GFJI,GFEK->IJKE", wmnef, sabej)         # [I,J,K,E]
    return uei, uam, uijke


def build_uajmi(T, variant=0):
    """S^IP hhhp uajmi/uajim [F,N,M,J] (v,o,act,o) from wmnef.smbij.
    Index pattern ambiguous -> 'variant' selects which occ pair contracts."""
    wmnef = T["wmnef"]; smbij = T["smbij"]   # wmnef[A,B,I,J] ; smbij[I,J,M,B]
    # contract vir B(wmnef)=B(smbij) and one occ pair; keep F=A, M, two occ out
    if variant == 0:   # out [F,N=I,M,J=L]
        uajmi = np.einsum("FBIK,KLMB->FIML", wmnef, smbij)
    elif variant == 1:
        uajmi = np.einsum("FBIK,LKMB->FIML", wmnef, smbij)
    elif variant == 2:
        uajmi = np.einsum("FBKI,KLMB->FIML", wmnef, smbij)
    elif variant == 3:
        uajmi = np.einsum("FBKI,LKMB->FIML", wmnef, smbij)
    return uajmi


def build_umaei(T, uei, uam, uijke, uajmi, uajim):
    """Cross umaei_final, umaie_final [E,M,B,J]."""
    sabej = T["sabej"]; smbij = T["smbij"]
    A = np.einsum("FBEJ,FM->EMBJ", sabej, uam)
    A -= np.einsum("FBEN,FNMJ->EMBJ", sabej, uajmi)
    ss = spinad(sabej)
    B = np.einsum("BFEJ,FM->EMBJ", ss, uam)
    uajim_mix = 2.0 * uajim - uajmi
    B += np.einsum("BFEN,FNMJ->EMBJ", ss, uajim_mix)
    A -= np.einsum("NJMB,EN->EMBJ", smbij, uei)
    A += np.einsum("ONMB,NOJE->EMBJ", smbij, uijke)
    sm = spinad(smbij)
    B -= np.einsum("JNMB,EN->EMBJ", sm, uei)
    B += np.einsum("ONMB,ONJE->EMBJ", sm, uijke)
    umaei = 1.5 * A - 0.5 * B
    umaie = (1.0 / 3.0) * umaei + (2.0 / 3.0) * B
    return umaei, umaie


def assemble_g(d, T, ujaei, ujaie, umabi, umaib):
    """gmaei/gmaie -> g_phhp/g_phph, then full singlet G."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    X_EA = d["X_EA"]; X_IP = d["X_IP"]
    # base (derived so g_phhp[b,j,i,a]=Wovvo[j,b,a,i], g_phph[a,j,b,i]=Wovov[j,a,i,b]):
    #   gmaei[A,I,B,J]=Wovvo[J,B,A,I];  gmaie[A,I,B,J]=Wovov[J,A,I,B]
    gmaei = np.einsum("JBAI->AIBJ", bar["Wovvo"]).copy()
    gmaie = np.einsum("JAIB->AIBJ", bar["Wovov"]).copy()
    gmaei += np.einsum("EIBJ,EA->AIBJ", ujaei, X_EA)
    gmaie += np.einsum("EIBJ,EA->AIBJ", ujaie, X_EA)
    gmaei -= np.einsum("AMBJ,MI->AIBJ", umabi, X_IP)
    gmaie -= np.einsum("AMBJ,MI->AIBJ", umaib, X_IP)
    # map to g_phhp[b,j,i,a]=gmaei[a,i,b,j]; g_phph[a,j,b,i]=gmaie[a,i,b,j]
    g_phhp = np.einsum("AIBJ->BJIA", gmaei)
    g_phph = np.einsum("AIBJ->AJBI", gmaie)
    return g_phhp, g_phph


def report(d, g_phhp, g_phph):
    nocc = d["nocc"]; nvir = d["nvir"]
    # off-diagonal mask i!=j,a!=b
    def res(code, exact, tag):
        w = 0.0
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        if i == j or a == b:
                            continue
                        if tag == "phhp":
                            w = max(w, abs(code[b, j, i, a] - exact[b, j, i, a]))
                        else:
                            w = max(w, abs(code[a, j, b, i] - exact[a, j, b, i]))
        return w
    rhp = res(g_phhp, d["exact_phhp"], "phhp")
    rph = res(g_phph, d["exact_phph"], "phph")
    print(f"  worst|Δ g_phhp|={rhp:.5e}   worst|Δ g_phph|={rph:.5e}")
    return rhp, rph


def build_feff(d):
    """F_eff_oo / F_eff_vv via verified normalized-s path."""
    from pyscf_steom_feff_reference import build_normalized_s
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    d["occ_idx"], d["vir_idx"], nocc, nvir)
    Fov = bar["Fov"]; Wooov = bar["Wooov"]; Wvovv = bar["Wvovv"]
    Foo = bar["Loo"].copy()
    for m in range(len(s_IP)):
        s = s_IP[m]; st = 2 * s - s.transpose(1, 0, 2)
        Foo[d["occ_idx"][m], :] += np.einsum("kc,ikc->i", Fov, st) - np.einsum("klid,kld->i", Wooov, st)
    Fvv = bar["Lvv"].copy()
    for e in range(len(s_EA)):
        s = s_EA[e]; st = 2 * s - s.transpose(0, 2, 1)
        Fvv[d["vir_idx"][e], :] += np.einsum("kc,kac->a", Fov, st) + np.einsum("alcd,lcd->a", Wvovv, st)
    return Foo, Fvv


def full_G_eigs(d, g_phhp, g_phph):
    nocc = d["nocc"]; nvir = d["nvir"]; dim = d["dim"]
    Foo, Fvv = build_feff(d)
    G = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    v = 2.0 * g_phhp[b, j, i, a] - g_phph[a, j, b, i]
                    if i == j: v += Fvv[a, b]
                    if a == b: v -= Foo[i, j]
                    G[r, c] = v
    return np.sort(np.linalg.eigvals(G).real)


def main(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=2):
    d = load(xyz, basis, ncore)
    T = cfour_tensors(d)
    # numerical self-check of base mapping (dressing off)
    g0_hp, g0_ph = assemble_g(d, T, np.zeros((len(d["r2_ea"]), d["nocc"], d["nvir"], d["nocc"])),
                              np.zeros((len(d["r2_ea"]), d["nocc"], d["nvir"], d["nocc"])),
                              np.zeros((d["nvir"], len(d["r2_ip"]), d["nvir"], d["nocc"])),
                              np.zeros((d["nvir"], len(d["r2_ip"]), d["nvir"], d["nocc"])))
    print(f"{xyz} {basis} ncore={ncore}  nocc={d['nocc']} nvir={d['nvir']}")
    print("base-only (no dressing):"); report(d, g0_hp, g0_ph)
    ujaei, ujaie = build_ujaei(T)
    umabi, umaib = build_umabi(T)
    g_hp, g_ph = assemble_g(d, T, ujaei, ujaie, umabi, umaib)
    print("CFOUR single-route (no cross):"); report(d, g_hp, g_ph)

    uei, uam, uijke = build_uei_uam_uijke(T)
    X_EA = d["X_EA"]
    for v in range(4):
        uajmi = build_uajmi(T, v); uajim = build_uajmi(T, v)
        umaei, umaie = build_umaei(T, uei, uam, uijke, uajmi, uajim)
        # fold cross into umabi/umaib via xea
        umabi_t = umabi + np.einsum("EMBJ,EA->AMBJ", umaei, X_EA)
        umaib_t = umaib + np.einsum("EMBJ,EA->AMBJ", umaie, X_EA)
        g_hp, g_ph = assemble_g(d, T, ujaei, ujaie, umabi_t, umaib_t)
        eigs = full_G_eigs(d, g_hp, g_ph)
        dmax = np.max(np.abs(eigs - d["e_s"])) * 1000
        print(f"CFOUR full v{v}: eig Δmax={dmax:.3f} mHa  ", end=""); report(d, g_hp, g_ph)
    # also: eigenvalues using EXACT element-wise g (sanity: must equal EOM)
    geig = full_G_eigs(d, d["exact_phhp"], d["exact_phph"])
    print("  sanity exact-g eigs Δmax(mHa)=", np.max(np.abs(geig - d["e_s"])) * 1000)
    print("  EOM e_s:", d["e_s"])
    print("  exactg :", geig)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    main(*( [a[0]] if len(a) > 0 else []),
         **({"basis": a[1]} if len(a) > 1 else {}),
         **({"ncore": int(a[2])} if len(a) > 2 else {}))
