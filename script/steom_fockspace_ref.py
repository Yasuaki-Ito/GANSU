#!/usr/bin/env python3
"""
DETERMINANT (Fock-space) ground-truth for STEOM, to settle the W^eff cross-term
sign/magnitude with ZERO formula derivation.

Build, in the active spin-orbital determinant basis:
  H   = bare active-space Hamiltonian (matrix in det basis)   -> validate vs FCI
  Hbar= exp(-T) H exp(T)  (CCSD similarity transform)          -> validate <0|Hbar|0>=E_corr
  G   = exp(-S) Hbar exp(S)  (STEOM second transform, S=S^IP+S^EA)
  project G onto 1h1p singles -> STEOM G^{1h1p}; eigenvalues vs ORCA.

Everything is explicit matrix algebra (expm + matrix commutators); no analytic
u-intermediate formula is used, so the result is an INDEPENDENT reference for
build_g_canonical_full. Tiny system (H2O sto-3g frozen=2: 10 spin-orbitals,
6 e-, C(10,6)=210 dets) makes the full matrices trivial.

Run /tmp data producer first (see steom_fockspace_probe in the session) or this
script regenerates it.  Usage:  wsl python3 script/steom_fockspace_ref.py
"""
import os
# force single-threaded LAPACK: degenerate IP/EA eigenspaces make multithreaded
# np.linalg.eig non-deterministic (arbitrary basis within degenerate pairs).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, itertools
import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------- integrals
def get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=2, atom=None, active=None):
    """Active-space integrals + CCSD(t1,t2) + FCI for the STEOM determinant oracle.

    xyz/atom : geometry from file (xyz) or a raw PySCF atom string (atom, overrides xyz).
    ncore    : freeze the lowest `ncore` spatial MOs (all higher MOs active).
    active   : explicit list of active spatial-MO indices (energy order). When given it
               OVERRIDES ncore: every occupied MO not in `active` becomes frozen-core
               (doubly-occ, folded into Dcore) and every virtual MO not in `active` is
               deleted. Enables a CAS window, e.g. butadiene pi-CAS(4,4)."""
    from pyscf import gto, scf, cc, ao2mo, fci
    def read_xyz(fn):
        L = open(fn).read().splitlines(); n = int(L[0]); return "\n".join(L[2:2+n])
    geom = atom if atom is not None else read_xyz(xyz)
    mol = gto.M(atom=geom, basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
    nmo = mf.mo_coeff.shape[1]
    nocc_tot = mol.nelectron // 2
    if active is None:
        active = list(range(ncore, nmo))
    active = sorted(active)
    frozen_occ = [p for p in range(nocc_tot) if p not in active]
    frozen_vir = [p for p in range(nocc_tot, nmo) if p not in active]
    frozen = frozen_occ + frozen_vir
    nact = len(active)
    C_act = mf.mo_coeff[:, active]; C_core = mf.mo_coeff[:, frozen_occ]
    Dcore = 2.0 * C_core @ C_core.T
    hcore = mf.get_hcore()
    vj, vk = scf.hf.get_jk(mol, Dcore)
    heff = hcore + vj - 0.5 * vk
    h1 = C_act.T @ heff @ C_act
    eri = ao2mo.kernel(mol, C_act, compact=False).reshape(nact, nact, nact, nact)
    Ecore = np.einsum("ij,ji->", Dcore, hcore) + 0.5 * np.einsum("ij,ji->", Dcore, vj - 0.5 * vk)
    Enuc = mol.energy_nuc()
    nelec_act = mol.nelectron - 2 * len(frozen_occ)
    mycc = cc.CCSD(mf, frozen=(frozen if frozen else 0))
    mycc.conv_tol = 1e-10; mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    cis = fci.direct_spin1.FCI()
    Efci, _ = cis.kernel(h1, eri, nact, nelec_act, ecore=Ecore + Enuc)
    nocc = nelec_act // 2; nvir = nact - nocc
    return dict(h1=h1, eri=eri, nact=nact, nocc=nocc, nvir=nvir, nelec=nelec_act,
                Ecore=Ecore, Enuc=Enuc, Ehf=mf.e_tot, Eccsd=mycc.e_tot, Efci=Efci,
                t1=mycc.t1, t2=mycc.t2, moe=mf.mo_energy[active], mycc=mycc, mf=mf)


# ------------------------------------------------------ determinant Fock space
# spin-orbital index: spatial p (0..nact-1), spin s in {0=a,1=b} -> P = p + s*nact
# determinant = python int bitmask over 2*nact spin-orbitals.
def popcount(x): return bin(x).count("1")

def build_dets(norb_so, nelec):
    """all determinants (bitmasks) with exactly nelec occupied spin-orbitals."""
    dets = []
    for occ in itertools.combinations(range(norb_so), nelec):
        m = 0
        for o in occ: m |= (1 << o)
        dets.append(m)
    dets.sort()
    index = {d: i for i, d in enumerate(dets)}
    return dets, index

def apply_ann(det, p):
    """a_p |det> -> (sign, newdet) or (0, 0) if unoccupied."""
    if not (det >> p) & 1: return 0, 0
    sign = (-1) ** popcount(det & ((1 << p) - 1))   # parity of occ below p
    return sign, det & ~(1 << p)

def apply_cre(det, p):
    """a^dag_p |det> -> (sign, newdet) or (0,0) if already occupied."""
    if (det >> p) & 1: return 0, 0
    sign = (-1) ** popcount(det & ((1 << p) - 1))
    return sign, det | (1 << p)

def so_index(p, s, nact): return p + s * nact


def build_H(data, dets, index):
    """bare H matrix: H1 = Σ h[P,Q] a†_P a_Q ; H2 = ½ Σ <PQ|RS> a†_P a†_Q a_S a_R."""
    nact = data["nact"]; h1 = data["h1"]; eri = data["eri"]
    nso = 2 * nact; N = len(dets)
    # spin-orbital one-body
    hso = np.zeros((nso, nso))
    for s in range(2):
        for p in range(nact):
            for q in range(nact):
                hso[so_index(p, s, nact), so_index(q, s, nact)] = h1[p, q]
    # physicist <PQ|RS> = (PR|QS)_chem with spin deltas
    H = np.zeros((N, N))
    # one-body
    for J, det in enumerate(dets):
        for Q in range(nso):
            s1, d1 = apply_ann(det, Q)
            if s1 == 0: continue
            for P in range(nso):
                s2, d2 = apply_cre(d1, P)
                if s2 == 0: continue
                if hso[P, Q] != 0.0:
                    H[index[d2], J] += s1 * s2 * hso[P, Q]
    # two-body  ½ Σ <PQ|RS> a†P a†Q aS aR
    # precompute spin-orbital physicist integral
    def chem(P, R, Q, S):
        # returns (PR|QS) with spin: need σP=σR and σQ=σS
        sP, sR, sQ, sS = P // nact, R // nact, Q // nact, S // nact
        if sP != sR or sQ != sS: return 0.0
        return eri[P % nact, R % nact, Q % nact, S % nact]
    for J, det in enumerate(dets):
        for R in range(nso):
            sa, da = apply_ann(det, R)
            if sa == 0: continue
            for S in range(nso):
                sb, db = apply_ann(da, S)
                if sb == 0: continue
                for Q in range(nso):
                    sc, dc = apply_cre(db, Q)
                    if sc == 0: continue
                    for P in range(nso):
                        sd, dd = apply_cre(dc, P)
                        if sd == 0: continue
                        v = chem(P, R, Q, S)   # <PQ|RS>=(PR|QS)
                        if v != 0.0:
                            H[index[dd], J] += 0.5 * sa * sb * sc * sd * v
    return H, hso


def hf_det(data):
    nact = data["nact"]; nocc = data["nocc"]
    m = 0
    for p in range(nocc):
        m |= (1 << so_index(p, 0, nact)) | (1 << so_index(p, 1, nact))
    return m


# ------------------------------------------------------------- excitation ops
def occ_so(data):
    nact = data["nact"]; nocc = data["nocc"]
    return [so_index(p, s, nact) for s in range(2) for p in range(nocc)]

def vir_so(data):
    nact = data["nact"]; nocc = data["nocc"]; nvir = data["nvir"]
    return [so_index(p, s, nact) for s in range(2) for p in range(nocc, nact)]

def spin_of(P, nact): return P // nact
def spat_of(P, nact): return P % nact

def build_t_so(data):
    """spin-orbital t1so[I,A], t2so[I,J,A,B] from closed-shell RCCSD t1,t2.
    I,J occ-so ; A,B vir-so. t2so[I,J,A,B] = δsIsA δsJsB t2[i,j,a,b]
                                            - δsIsB δsJsA t2[i,j,b,a]."""
    nact = data["nact"]; nocc = data["nocc"]
    t1 = data["t1"]; t2 = data["t2"]
    occ = occ_so(data); vir = vir_so(data)
    nso = 2 * nact
    t1so = np.zeros((nso, nso))   # [I,A]
    for I in occ:
        for A in vir:
            if spin_of(I, nact) == spin_of(A, nact):
                t1so[I, A] = t1[spat_of(I, nact), spat_of(A, nact) - nocc]
    t2so = np.zeros((nso, nso, nso, nso))  # [I,J,A,B]
    for I in occ:
        for J in occ:
            for A in vir:
                for B in vir:
                    i, j = spat_of(I, nact), spat_of(J, nact)
                    a, b = spat_of(A, nact) - nocc, spat_of(B, nact) - nocc
                    sI, sJ, sA, sB = (spin_of(I, nact), spin_of(J, nact),
                                      spin_of(A, nact), spin_of(B, nact))
                    v = 0.0
                    if sI == sA and sJ == sB: v += t2[i, j, a, b]
                    if sI == sB and sJ == sA: v -= t2[i, j, b, a]
                    t2so[I, J, A, B] = v
    return t1so, t2so

def build_excitation_matrix(data, dets, index, amp_so, kind):
    """T1: Σ t1so[I,A] a†_A a_I ; T2: ¼ Σ t2so[I,J,A,B] a†_A a†_B a_J a_I.
    kind in {'t1','t2'}.  amp_so = t1so or t2so."""
    nso = 2 * data["nact"]; N = len(dets)
    M = np.zeros((N, N))
    occ = occ_so(data); vir = vir_so(data)
    if kind == "t1":
        for Jc, det in enumerate(dets):
            for I in occ:
                s1, d1 = apply_ann(det, I)
                if s1 == 0: continue
                for A in vir:
                    if amp_so[I, A] == 0.0: continue
                    s2, d2 = apply_cre(d1, A)
                    if s2 == 0: continue
                    M[index[d2], Jc] += s1 * s2 * amp_so[I, A]
    else:
        for Jc, det in enumerate(dets):
            for I in occ:
                sa, da = apply_ann(det, I)
                if sa == 0: continue
                for J in occ:
                    sb, db = apply_ann(da, J)
                    if sb == 0: continue
                    for B in vir:
                        sc, dc = apply_cre(db, B)
                        if sc == 0: continue
                        for A in vir:
                            v = amp_so[I, J, A, B]
                            if v == 0.0: continue
                            sd, dd = apply_cre(dc, A)
                            if sd == 0: continue
                            M[index[dd], Jc] += 0.25 * sa * sb * sc * sd * v
    return M

def singles_dets(data, dets, index):
    """map (I_occ_so, A_vir_so) -> determinant a†_A a_I |HF>, with sign convention
    so that <det| a†_A a_I |HF> = +1. Returns dict and list."""
    hf = hf_det(data); occ = occ_so(data); vir = vir_so(data)
    out = {}
    for I in occ:
        for A in vir:
            s1, d1 = apply_ann(hf, I)
            s2, d2 = apply_cre(d1, A)
            if s1 * s2 != 0:
                out[(I, A)] = (d2, s1 * s2)
    return out


def apply_string(det, ops):
    """apply a sequence of (kind,p) right-to-left; kind 'c'=create 'a'=annihilate.
    returns (sign, newdet) or (0,0)."""
    sgn = 1; d = det
    for kind, p in reversed(ops):
        if kind == "a":
            s, d = apply_ann(d, p)
        else:
            s, d = apply_cre(d, p)
        if s == 0: return 0, 0
        sgn *= s
    return sgn, d

def build_sector(data, nelec):
    """H, T matrices and Hbar=exp(-T)H exp(T) in the nelec sector."""
    nso = 2 * data["nact"]
    dets, index = build_dets(nso, nelec)
    H, _ = build_H(data, dets, index)
    t1so, t2so = build_t_so(data)
    T = (build_excitation_matrix(data, dets, index, t1so, "t1")
         + build_excitation_matrix(data, dets, index, t2so, "t2"))
    Hbar = expm(-T) @ H @ expm(T)
    return dets, index, Hbar

def solve_ip(data, E_N):
    """IP-EOM-CCSD: diagonalize Hbar in the TRUNCATED 1h+2h1p subspace (NOT the
    full (N-1) sector — that would give exact-CI, not EOM-CCSD). Returns per
    active-occ-so s_IP[m][I,J,B] via Eq30 s=-R2 R1^{-1}."""
    nso = 2 * data["nact"]
    occ = occ_so(data); vir = vir_so(data); hf = hf_det(data)
    dets, index, Hbar = build_sector(data, data["nelec"] - 1)
    # p-space (1h): a_I|HF>  ; q-space (2h1p): a†_B a_I a_J|HF> (I<J)
    p_dets = {}; p_list = []
    for I in occ:
        s, d = apply_ann(hf, I)
        p_dets[I] = (len(p_list), s); p_list.append(index[d])
    q_keys = []; q_list = []
    seen = set(p_list)
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            I, J = occ[ii], occ[jj]
            for B in vir:
                sg, d = apply_string(hf, [("c", B), ("a", I), ("a", J)])
                if sg == 0 or index[d] in seen: continue
                seen.add(index[d]); q_keys.append((I, J, B, sg)); q_list.append(index[d])
    sub = p_list + q_list
    Hsub = Hbar[np.ix_(sub, sub)]
    w, vr = np.linalg.eig(Hsub)
    no = len(occ)
    weights = sorted(((sum(abs(vr[p_dets[I][0], k])**2 for I in occ), k)
                      for k in range(len(w))), reverse=True)
    roots = [k for _, k in weights[:no]]
    R1 = np.zeros((no, no), dtype=complex); R2 = []
    for r, k in enumerate(roots):
        for mi, I in enumerate(occ):
            pos, s = p_dets[I]; R1[r, mi] = vr[pos, k] * s
        r2 = np.zeros((nso, nso, nso), dtype=complex)
        for qi, (I, J, B, sg) in enumerate(q_keys):
            amp = vr[no + qi, k] * sg
            r2[I, J, B] += amp; r2[J, I, B] -= amp        # antisymmetrize (I,J)
        R2.append(r2)
    R1inv = np.linalg.inv(R1)   # R1[root,orb]; s^m = -Σ_k R1inv[m,k] R2^k
    sIP = {}
    for mi, m in enumerate(occ):
        acc = np.zeros((nso, nso, nso), dtype=complex)
        for r in range(no):
            acc += R1inv[mi, r] * R2[r]
        sIP[m] = (-acc).real
    return sIP

def solve_ea(data):
    """EA-EOM-CCSD: Hbar in truncated 1p+2p1h subspace."""
    nso = 2 * data["nact"]
    occ = occ_so(data); vir = vir_so(data); hf = hf_det(data)
    dets, index, Hbar = build_sector(data, data["nelec"] + 1)
    p_dets = {}; p_list = []
    for A in vir:
        s, d = apply_cre(hf, A)
        p_dets[A] = (len(p_list), s); p_list.append(index[d])
    q_keys = []; q_list = []; seen = set(p_list)
    for ai in range(len(vir)):
        for bi in range(ai + 1, len(vir)):
            A, B = vir[ai], vir[bi]
            for J in occ:
                sg, d = apply_string(hf, [("c", A), ("c", B), ("a", J)])
                if sg == 0 or index[d] in seen: continue
                seen.add(index[d]); q_keys.append((J, A, B, sg)); q_list.append(index[d])
    sub = p_list + q_list
    Hsub = Hbar[np.ix_(sub, sub)]
    w, vr = np.linalg.eig(Hsub)
    nv = len(vir)
    weights = sorted(((sum(abs(vr[p_dets[A][0], k])**2 for A in vir), k)
                      for k in range(len(w))), reverse=True)
    roots = [k for _, k in weights[:nv]]
    R1 = np.zeros((nv, nv), dtype=complex); R2 = []
    for r, k in enumerate(roots):
        for ai, A in enumerate(vir):
            pos, s = p_dets[A]; R1[r, ai] = vr[pos, k] * s
        r2 = np.zeros((nso, nso, nso), dtype=complex)   # [J,A,B]
        for qi, (J, A, B, sg) in enumerate(q_keys):
            amp = vr[nv + qi, k] * sg
            r2[J, A, B] += amp; r2[J, B, A] -= amp        # antisymmetrize (A,B)
        R2.append(r2)
    R1inv = np.linalg.inv(R1)
    sEA = {}
    for ei, e in enumerate(vir):
        acc = np.zeros((nso, nso, nso), dtype=complex)
        for r in range(nv):
            acc += R1inv[ei, r] * R2[r]
        sEA[e] = (+acc).real
    return sEA

def _add_op(S, dets, index, ops, coeff):
    for Jc, det in enumerate(dets):
        sg, d = apply_string(det, ops)
        if sg == 0: continue
        S[index[d], Jc] += coeff * sg

def build_S(data, dets, index, sIP, sEA):
    """NORMAL-ORDERED S (Nooijen Eq 9-10).
    {m̂^†î b̂^†ĵ} = a†_m a_I a†_B a_J - δ_mI a†_B a_J + δ_mJ a†_B a_I  (Wick, ref |HF>)
    {â^†ê b̂^†ĵ} = a†_A a_e a†_B a_J - δ_eB a†_A a_J."""
    occ = occ_so(data); vir = vir_so(data); N = len(dets)
    S = np.zeros((N, N))
    for m in occ:
        amp = sIP[m]
        for I in occ:
            for J in occ:
                for B in vir:
                    v = amp[I, J, B]
                    if abs(v) < 1e-14: continue
                    _add_op(S, dets, index, [("c", m), ("a", I), ("c", B), ("a", J)], 0.5 * v)
                    if m == I: _add_op(S, dets, index, [("c", B), ("a", J)], -0.5 * v)
                    if m == J: _add_op(S, dets, index, [("c", B), ("a", I)], +0.5 * v)
    for e in vir:
        amp = sEA[e]
        for J in occ:
            for A in vir:
                for B in vir:
                    v = amp[J, A, B]
                    if abs(v) < 1e-14: continue
                    _add_op(S, dets, index, [("c", A), ("a", e), ("c", B), ("a", J)], 0.5 * v)
                    if e == B: _add_op(S, dets, index, [("c", A), ("a", J)], -0.5 * v)
    return S

def project_1h1p(data, dets, index, G):
    """spin-adapt the Sz=0 1h1p block of G into singlet/triplet spatial blocks."""
    nact = data["nact"]; nocc = data["nocc"]; nvir = data["nvir"]; hf = hf_det(data)
    # singles det for spatial (i,a) and spin s: a†_{a,s} a_{i,s}|HF>
    def sdet(i, a, s):
        I = so_index(i, s, nact); A = so_index(a + nocc, s, nact)
        sg, d = apply_string(hf, [("c", A), ("a", I)])
        return index[d], sg
    dim = nocc * nvir
    # build 12x12-ish via alpha/beta; assemble singlet/triplet directly
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    rows = [(i, a) for i in range(nocc) for a in range(nvir)]
    for r, (i, a) in enumerate(rows):
        ida, sga = sdet(i, a, 0); idb, sgb = sdet(i, a, 1)
        for c, (j, b) in enumerate(rows):
            jda, ta = sdet(j, b, 0); jdb, tb = sdet(j, b, 1)
            # G matrix element <bra|G|ket>; bra = singlet/triplet combo of (i,a)
            # singlet ket = (|j b a> + |j b b>)/√2 ; <ia,aa| applies
            aa = G[ida, jda] * sga * ta
            bb = G[idb, jdb] * sgb * tb
            ab = G[ida, jdb] * sga * tb
            ba = G[idb, jda] * sgb * ta
            Gs[r, c] = 0.5 * (aa + bb + ab + ba)   # singlet
            Gt[r, c] = 0.5 * (aa + bb - ab - ba)   # triplet (Sz=0)
    return Gs, Gt

def steom_main():
    Ha2eV = 27.211386245988
    data = get_active_data()
    print(f"nact={data['nact']} nocc={data['nocc']} nvir={data['nvir']}  "
          f"HF={data['Ehf']:.6f} CCSD={data['Eccsd']:.6f}")
    dets, index, HbarN = build_sector(data, data["nelec"])
    hf = hf_det(data); iHF = index[hf]
    E_N = HbarN[iHF, iHF]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    print(f"built sIP ({len(sIP)} active occ-so), sEA ({len(sEA)} active vir-so)")
    S = build_S(data, dets, index, sIP, sEA)
    # STEOM second transform sign: e^{+S} Hbar e^{-S} (decoupling convention,
    # validated: ncore=1 H2O sto-3g IROOT1 = 11.847 = ORCA 11.849).  The opposite
    # sign e^{-S}..e^{+S} over-dresses ~2x (gives a spurious 15.4 eV lowest state).
    G = expm(S) @ HbarN @ expm(-S)
    Gs, Gt = project_1h1p(data, dets, index, G)
    es = np.sort(np.linalg.eigvals(Gs).real - E_N) * Ha2eV
    et = np.sort(np.linalg.eigvals(Gt).real - E_N) * Ha2eV
    print("\nDETERMINANT STEOM 1h1p eigenvalues (eV):")
    print("  singlet:", np.round(es, 3))
    print("  triplet:", np.round(et, 3))
    # compare to EOM-EE and my analytic
    import steom_cfour_weff as C
    d = C.load("xyz/H2O.xyz", "sto-3g", 2)
    print("  EOM-EE singlet:", np.round(np.sort(d['e_s']) * Ha2eV, 3))
    print("  EOM-EE triplet:", np.round(np.sort(d['e_t']) * Ha2eV, 3))
    from pyscf_steom_feff_reference import build_g_canonical_full
    import steom_cfour_weff as CC
    from steom_route_probe import route_tensors
    Ga, g_ph_a, g_hp_a, u_amei, u_bmje, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    print("  ANALYTIC singlet:", np.round(np.sort(np.linalg.eigvals(Ga).real) * Ha2eV, 3))

    # ----- shift determinant G blocks to ground (subtract E_N) and compare -----
    nocc, nvir = d["nocc"], d["nvir"]
    GsD = Gs - E_N * np.eye(nocc * nvir)
    GtD = Gt - E_N * np.eye(nocc * nvir)
    # analytic singlet matrix = Ga ; analytic triplet matrix:
    Foo, Fvv = CC.build_feff(d)
    GtA = np.zeros((nocc * nvir, nocc * nvir))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    v = -g_ph_a[a, j, b, i]
                    if i == j: v += Fvv[a, b]
                    if a == b: v -= Foo[i, j]
                    GtA[r, c] = v
    # NB ordering: determinant rows use (i,a) with r=i*nvir+a — matches analytic.
    Dt = GtD - GtA
    Ds = GsD - Ga
    print(f"\n  ||Gt_det - Gt_analytic|| = {np.linalg.norm(Dt):.4f}  "
          f"||Gs_det - Gs_analytic|| = {np.linalg.norm(Ds):.4f}")
    # the triplet difference = -(g_phph_true - g_phph_mine). Compare to my CROSS tensor.
    base, t_amci, t_akei = route_tensors(d)
    mycross = g_ph_a - (base + t_amci + t_akei)   # u_amei contribution to g_phph
    # map mycross[a,j,b,i] into matrix form Mcross[r=i*nvir+a, c=j*nvir+b] as -g enters Gt
    Mcross = np.zeros_like(Dt)
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    Mcross[i*nvir+a, j*nvir+b] = -mycross[a, j, b, i]
    # best scalar: Dt ~ kappa * Mcross  (kappa tells sign/factor of the cross fix)
    denom = float(np.vdot(Mcross, Mcross))
    kappa = float(np.vdot(Mcross, Dt)) / denom if denom > 0 else 0.0
    resid = np.linalg.norm(Dt - kappa * Mcross) / np.linalg.norm(Dt)
    print(f"  Dt ≈ kappa * (my cross contribution): kappa={kappa:+.3f}  resid={resid:.3f}")
    print(f"    (kappa>0 => true cross = (1+kappa)x mine; Dt is what must be ADDED to Gt)")
    # also single routes: does Dt overlap with u_amci/u_akei shapes?
    for nm, tns in [("u_amci", t_amci), ("u_akei", t_akei), ("base", base)]:
        Mt = np.zeros_like(Dt)
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        Mt[i*nvir+a, j*nvir+b] = -tns[a, j, b, i]
        dn = float(np.vdot(Mt, Mt)); k = float(np.vdot(Mt, Dt))/dn if dn>0 else 0
        print(f"    overlap Dt vs -{nm}: coef={k:+.3f} resid={np.linalg.norm(Dt-k*Mt)/np.linalg.norm(Dt):.3f}")

    # ===== extract TRUE g tensors from determinant and decompose by route =====
    # g_phhp_true[b,j,i,a] = (Gs_det - Gt_det)/2  (F cancels, clean everywhere)
    # g_phph_true[a,j,b,i] = δFvv - δFoo - Gt_det  (uses verified build_feff F)
    g_phph_true = np.zeros((nvir, nocc, nvir, nocc))
    g_phhp_true = np.zeros((nvir, nocc, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            r = i*nvir+a
            for j in range(nocc):
                for b in range(nvir):
                    c = j*nvir+b
                    Fp = (Fvv[a,b] if i==j else 0.0) - (Foo[i,j] if a==b else 0.0)
                    g_phph_true[a,j,b,i] = Fp - GtD[r,c]
                    g_phhp_true[b,j,i,a] = 0.5*(GsD[r,c]-GtD[r,c])
    # candidate routes for g_phph (all [a,j,b,i])
    cand = {"base":base, "u_amci":t_amci, "u_akei":t_akei, "cross":mycross}
    names=list(cand); A=np.stack([cand[n].ravel() for n in names],1)
    coef,*_=np.linalg.lstsq(A,g_phph_true.ravel(),rcond=None)
    resid=np.linalg.norm(A@coef-g_phph_true.ravel())/np.linalg.norm(g_phph_true.ravel())
    print(f"\n  ===== g_phph_true = Σ c_k route_k   (resid={resid:.3f}) =====")
    for n,c in zip(names,coef): print(f"    {n:8s} c={c:+.3f}")
    print(f"    [c=1 means route is correct as-is; c≠1 => wrong factor; cross c shows sign+mag]")
    # residual tensor after removing all 4 routes — is there a 5th missing structure?
    R = g_phph_true - sum(coef[i]*cand[names[i]] for i in range(len(names)))
    print(f"    ||g_phph_true||={np.linalg.norm(g_phph_true):.4f}  ||residual after fit||={np.linalg.norm(R):.4f}")
    # save for downstream term-hunting
    np.save("/tmp/g_phph_true.npy", g_phph_true)
    np.save("/tmp/g_phhp_true.npy", g_phhp_true)
    np.savez("/tmp/g_routes.npz", base=base, u_amci=t_amci, u_akei=t_akei, cross=mycross,
             g_phph_ana=g_ph_a, g_phhp_ana=g_hp_a)
    print("  saved true/ana g tensors + routes to /tmp.")

    # ===== ROUTE ISOLATION: rebuild G with only S^IP / only S^EA =====
    zeroIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    zeroEA = {e: np.zeros_like(sEA[e]) for e in sEA}
    def gphph_offdiag(Smat):
        Gx = expm(-Smat) @ HbarN @ expm(Smat)
        _, Gtx = project_1h1p(data, dets, index, Gx)
        Gtx = Gtx - E_N * np.eye(nocc * nvir)
        out = np.zeros((nvir, nocc, nvir, nocc))
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        if i != j and a != b:           # off-diag: Gt = -g_phph, no F
                            out[a, j, b, i] = -Gtx[i*nvir+a, j*nvir+b]
        return out
    S_ip = build_S(data, dets, index, sIP, zeroEA)
    S_ea = build_S(data, dets, index, zeroIP, sEA)
    g_ip = gphph_offdiag(S_ip)        # = base + true u_amci  (off-diag)
    g_ea = gphph_offdiag(S_ea)        # = base + true u_akei
    g_full = gphph_offdiag(S)         # = base + uA + uE + uX
    def offnorm(X):
        s = 0.0
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        if i != j and a != b: s += X[a,j,b,i]**2
        return s**0.5
    def offratio(num, den):
        # best scalar num ≈ k*den over off-diag
        n=d_=0.0
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        if i!=j and a!=b:
                            n+=den[a,j,b,i]*num[a,j,b,i]; d_+=den[a,j,b,i]**2
        return n/d_ if d_>0 else 0.0
    uA_true = g_ip - base; uE_true = g_ea - base
    uX_true = g_full - g_ip - g_ea + base
    print(f"\n  ===== ROUTE ISOLATION (off-diagonal, F-free) =====")
    print(f"    u_amci: ||mine||={offnorm(t_amci):.4f} ||true||={offnorm(uA_true):.4f}  "
          f"true≈{offratio(uA_true,t_amci):+.2f}×mine")
    print(f"    u_akei: ||mine||={offnorm(t_akei):.4f} ||true||={offnorm(uE_true):.4f}  "
          f"true≈{offratio(uE_true,t_akei):+.2f}×mine")
    print(f"    cross : ||mine||={offnorm(mycross):.4f} ||true||={offnorm(uX_true):.4f}  "
          f"true≈{offratio(uX_true,mycross):+.2f}×mine")


def main():
    data = get_active_data()
    nact = data["nact"]; nso = 2 * nact
    print(f"nact={nact} nso={nso} nelec={data['nelec']} nocc={data['nocc']} nvir={data['nvir']}")
    print(f"HF={data['Ehf']:.8f}  CCSD={data['Eccsd']:.8f}  FCI={data['Efci']:.8f}")
    dets, index = build_dets(nso, data["nelec"])
    print(f"n_dets(N={data['nelec']} sector) = {len(dets)}")
    H, hso = build_H(data, dets, index)
    Econst = data["Ecore"] + data["Enuc"]
    w = np.linalg.eigvalsh(H)
    print(f"\nVALIDATE bare H: lowest eig + Econst = {w[0] + Econst:.8f}  (FCI {data['Efci']:.8f})  "
          f"Δ={abs(w[0] + Econst - data['Efci'])*1e6:.2f} µHa")
    hf = hf_det(data)
    Ehf_det = H[index[hf], index[hf]] + Econst
    print(f"VALIDATE <HF|H|HF> + Econst = {Ehf_det:.8f}  (HF {data['Ehf']:.8f})  "
          f"Δ={abs(Ehf_det - data['Ehf'])*1e6:.2f} µHa")
    # ---- T operator + Hbar = exp(-T) H exp(T) ----
    t1so, t2so = build_t_so(data)
    T1 = build_excitation_matrix(data, dets, index, t1so, "t1")
    T2 = build_excitation_matrix(data, dets, index, t2so, "t2")
    Tm = T1 + T2
    eT = expm(Tm); emT = expm(-Tm)
    Hbar = emT @ H @ eT
    iHF = index[hf]
    Ecorr_det = Hbar[iHF, iHF] - H[iHF, iHF]
    print(f"\nVALIDATE <HF|Hbar|HF> - <HF|H|HF> = {Ecorr_det:.8f}  (E_corr CCSD "
          f"{data['Eccsd'] - data['Ehf']:.8f})  Δ={abs(Ecorr_det - (data['Eccsd']-data['Ehf']))*1e6:.2f} µHa")
    # CCSD residuals: <S|Hbar|HF> and <D|Hbar|HF> must be ~0
    sd = singles_dets(data, dets, index)
    rS = max(abs(Hbar[index[d], iHF] * sgn) for (d, sgn) in sd.values())
    print(f"VALIDATE max|<singles|Hbar|HF>| = {rS:.2e}  (CCSD singles residual ~0)")
    # doubles residual: sample a few double excitations
    np.save("/tmp/fock_H.npy", H)
    np.save("/tmp/fock_Hbar.npy", Hbar)
    json.dump({"dets": dets, "hf": hf, "Econst": Econst},
              open("/tmp/fock_meta.json", "w"))
    print("saved H, Hbar + meta.")


if __name__ == "__main__":
    main()
