#!/usr/bin/env python3
"""
Brute-force the correct .spinad()/tilde pattern of the STEOM g_phhp dressing.

Strategy (rigorous, low-overfit): the single-route dressing of g_phhp is a sum
of 3 S^IP terms (Fov,Wovoo,Wvovv) + 3 S^EA terms (Fov,Wvovv,Wooov) with KNOWN
coefficients; the only freedom is whether each term contracts the raw s or the
spin-adapted s̃ = 2s - s_swap.  We hold the cross (S^IP·S^EA) term fixed at its
current Nooijen form and enumerate all 2^6 tilde on/off choices, scoring each by
the element-wise residual vs the EXACT g_phhp (from singlet/triplet EOM) on the
(i!=j,a!=b) off-diagonal block, summed over several small molecules.

Usage: wsl python3 script/steom_spinad_brute.py
"""
import sys, itertools
import numpy as np
sys.path.insert(0, "script")
from steom_spinad_search import setup, base_phhp, mask_phhp, scatter_IP, scatter_EA, DEFAULT_MOLS

np.set_printoptions(precision=5, suppress=True, linewidth=160)


def cross_phhp(d):
    """Current Nooijen cross (S^IP·S^EA) u_bmje, scattered to [b,k,j,c]."""
    bar = d["bar_h"]; s_IP = d["s_IP"]; s_EA = d["s_EA"]
    nocc = d["nocc"]; nvir = d["nvir"]; occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    Wovvo = bar["Wovvo"]; eri_ovov = bar["eri_ovov"]
    n_act_occ = len(s_IP); n_act_vir = len(s_EA)
    # supporting hp / hhhp intermediates (Eq.38-44)
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
    # cross u_bmje (Eq.62)
    out = np.zeros((nvir, nocc, nocc, nvir))
    for m in range(n_act_occ):
        for e in range(n_act_vir):
            sIP = s_IP[m]; sEA = s_EA[e]
            t  = np.einsum("d,jdb->bj", u_ma[m], sEA)
            t -= np.einsum("k,kjb->bj", u_ie[:, e], sIP)
            t += np.einsum("klj,klb->bj", u_klie[:, :, :, e], sIP)
            t -= np.einsum("ljd,ldb->bj", u_mlid[m], sEA)
            out[:, occ_idx[m], :, vir_idx[e]] += t
    return out


def ip_terms(d):
    """Return list of (name, direct_full, tilde_full) for 3 S^IP terms."""
    bar = d["bar_h"]; s_IP = d["s_IP"]
    nocc = d["nocc"]; nvir = d["nvir"]; occ_idx = d["occ_idx"]
    Fov = bar["Fov"]; Wovoo = bar["Wovoo"]; Wvovv = bar["Wvovv"]
    n = len(s_IP)

    def tIP(fn):  # direct
        return scatter_IP([fn(s_IP[m]) for m in range(n)], occ_idx, nocc, nvir)

    def tIPt(fn):  # tilde over the two occ slots of s
        return scatter_IP([fn(2.0 * s_IP[m] - s_IP[m].transpose(1, 0, 2))
                           for m in range(n)], occ_idx, nocc, nvir)
    res = []
    fns = [("F",  lambda s: -np.einsum("kc,kjb->bjc", Fov, s)),
           ("O",  lambda s:  np.einsum("kclj,klb->bjc", Wovoo, s)),
           ("V",  lambda s: -np.einsum("bkdc,kjd->bjc", Wvovv, s))]
    for nm, fn in fns:
        res.append(("IP_" + nm, tIP(fn), tIPt(fn)))
    return res


def ea_terms(d):
    bar = d["bar_h"]; s_EA = d["s_EA"]
    nocc = d["nocc"]; nvir = d["nvir"]; vir_idx = d["vir_idx"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]
    n = len(s_EA)

    def tEA(fn):
        return scatter_EA([fn(s_EA[e]) for e in range(n)], vir_idx, nocc, nvir)

    def tEAt(fn):  # tilde over the two vir slots of s_EA[e]=[j,d,b]
        return scatter_EA([fn(2.0 * s_EA[e] - s_EA[e].transpose(0, 2, 1))
                           for e in range(n)], vir_idx, nocc, nvir)
    res = []
    fns = [("F",  lambda s:  np.einsum("kd,jdb->bkj", Fov, s)),
           ("V",  lambda s:  np.einsum("bkdc,jcd->bkj", Wvovv, s)),
           ("O",  lambda s: -np.einsum("lkjd,ldb->bkj", Wooov, s))]
    for nm, fn in fns:
        res.append(("EA_" + nm, tEA(fn), tEAt(fn)))
    return res


def main(mols=DEFAULT_MOLS):
    data = []
    for (xyz, basis, ncore) in mols:
        d = setup(xyz, basis, ncore)
        msk = mask_phhp(d["nocc"], d["nvir"])
        if msk.sum() == 0:
            continue
        base = base_phhp(d)
        cross = cross_phhp(d)
        ipt = ip_terms(d); eat = ea_terms(d)
        terms = ipt + eat
        names = [t[0] for t in terms]
        # stack masked vectors
        tgt = (d["exact_phhp"] - base - cross)[msk]
        direct = [t[1][msk] for t in terms]
        tilde = [t[2][msk] for t in terms]
        data.append(dict(xyz=xyz, tgt=tgt, direct=direct, tilde=tilde, names=names))
        print(f"{xyz}: #elems={msk.sum()}  ‖target_dress‖={np.linalg.norm(tgt):.4e}")
    names = data[0]["names"]
    nterm = len(names)

    # brute force 2^nterm tilde flags (0=direct,1=tilde)
    best = []
    for flags in itertools.product([0, 1], repeat=nterm):
        resid2 = 0.0; norm2 = 0.0
        for d in data:
            pred = np.zeros_like(d["tgt"])
            for k, f in enumerate(flags):
                pred += d["tilde"][k] if f else d["direct"][k]
            resid2 += np.sum((pred - d["tgt"]) ** 2)
            norm2 += np.sum(d["tgt"] ** 2)
        best.append((np.sqrt(resid2), flags, np.sqrt(resid2 / norm2)))
    best.sort()
    print(f"\n{'rank':>4} {'resid':>11} {'rel':>8}   flags (1=tilde): " + " ".join(names))
    for r, (resid, flags, rel) in enumerate(best[:8]):
        fl = " ".join(f"{names[k]}={f}" for k, f in enumerate(flags))
        print(f"{r:>4} {resid:>11.4e} {rel:>8.3f}   {fl}")
    # also report all-direct (current code) for reference
    cur = next(b for b in best if b[1] == tuple([0] * nterm))
    print(f"\ncurrent (all-direct): resid={cur[0]:.4e} rel={cur[2]:.3f}")


if __name__ == "__main__":
    main()
