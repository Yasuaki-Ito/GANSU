#!/usr/bin/env python3
"""C++ CODE GENERATOR for the projection g_phph/g_phhp off-diagonal routes.

Emits include/steom_gphph_projection.inc — evaluated inside build_W_eff_and_G
(steom_ccsd_operator.cu) where the lambdas fov/wooov/wvovv/eriov/siP/seA and
NO/NV/NMo/NMv/active_occ_idx_/active_vir_idx_ are in scope.  The .inc fills
  proj_route_ph[i,a,j,b]  (= Mc - Ms, 96 non-delta terms)
  proj_route_mc[i,a,j,b]  (= Mc,      84 terms; g_phhp route)
Each term: pairwise-factorized (W x amp -> tmp -> x amp).  Root letters (axis0 of
siP/seA) loop over active roots with full-index mapping.  A naive full-nested
reference for a sampled element set is emitted under GANSU_STEOM_PROJ_VALIDATE.

Also verifies the PLAN execution in numpy against GEN.evaluate (machine-exact)
before emitting, so the factorization/bookkeeping is proven on the spot.

Run:  wsl python3 script/steom_gphph_cppgen.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN

EXT = ('i', 'a', 'j', 'b')
ARR_CPP = {'Fov': 'fov', 'Wooov': 'wooov', 'Wvovv': 'wvovv', 'eri_ovov': 'eriov',
           'rx': 'siP', 'ry': 'seA'}


def analyze(term):
    """-> (coeff, ops, letters) ; letters[name] = dict(space, root:'','O','V')."""
    coeff, ops = term
    letters = {}
    for arr, toks in ops:
        for pos, (nm, sp) in enumerate(toks):
            L = letters.setdefault(nm, {'space': sp, 'root': ''})
            if arr == 'rx' and pos == 0: L['root'] = 'O'
            if arr == 'ry' and pos == 0: L['root'] = 'V'
    return coeff, ops, letters


def dim_of(nm, letters):
    L = letters[nm]
    if L['root'] == 'O': return 'NMo'
    if L['root'] == 'V': return 'NMv'
    return 'NO' if L['space'] == 'o' else 'NV'


def ndim_of(nm, letters, NO, NV, NMo, NMv):
    L = letters[nm]
    if L['root'] == 'O': return NMo
    if L['root'] == 'V': return NMv
    return NO if L['space'] == 'o' else NV


def plan_term(term):
    """3-op: choose the pair (integral, one amp) minimizing cost; return plan dict."""
    coeff, ops, letters = analyze(term)
    if len(ops) == 2:
        return dict(coeff=coeff, ops=ops, letters=letters, pair=None)
    # candidate pairings: (0,1),(0,2),(1,2)
    best = None
    for pi in ((0, 1), (0, 2), (1, 2)):
        other = ({0, 1, 2} - set(pi)).pop()
        lets_pair = {nm for k in pi for nm, _ in ops[k][1]}
        lets_other = {nm for nm, _ in ops[other][1]}
        tmp_lets = sorted(nm for nm in lets_pair if nm in lets_other or nm in EXT)
        cost = len(lets_pair) + len(set(tmp_lets) | lets_other)   # crude: loop depth
        key = (cost, len(tmp_lets))
        if best is None or key < best[0]:
            best = (key, pi, other, tmp_lets)
    _, pi, other, tmp_lets = best
    return dict(coeff=coeff, ops=ops, letters=letters, pair=pi, other=other, tmp=tmp_lets)


# ------------------------------------------------------------------ numpy executor
def exec_plan(plan, arrays, NO, NV):
    """Execute the plan with numpy loops semantics via einsum (roots = full: the
    Python validation uses all-active).  Verifies factorization bookkeeping."""
    coeff, ops, letters = plan['coeff'], plan['ops'], plan['letters']
    def sub(op): return "".join(nm for nm, _ in op[1])
    if plan['pair'] is None:
        s = ",".join(sub(o) for o in ops) + "->iajb"
        return coeff*np.einsum(s, *[arrays[o[0]] for o in ops], optimize=True)
    pi, other, tmp = plan['pair'], plan['other'], plan['tmp']
    s1 = ",".join(sub(ops[k]) for k in pi) + "->" + "".join(tmp)
    t = np.einsum(s1, *[arrays[ops[k][0]] for k in pi], optimize=True)
    s2 = "".join(tmp) + "," + sub(ops[other]) + "->iajb"
    return coeff*np.einsum(s2, t, arrays[ops[other][0]], optimize=True)


# ------------------------------------------------------------------ C++ emission
def loop_open(nm, letters, ind):
    L = letters[nm]
    if L['root'] == 'O':
        return (f"{ind}for (int r_{nm}=0;r_{nm}<NMo;++r_{nm}){{ const int q_{nm}=active_occ_idx_[r_{nm}];\n")
    if L['root'] == 'V':
        return (f"{ind}for (int r_{nm}=0;r_{nm}<NMv;++r_{nm}){{ const int q_{nm}=active_vir_idx_[r_{nm}];\n")
    d = 'NO' if L['space'] == 'o' else 'NV'
    return f"{ind}for (int q_{nm}=0;q_{nm}<{d};++q_{nm}){{\n"


def access(arr, toks, letters):
    args = []
    for pos, (nm, sp) in enumerate(toks):
        if arr in ('rx', 'ry') and pos == 0:
            args.append(f"r_{nm}")
        else:
            args.append(f"q_{nm}")
    return f"{ARR_CPP[arr]}({','.join(args)})"


def tmp_index(tmp, letters):
    """flat row-major index expression + total-size expression for tmp letters."""
    if not tmp: return "0", "1"
    idx = ""
    for k, nm in enumerate(tmp):
        v = f"r_{nm}" if letters[nm]['root'] else f"q_{nm}"
        idx = v if k == 0 else f"({idx})*{dim_of(nm, letters)}+{v}"
    sz = "*".join(f"(size_t){dim_of(nm, letters)}" if k == 0 else dim_of(nm, letters)
                  for k, nm in enumerate(tmp))
    return idx, sz


def out_index():
    return "(((size_t)q_i*NV+q_a)*NO+q_j)*NV+q_b"


def emit_term(tid, plan, target):
    coeff, ops, letters = plan['coeff'], plan['ops'], plan['letters']
    lines = [f"    {{ // term {tid}: {coeff:+.6f} * " +
             " * ".join(f"{a}[{','.join(nm for nm,_ in t)}]" for a, t in ops) + "\n"]
    if plan['pair'] is None:
        outer = [nm for nm in EXT]
        summed = [nm for nm in letters if nm not in EXT]
        ind = "    "
        lines.append(f"{ind}#pragma omp parallel for\n")
        for k, nm in enumerate(outer):
            lines.append(loop_open(nm, letters, ind + "  "*(k+1)))
        ind2 = "    " + "  "*(len(outer)+1)
        lines.append(f"{ind2}real_t acc=0.0;\n")
        for k, nm in enumerate(summed):
            lines.append(loop_open(nm, letters, ind2 + "  "*k))
        ind3 = ind2 + "  "*len(summed)
        prod = "*".join(access(a, t, letters) for a, t in ops)
        lines.append(f"{ind3}acc += {prod};\n")
        for k in range(len(summed)):
            lines.append(ind2 + "  "*(len(summed)-1-k) + "}\n")
        lines.append(f"{ind2}{target}[{out_index()}] += {coeff:+.16e}*acc;\n")
        for k in range(len(outer)):
            lines.append("    " + "  "*(len(outer)-k) + "}\n")
        lines.append("    }\n")
        return "".join(lines)
    # 3-op factorized
    pi, other, tmp = plan['pair'], plan['other'], plan['tmp']
    pair_lets = []
    for k in pi:
        for nm, _ in ops[k][1]:
            if nm not in pair_lets: pair_lets.append(nm)
    sum1 = [nm for nm in pair_lets if nm not in tmp]
    tidx, tsz = tmp_index(tmp, letters)
    ind = "    "
    lines.append(f"{ind}std::vector<real_t> t{tid}({tsz}, 0.0);\n")
    lines.append(f"{ind}#pragma omp parallel for\n")
    for k, nm in enumerate(tmp):
        lines.append(loop_open(nm, letters, ind + "  "*(k+1)))
    ind2 = ind + "  "*(len(tmp)+1)
    lines.append(f"{ind2}real_t acc=0.0;\n")
    for k, nm in enumerate(sum1):
        lines.append(loop_open(nm, letters, ind2 + "  "*k))
    ind3 = ind2 + "  "*len(sum1)
    prod = "*".join(access(ops[k][0], ops[k][1], letters) for k in pi)
    lines.append(f"{ind3}acc += {prod};\n")
    for k in range(len(sum1)):
        lines.append(ind2 + "  "*(len(sum1)-1-k) + "}\n")
    lines.append(f"{ind2}t{tid}[{tidx}] = acc;\n")
    for k in range(len(tmp)):
        lines.append(ind + "  "*(len(tmp)-k) + "}\n")
    # step2
    step2_sum = [nm for nm in letters if nm not in EXT and (nm in tmp or nm in
                 [x for x, _ in ops[other][1]])]
    step2_sum = [nm for nm in step2_sum if nm not in EXT]
    lines.append(f"{ind}#pragma omp parallel for\n")
    for k, nm in enumerate(EXT):
        lines.append(loop_open(nm, letters, ind + "  "*(k+1)))
    ind2 = ind + "  "*5
    lines.append(f"{ind2}real_t acc=0.0;\n")
    for k, nm in enumerate(step2_sum):
        lines.append(loop_open(nm, letters, ind2 + "  "*k))
    ind3 = ind2 + "  "*len(step2_sum)
    lines.append(f"{ind3}acc += t{tid}[{tidx}]*{access(ops[other][0], ops[other][1], letters)};\n")
    for k in range(len(step2_sum)):
        lines.append(ind2 + "  "*(len(step2_sum)-1-k) + "}\n")
    lines.append(f"{ind2}{target}[{out_index()}] += {coeff:+.16e}*acc;\n")
    for k in range(len(EXT)):
        lines.append(ind + "  "*(len(EXT)-k) + "}\n")
    lines.append("    }\n")
    return "".join(lines)


def emit_naive(tid, plan, target):
    """naive full-nested reference (validation only)."""
    coeff, ops, letters = plan['coeff'], plan['ops'], plan['letters']
    lines = []
    ind = "        "
    outer = list(EXT); summed = [nm for nm in letters if nm not in EXT]
    for k, nm in enumerate(outer):
        lines.append(loop_open(nm, letters, ind + "  "*k))
    ind2 = ind + "  "*len(outer)
    lines.append(f"{ind2}real_t acc=0.0;\n")
    for k, nm in enumerate(summed):
        lines.append(loop_open(nm, letters, ind2 + "  "*k))
    ind3 = ind2 + "  "*len(summed)
    prod = "*".join(access(a, t, letters) for a, t in ops)
    lines.append(f"{ind3}acc += {prod};\n")
    for k in range(len(summed)):
        lines.append(ind2 + "  "*(len(summed)-1-k) + "}\n")
    lines.append(f"{ind2}{target}[{out_index()}] += {coeff:+.16e}*acc;\n")
    for k in range(len(outer)):
        lines.append(ind + "  "*(len(outer)-1-k) + "}\n")
    return "".join(lines)


def main():
    lin = GEN.linear_struct()
    print("deriving quadratic (slow)...")
    quad = GEN.quad_struct()
    allt = lin + quad
    Ms = GEN.expand(allt, 0); Mc = GEN.expand(allt, 1)
    route = GEN.subtract(Mc, Ms)
    nodelta = lambda lst: [(c, ops) for c, ops in lst
                           if not any(a.startswith('delta') for a, _ in ops)]
    route_nd = nodelta(route); mc_nd = nodelta(Mc)
    print(f"route(no-delta)={len(route_nd)}  Mc(no-delta)={len(mc_nd)}")

    # ---------- verify plans numerically vs GEN.evaluate ----------
    import steom_cfour_weff as CW
    import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA
    from steom_fockspace_ref import get_active_data, build_sector, solve_ip, solve_ea, hf_det
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)
    se_det = EA.extract_spatial_amp(solve_ea(data), data)
    rx = np.stack([sp_det[m] for m in range(nocc)], 0); ry = se_det
    d = CW.load("xyz/H2O.xyz", "sto-3g", 1); bar = d["bar"]
    arrays = {'Fov': bar["Fov"], 'Wooov': bar["Wooov"], 'Wvovv': bar["Wvovv"],
              'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    for nm, lst in [("route", route_nd), ("Mc", mc_nd)]:
        plans = [plan_term(t) for t in lst]
        acc = np.zeros((nocc, nvir, nocc, nvir))
        for p in plans: acc += exec_plan(p, arrays, nocc, nvir)
        ref = GEN.evaluate(lst, arrays, nocc, nvir)
        print(f"[plan verify {nm}] ||plan - direct|| = {np.linalg.norm(acc-ref):.3e}")

    # ---------- emit ----------
    out = []
    out.append("// AUTO-GENERATED by script/steom_gphph_cppgen.py — DO NOT EDIT.\n")
    out.append("// Projection g_phph/g_phhp off-diagonal routes (STEOM overshoot fix).\n")
    out.append("// Fills proj_route_ph[i,a,j,b] (= Mc-Ms) and proj_route_mc[i,a,j,b] (= Mc).\n")
    out.append("// Scope requirements: NO,NV,NMo,NMv, active_occ_idx_, active_vir_idx_,\n")
    out.append("//   lambdas fov,wooov,wvovv,eriov,siP,seA; vectors proj_route_ph, proj_route_mc.\n")
    out.append("// Derivation/validation: memory project_dlpno_steom_orca_reconsider step1 1-37;\n")
    out.append("//   gates: H2O FC1 + hexatriene pi-CAS eigenvalues = det oracle (1e-8 eV).\n\n")
    for tgt, lst in [("proj_route_ph", route_nd), ("proj_route_mc", mc_nd)]:
        out.append(f"    // ======== {tgt}: {len(lst)} terms ========\n")
        for tid, t in enumerate(lst):
            p = plan_term(t)
            out.append(emit_term(f"{tgt[-2:]}{tid:03d}", p, tgt))
    # naive validation block
    out.append("    if (std::getenv(\"GANSU_STEOM_PROJ_VALIDATE\")) {\n")
    out.append("      std::vector<real_t> ref_ph(proj_route_ph.size(),0.0), ref_mc(proj_route_mc.size(),0.0);\n")
    for tgt, ref, lst in [("proj_route_ph", "ref_ph", route_nd), ("proj_route_mc", "ref_mc", mc_nd)]:
        out.append(f"      {{ // naive reference for {tgt}\n")
        for t in lst:
            p = plan_term(t)
            out.append(emit_naive("n", p, ref))
        out.append("      }\n")
    out.append("      real_t dph=0.0,dmc=0.0;\n")
    out.append("      for (size_t x=0;x<proj_route_ph.size();++x) dph=std::max(dph,std::fabs(proj_route_ph[x]-ref_ph[x]));\n")
    out.append("      for (size_t x=0;x<proj_route_mc.size();++x) dmc=std::max(dmc,std::fabs(proj_route_mc[x]-ref_mc[x]));\n")
    out.append("      std::cout << \"[proj self-check] factorized vs naive: route_ph max|d| = \" << std::scientific\n")
    out.append("                << dph << \", route_mc max|d| = \" << dmc << \" (expect <=1e-12)\" << std::defaultfloat << std::endl;\n")
    out.append("    }\n")
    with open("include/steom_gphph_projection.inc", "w") as f:
        f.write("".join(out))
    n = sum(1 for _ in open("include/steom_gphph_projection.inc"))
    print(f"[emitted] include/steom_gphph_projection.inc  ({n} lines)")


if __name__ == "__main__":
    main()
