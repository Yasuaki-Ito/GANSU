#!/usr/bin/env python3
"""C++ CODE GENERATOR — FULL-CLASS projection routes (franken-G fix, GEMM edition).

Emits include/steom_gphph_projection_full.inc from the verified blueprint
script/gphph_projection_full_terms.json (steom_gphph_diagsemi.py, machine-exact
on all element classes).  Differences vs steom_gphph_cppgen.py (v2, off-diag):
  * terms loaded from the JSON blueprint (no sympy re-derivation),
  * DELTA terms (delta_o[i,j] / delta_v[a,b], Ms only) are emitted: the
    contraction runs over the reduced kept-external set and the unpack loop
    binds the aliased external (j:=i or b:=a),
  * route arrays are exact on ALL classes -> the .cu applies them everywhere
    under GANSU_STEOM_PROJ_FULL (with the EE F-base swap in the G assembly).
The old include/steom_gphph_projection.inc (off-diag mode) is left untouched.

Run:  wsl python3 script/steom_gphph_cppgen_full.py
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN
import steom_gphph_cppgen as CG

EXT = ('i', 'a', 'j', 'b')
CG.ARR_CPP['Xov'] = 'fov'


def parse_json():
    blob = json.load(open("script/gphph_projection_full_terms.json"))

    def space(nm):
        return 'o' if nm in ('i', 'j') or nm in 'klmn' else 'v'

    def parse(lst):
        out = []
        for t in lst:
            ops = tuple((arr, tuple((nm, space(nm)) for nm in toks))
                        for arr, toks in t["ops"])
            out.append((t["coeff"], ops))
        return out
    return parse(blob["ms"]), parse(blob["mc"])


def split_term(term):
    coeff, ops = term
    deltas = [op for op in ops if op[0].startswith('delta')]
    rest = tuple(op for op in ops if not op[0].startswith('delta'))
    assert len(deltas) <= 1, f"multiple deltas in {term}"
    if deltas:
        dk, dtoks = deltas[0]
        p, q = dtoks[0][0], dtoks[1][0]
        rest_lets = {nm for _, toks in rest for nm, _ in toks}
        assert p not in rest_lets and q not in rest_lets, \
            f"delta letters appear in operands: {term}"
    return coeff, rest, deltas


def plan_full(term):
    coeff, rest, deltas = split_term(term)
    rest_lets = {nm for _, toks in rest for nm, _ in toks}
    kept = [e for e in EXT if e in rest_lets]
    plan = CG.plan_term((coeff, rest))
    plan['deltas'] = deltas
    plan['kept'] = kept
    if deltas:
        dk, dtoks = deltas[0]
        p, q = dtoks[0][0], dtoks[1][0]
        plan['alias'] = {q: p}
        plan['letters'].setdefault(p, {'space': 'o' if dk == 'delta_o' else 'v',
                                       'root': ''})
    else:
        plan['alias'] = {}
    return plan


# ---------------------------------------------------------------- numpy executor
def exec_plan_full(plan, arrays, NO, NV):
    coeff, ops, kept = plan['coeff'], plan['ops'], plan['kept']

    def sub(op):
        return "".join(nm for nm, _ in op[1])
    outsub = "".join(kept)
    if plan['pair'] is None:
        s = ",".join(sub(o) for o in ops) + "->" + outsub
        C = coeff * np.einsum(s, *[arrays[o[0]] for o in ops], optimize=True)
    else:
        pi, other, tmp = plan['pair'], plan['other'], plan['tmp']
        s1 = ",".join(sub(ops[k]) for k in pi) + "->" + "".join(tmp)
        t = np.einsum(s1, *[arrays[ops[k][0]] for k in pi], optimize=True)
        s2 = "".join(tmp) + "," + sub(ops[other]) + "->" + outsub
        C = coeff * np.einsum(s2, t, arrays[ops[other][0]], optimize=True)
    if not plan['deltas']:
        return C
    dk, dtoks = plan['deltas'][0]
    p, q = dtoks[0][0], dtoks[1][0]
    eye = np.eye(NO if dk == 'delta_o' else NV)
    s = outsub + "," + p + q + "->iajb"
    return np.einsum(s, C, eye, optimize=True)


# ------------------------------------------------------------------ C++ emission
def emit_term_full(tid, plan, target):
    coeff, ops, letters = plan['coeff'], plan['ops'], plan['letters']
    kept, alias = plan['kept'], plan['alias']
    dcomment = (" * " + "".join(f"delta({p}={q})" for q, p in alias.items())) if alias else ""
    lines = [f"    {{ // term {tid}: {coeff:+.6f} * " +
             " * ".join(f"{a}[{','.join(nm for nm, _ in t)}]" for a, t in ops) +
             dcomment + "\n"]
    if plan['pair'] is None:
        A, B = ops[0], ops[1]
        code, clay, cbuf = CG.emit_pair_gemm(
            tid, 1, CG.access(A[0], A[1], letters), [nm for nm, _ in A[1]],
            CG.access(B[0], B[1], letters), [nm for nm, _ in B[1]], kept, letters, target)
        lines.append(code)
    else:
        pi, other = plan['pair'], plan['other']
        A, B, C = ops[pi[0]], ops[pi[1]], ops[other]
        tmp = plan['tmp']
        code1, lay1, buf1 = CG.emit_pair_gemm(
            tid, 1, CG.access(A[0], A[1], letters), [nm for nm, _ in A[1]],
            CG.access(B[0], B[1], letters), [nm for nm, _ in B[1]], tmp, letters, target)
        lines.append(code1)
        tmp_expr = f"{buf1}[{CG.flat_index(lay1, letters)}]"
        code2, clay, cbuf = CG.emit_pair_gemm(
            tid, 2, tmp_expr, lay1,
            CG.access(C[0], C[1], letters), [nm for nm, _ in C[1]], kept, letters, target)
        lines.append(code2)
    # unpack over externals; aliased letters bound to their representative
    loop_lets = [e for e in EXT if e not in alias]
    lines.append("    #pragma omp parallel for\n")
    for k, nm in enumerate(loop_lets):
        lines.append(CG.loop_open(nm, letters, "    " + "  " * (k + 1)))
    ind2 = "    " + "  " * (len(loop_lets) + 1)

    def qv(nm):
        return f"q_{alias.get(nm, nm)}"
    tgt_idx = f"(((size_t){qv('i')}*NV+{qv('a')})*NO+{qv('j')})*NV+{qv('b')}"
    lines.append(f"{ind2}{target}[{tgt_idx}] += "
                 f"{coeff:+.16e}*{cbuf}[{CG.flat_index(clay, letters)}];\n")
    for k in range(len(loop_lets)):
        lines.append("    " + "  " * (len(loop_lets) - k) + "}\n")
    lines.append("    }\n")
    return "".join(lines)


def emit_naive_full(tid, plan, target):
    coeff, ops, letters = plan['coeff'], plan['ops'], plan['letters']
    alias = plan['alias']
    lines = []
    ind = "        "
    outer = [e for e in EXT if e not in alias]
    summed = [nm for nm in letters if nm not in EXT]
    for k, nm in enumerate(outer):
        lines.append(CG.loop_open(nm, letters, ind + "  " * k))
    ind2 = ind + "  " * len(outer)
    lines.append(f"{ind2}real_t acc=0.0;\n")
    for k, nm in enumerate(summed):
        lines.append(CG.loop_open(nm, letters, ind2 + "  " * k))
    ind3 = ind2 + "  " * len(summed)
    prod = "*".join(CG.access(a, t, letters) for a, t in ops)
    lines.append(f"{ind3}acc += {prod};\n")
    for k in range(len(summed)):
        lines.append(ind2 + "  " * (len(summed) - 1 - k) + "}\n")

    def qv(nm):
        return f"q_{alias.get(nm, nm)}"
    tgt_idx = f"(((size_t){qv('i')}*NV+{qv('a')})*NO+{qv('j')})*NV+{qv('b')}"
    lines.append(f"{ind2}{target}[{tgt_idx}] += {coeff:+.16e}*acc;\n")
    for k in range(len(outer)):
        lines.append(ind + "  " * (len(outer) - 1 - k) + "}\n")
    return "".join(lines)


def main():
    Ms, Mc = parse_json()
    route_ph = GEN.subtract(Mc, Ms)   # Mc - Ms (deltas enter with -Ms coeff)
    route_mc = Mc
    nd = lambda lst: sum(1 for c, ops in lst
                         if any(a.startswith('delta') for a, _ in ops))
    print(f"route_ph terms={len(route_ph)} (delta {nd(route_ph)})  "
          f"route_mc terms={len(route_mc)} (delta {nd(route_mc)})")

    # ---------- verify plans numerically vs GEN.evaluate (H2O FC1) ----------
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
    arrays = {'Fov': bar["Fov"], 'Xov': bar["Fov"], 'Wooov': bar["Wooov"],
              'Wvovv': bar["Wvovv"], 'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    for nm, lst in [("route_ph", route_ph), ("route_mc", route_mc)]:
        plans = [plan_full(t) for t in lst]
        acc = np.zeros((nocc, nvir, nocc, nvir))
        for p in plans:
            acc += exec_plan_full(p, arrays, nocc, nvir)
        ref = GEN.evaluate(lst, arrays, nocc, nvir)
        print(f"[plan verify {nm}] ||plan - direct|| = {np.linalg.norm(acc - ref):.3e}")

    # ---------- emit ----------
    out = []
    out.append("// AUTO-GENERATED by script/steom_gphph_cppgen_full.py — DO NOT EDIT.\n")
    out.append("// FULL-CLASS projection g_phph/g_phhp routes (franken-G fix): route arrays\n")
    out.append("// are exact on ALL element classes (off/semi/diag), including the delta\n")
    out.append("// terms (delta_o/delta_v, Ms only).  Apply with GANSU_STEOM_PROJ_FULL=1\n")
    out.append("// together with the EE F-base swap in the G assembly (Foo_ee[j,i]/Fvv_ee).\n")
    out.append("// Blueprint: script/gphph_projection_full_terms.json (machine-exact vs the\n")
    out.append("// determinant e^S oracle; gates: H2O FC1 + hexatriene pi-CAS eigenvalues\n")
    out.append("// == oracle to <1e-4 eV, steom_gphph_fullgate.py).\n\n")
    out.append(CG.PROJ_GEMM_HELPER)
    out.append("\n")
    for tgt, lst in [("proj_route_ph", route_ph), ("proj_route_mc", route_mc)]:
        out.append(f"    // ======== {tgt}: {len(lst)} terms (full-class) ========\n")
        for tid, t in enumerate(lst):
            p = plan_full(t)
            out.append(emit_term_full(f"{tgt[-2:]}{tid:03d}", p, tgt))
    out.append("    if (std::getenv(\"GANSU_STEOM_PROJ_VALIDATE\")) {\n")
    out.append("      std::vector<real_t> ref_ph(proj_route_ph.size(),0.0), ref_mc(proj_route_mc.size(),0.0);\n")
    for tgt, ref, lst in [("proj_route_ph", "ref_ph", route_ph),
                          ("proj_route_mc", "ref_mc", route_mc)]:
        out.append(f"      {{ // naive reference for {tgt}\n")
        for t in lst:
            p = plan_full(t)
            out.append(emit_naive_full("n", p, ref))
        out.append("      }\n")
    out.append("      real_t dph=0.0,dmc=0.0;\n")
    out.append("      for (size_t x=0;x<proj_route_ph.size();++x) dph=std::max(dph,std::fabs(proj_route_ph[x]-ref_ph[x]));\n")
    out.append("      for (size_t x=0;x<proj_route_mc.size();++x) dmc=std::max(dmc,std::fabs(proj_route_mc[x]-ref_mc[x]));\n")
    out.append("      std::cout << \"[proj-full self-check] GEMM vs naive: route_ph max|d| = \" << std::scientific\n")
    out.append("                << dph << \", route_mc max|d| = \" << dmc << \" (expect <=1e-12)\" << std::defaultfloat << std::endl;\n")
    out.append("    }\n")
    with open("include/steom_gphph_projection_full.inc", "w") as f:
        f.write("".join(out))
    n = sum(1 for _ in open("include/steom_gphph_projection_full.inc"))
    print(f"[emitted] include/steom_gphph_projection_full.inc  ({n} lines)")


if __name__ == "__main__":
    main()
