#!/usr/bin/env python3
"""C++ CODE GENERATOR for the projection g_phph/g_phhp off-diagonal routes — GEMM edition.

Emits include/steom_gphph_projection.inc — evaluated inside build_W_eff_and_G
(steom_ccsd_operator.cu).  Every pairwise contraction is emitted as
   host PACK loops  ->  proj_gemm (cuBLAS DgemmStridedBatched; OMP host fallback)
   ->  UNPACK into tmp / route,
replacing the v1 host-loop evaluation (naphthalene assemble 1198 s -> GEMM).
The naive full-nested reference under GANSU_STEOM_PROJ_VALIDATE is kept unchanged
(small systems only).

Pairwise classification per contraction with kept-letter set KP:
  BATCH = A∩B∩KP  (strided-batch dims), M = A∩KP−B, N = B∩KP−A, K = A∩B−KP,
  extraA/extraB = operand-only summed letters (reduced inside the pack loop).
Buffer layouts: pA[[BATCH][M][K]], pB[[BATCH][N][K]], C[[BATCH][M][N]].

Verifies plan execution in numpy vs GEN.evaluate before emitting.

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


def plan_term(term):
    coeff, ops, letters = analyze(term)
    if len(ops) == 2:
        return dict(coeff=coeff, ops=ops, letters=letters, pair=None)
    best = None
    for pi in ((0, 1), (0, 2), (1, 2)):
        other = ({0, 1, 2} - set(pi)).pop()
        lets_pair = {nm for k in pi for nm, _ in ops[k][1]}
        lets_other = {nm for nm, _ in ops[other][1]}
        tmp_lets = sorted(nm for nm in lets_pair if nm in lets_other or nm in EXT)
        cost = len(lets_pair) + len(set(tmp_lets) | lets_other)
        key = (cost, len(tmp_lets))
        if best is None or key < best[0]:
            best = (key, pi, other, tmp_lets)
    _, pi, other, tmp_lets = best
    return dict(coeff=coeff, ops=ops, letters=letters, pair=pi, other=other, tmp=tmp_lets)


# ------------------------------------------------------------------ numpy executor
def exec_plan(plan, arrays, NO, NV):
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


# ------------------------------------------------------------------ C++ helpers
def loop_open(nm, letters, ind):
    L = letters[nm]
    if L['root'] == 'O':
        return f"{ind}for (int r_{nm}=0;r_{nm}<NMo;++r_{nm}){{ const int q_{nm}=active_occ_idx_[r_{nm}];\n"
    if L['root'] == 'V':
        return f"{ind}for (int r_{nm}=0;r_{nm}<NMv;++r_{nm}){{ const int q_{nm}=active_vir_idx_[r_{nm}];\n"
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


def flat_index(lets, letters):
    """row-major flat index over the letter list (r_ for roots, q_ else)."""
    if not lets: return "0"
    idx = ""
    for k, nm in enumerate(lets):
        v = f"r_{nm}" if letters[nm]['root'] else f"q_{nm}"
        idx = v if k == 0 else f"({idx})*{dim_of(nm, letters)}+{v}"
    return idx


def size_expr(lets, letters):
    if not lets: return "(size_t)1"
    return "*".join((f"(size_t){dim_of(nm, letters)}" if k == 0 else dim_of(nm, letters))
                    for k, nm in enumerate(lets))


def classify_pair(letsA, letsB, kept):
    A, B, KP = set(letsA), set(letsB), set(kept)
    BATCH = sorted(A & B & KP)
    M = sorted((A & KP) - B)
    N = sorted((B & KP) - A)
    K = sorted((A & B) - KP)
    extraA = sorted(A - KP - set(K))
    extraB = sorted(B - KP - set(K))
    return BATCH, M, N, K, extraA, extraB


def emit_pack(buf, acc_expr, loop_lets, extra_lets, letters, ind="    "):
    """pack loops: buf[flat(loop_lets)] = sum_{extra} acc_expr."""
    L = []
    if loop_lets:
        L.append(f"{ind}#pragma omp parallel for\n")
    for k, nm in enumerate(loop_lets):
        L.append(loop_open(nm, letters, ind + "  "*(k+1)))
    ind2 = ind + "  "*(len(loop_lets)+1)
    if extra_lets:
        L.append(f"{ind2}real_t acc=0.0;\n")
        for k, nm in enumerate(extra_lets):
            L.append(loop_open(nm, letters, ind2 + "  "*k))
        ind3 = ind2 + "  "*len(extra_lets)
        L.append(f"{ind3}acc += {acc_expr};\n")
        for k in range(len(extra_lets)):
            L.append(ind2 + "  "*(len(extra_lets)-1-k) + "}\n")
        L.append(f"{ind2}{buf}[{flat_index(loop_lets, letters)}] = acc;\n")
    else:
        L.append(f"{ind2}{buf}[{flat_index(loop_lets, letters)}] = {acc_expr};\n")
    for k in range(len(loop_lets)):
        L.append(ind + "  "*(len(loop_lets)-k) + "}\n")
    return "".join(L)


def emit_pair_gemm(tid, stage, opA_expr, letsA, opB_expr, letsB, kept, letters, out):
    """emit pack A/B + proj_gemm for one pairwise contraction; C layout = BATCH+M+N.
    opX_expr: C++ scalar access expression valid inside loops over its letters.
    Returns (code, C_layout_letters, C_bufname)."""
    BATCH, M, N, K, extraA, extraB = classify_pair(letsA, letsB, kept)
    assert sorted(kept) == sorted(BATCH + M + N), f"kept mismatch {kept} vs {BATCH+M+N}"
    pa, pb, pc = f"pA{tid}_{stage}", f"pB{tid}_{stage}", f"tC{tid}_{stage}"
    szB, szM, szN, szK = (size_expr(BATCH, letters), size_expr(M, letters),
                          size_expr(N, letters), size_expr(K, letters))
    L = []
    L.append(f"    std::vector<real_t> {pa}(({szB})*({szM})*({szK}));\n")
    L.append(f"    std::vector<real_t> {pb}(({szB})*({szN})*({szK}));\n")
    L.append(emit_pack(pa, opA_expr, BATCH + M + K, extraA, letters))
    L.append(emit_pack(pb, opB_expr, BATCH + N + K, extraB, letters))
    L.append(f"    std::vector<real_t> {pc};\n")
    L.append(f"    proj_gemm({pa}, {pb}, {szB}, {szM}, {szN}, {szK}, {pc});\n")
    return "".join(L), BATCH + M + N, pc


def emit_term_gemm(tid, plan, target):
    coeff, ops, letters = plan['coeff'], plan['ops'], plan['letters']
    lines = [f"    {{ // term {tid}: {coeff:+.6f} * " +
             " * ".join(f"{a}[{','.join(nm for nm,_ in t)}]" for a, t in ops) + "\n"]
    if plan['pair'] is None:
        A, B = ops[0], ops[1]
        code, clay, cbuf = emit_pair_gemm(
            tid, 1, access(A[0], A[1], letters), [nm for nm, _ in A[1]],
            access(B[0], B[1], letters), [nm for nm, _ in B[1]], list(EXT), letters, target)
        lines.append(code)
    else:
        pi, other = plan['pair'], plan['other']
        A, B, C = ops[pi[0]], ops[pi[1]], ops[other]
        tmp = plan['tmp']
        code1, lay1, buf1 = emit_pair_gemm(
            tid, 1, access(A[0], A[1], letters), [nm for nm, _ in A[1]],
            access(B[0], B[1], letters), [nm for nm, _ in B[1]], tmp, letters, target)
        lines.append(code1)
        tmp_expr = f"{buf1}[{flat_index(lay1, letters)}]"
        code2, clay, cbuf = emit_pair_gemm(
            tid, 2, tmp_expr, lay1,
            access(C[0], C[1], letters), [nm for nm, _ in C[1]], list(EXT), letters, target)
        lines.append(code2)
    # unpack into target over EXT (root-aware), C layout = clay (permutation of EXT)
    lines.append("    #pragma omp parallel for\n")
    for k, nm in enumerate(EXT):
        lines.append(loop_open(nm, letters, "    " + "  "*(k+1)))
    ind2 = "    " + "  "*5
    lines.append(f"{ind2}{target}[(((size_t)q_i*NV+q_a)*NO+q_j)*NV+q_b] += "
                 f"{coeff:+.16e}*{cbuf}[{flat_index(clay, letters)}];\n")
    for k in range(len(EXT)):
        lines.append("    " + "  "*(len(EXT)-k) + "}\n")
    lines.append("    }\n")
    return "".join(lines)


def emit_naive(tid, plan, target):
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
    lines.append(f"{ind2}{target}[(((size_t)q_i*NV+q_a)*NO+q_j)*NV+q_b] += {coeff:+.16e}*acc;\n")
    for k in range(len(outer)):
        lines.append(ind + "  "*(len(outer)-1-k) + "}\n")
    return "".join(lines)


PROJ_GEMM_HELPER = r"""    // generic pairwise contraction: C[[B][M][N]] = sum_K A[[B][M][K]] * B[[B][N][K]]
    // (row-major; cuBLAS strided-batched on GPU, OMP dot-product loops on CPU-only)
    auto proj_gemm = [&](const std::vector<real_t>& hA, const std::vector<real_t>& hB,
                         size_t PB, size_t PM, size_t PN, size_t PK, std::vector<real_t>& hC){
        hC.assign(PB*PM*PN, 0.0);
        if (PB*PM*PN == 0) return;
        if (PK == 0) PK = 1;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            cublasHandle_t _h = gansu::gpu::GPUHandle::cublas();
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA, hA.size()*sizeof(real_t));
            tracked_cudaMalloc(&dB, hB.size()*sizeof(real_t));
            tracked_cudaMalloc(&dC, hC.size()*sizeof(real_t));
            cudaMemcpy(dA, hA.data(), hA.size()*sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB.data(), hB.size()*sizeof(real_t), cudaMemcpyHostToDevice);
            const real_t _one=1.0, _zero=0.0;
            cublasDgemmStridedBatched(_h, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)PN, (int)PM, (int)PK, &_one,
                dB, (int)PK, (long long)(PN*PK),
                dA, (int)PK, (long long)(PM*PK), &_zero,
                dC, (int)PN, (long long)(PM*PN), (int)PB);
            cudaMemcpy(hC.data(), dC, hC.size()*sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dC);
            return;
        }
#endif
        #pragma omp parallel for collapse(2)
        for (long long _t=0; _t<(long long)PB; ++_t)
            for (long long _m=0; _m<(long long)PM; ++_m) {
                const real_t* _a = &hA[((size_t)_t*PM+_m)*PK];
                for (size_t _n=0; _n<PN; ++_n) {
                    const real_t* _b = &hB[((size_t)_t*PN+_n)*PK];
                    real_t _s = 0.0;
                    for (size_t _k=0; _k<PK; ++_k) _s += _a[_k]*_b[_k];
                    hC[((size_t)_t*PM+_m)*PN+_n] = _s;
                }
            }
    };
"""


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
    nbatch = 0
    for nm, lst in [("route", route_nd), ("Mc", mc_nd)]:
        plans = [plan_term(t) for t in lst]
        acc = np.zeros((nocc, nvir, nocc, nvir))
        for p in plans:
            acc += exec_plan(p, arrays, nocc, nvir)
            # count batched contractions (informational)
            if p['pair'] is not None:
                A, B = p['ops'][p['pair'][0]], p['ops'][p['pair'][1]]
                BATCH, *_ = classify_pair([x for x, _ in A[1]], [x for x, _ in B[1]], p['tmp'])
                if BATCH: nbatch += 1
        ref = GEN.evaluate(lst, arrays, nocc, nvir)
        print(f"[plan verify {nm}] ||plan - direct|| = {np.linalg.norm(acc-ref):.3e}")
    print(f"[info] contractions with BATCH dims: {nbatch}")

    # ---------- emit ----------
    out = []
    out.append("// AUTO-GENERATED by script/steom_gphph_cppgen.py (GEMM edition) — DO NOT EDIT.\n")
    out.append("// Projection g_phph/g_phhp off-diagonal routes (STEOM overshoot fix).\n")
    out.append("// Fills proj_route_ph[i,a,j,b] (= Mc-Ms) and proj_route_mc[i,a,j,b] (= Mc).\n")
    out.append("// Every pairwise contraction = host pack -> proj_gemm (cuBLAS strided-batched;\n")
    out.append("// OMP host fallback) -> unpack.  Scope: NO,NV,NMo,NMv, active_occ_idx_,\n")
    out.append("// active_vir_idx_, lambdas fov,wooov,wvovv,eriov,siP,seA; proj_route_ph/_mc.\n")
    out.append("// Derivation/gates: memory project_dlpno_steom_orca_reconsider step1続31-40.\n\n")
    out.append(PROJ_GEMM_HELPER)
    out.append("\n")
    for tgt, lst in [("proj_route_ph", route_nd), ("proj_route_mc", mc_nd)]:
        out.append(f"    // ======== {tgt}: {len(lst)} terms ========\n")
        for tid, t in enumerate(lst):
            p = plan_term(t)
            out.append(emit_term_gemm(f"{tgt[-2:]}{tid:03d}", p, tgt))
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
    out.append("      std::cout << \"[proj self-check] GEMM vs naive: route_ph max|d| = \" << std::scientific\n")
    out.append("                << dph << \", route_mc max|d| = \" << dmc << \" (expect <=1e-12)\" << std::defaultfloat << std::endl;\n")
    out.append("    }\n")
    with open("include/steom_gphph_projection.inc", "w") as f:
        f.write("".join(out))
    n = sum(1 for _ in open("include/steom_gphph_projection.inc"))
    print(f"[emitted] include/steom_gphph_projection.inc  ({n} lines, GEMM edition)")


if __name__ == "__main__":
    main()
