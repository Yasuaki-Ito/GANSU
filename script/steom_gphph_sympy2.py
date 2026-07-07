#!/usr/bin/env python3
"""Full 2-body route via sympy AntiSymmetricTensor (auto antisymmetry):
Ms_2[i,a,j,b] = <HF| Fd(i)F(a) [S_ip, (1/4)Σ v[pq,rs]{p†q†sr}] Fd(b)F(j) |HF>.
Output groups into Wooov (v[oo,ov]) & Wvovv (v[vo,vv]) with correct coefficients.
Fov 1-body already machine-exact (steom_gphph_sympy_verify).

Run: wsl python3 script/steom_gphph_sympy2.py
"""
from sympy import symbols, Rational, IndexedBase, Dummy
from sympy.physics.secondquant import (F, Fd, wicks, NO, AntiSymmetricTensor,
                                        substitute_dummies, evaluate_deltas)

pretty = dict(above_fermi='abcdef', below_fermi='ijklmn')


def contract(expr):
    e = wicks(expr, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    e = evaluate_deltas(e)
    e = substitute_dummies(e, new_indices=True, pretty_indices=pretty)
    return e


def main():
    i, j = symbols('i j', below_fermi=True)
    a, b = symbols('a b', above_fermi=True)
    s = IndexedBase('s')
    M = symbols('M', below_fermi=True); I = symbols('I', below_fermi=True)
    J = symbols('J', below_fermi=True); B = symbols('B', above_fermi=True)
    S_ip = Rational(1, 2) * s[M, I, J, B] * NO(Fd(M) * F(I) * Fd(B) * F(J))

    p, q, r, t = symbols('p q r t', cls=Dummy)  # general Dummies => wicks sums occ+vir
    v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4) * v * NO(Fd(p) * Fd(q) * F(t) * F(r))

    expr = Fd(i) * F(a) * (S_ip * V2 - V2 * S_ip) * Fd(b) * F(j)
    print("computing full 2-body route (may take a minute)...")
    res = contract(expr)
    print("\nMs_2[i,a,j,b] =")
    print(res)


if __name__ == "__main__":
    main()
