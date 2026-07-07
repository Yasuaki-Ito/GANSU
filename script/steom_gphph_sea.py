#!/usr/bin/env python3
"""g_phph S_ea route via sympy (same validated method as S_ip):
Ms_ea[i,a,j,b] = <HF| Fd(i)F(a) [S_ea, O] Fd(b)F(j) |HF> for O in {Fov(1b), V2(2b)}.
S_ea = 1/2 s_ea[E,J,A,B] NO(Fd(A)F(E)Fd(B)F(J))  (E,A,B vir ; J occ).

Run: wsl python3 script/steom_gphph_sea.py
"""
from sympy import symbols, Rational, Dummy, IndexedBase
from sympy.physics.secondquant import (F, Fd, wicks, NO, AntiSymmetricTensor,
                                        substitute_dummies, evaluate_deltas)

pretty = dict(above_fermi='abcdef', below_fermi='ijklmn')


def contract(expr):
    e = wicks(expr, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    return substitute_dummies(evaluate_deltas(e), new_indices=True, pretty_indices=pretty)


def main():
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    se = IndexedBase('se'); fF = IndexedBase('f')
    E = symbols('E', above_fermi=True); Jo = symbols('Jo', below_fermi=True)
    A = symbols('A', above_fermi=True); B = symbols('B', above_fermi=True)
    S_ea = Rational(1, 2) * se[E, Jo, A, B] * NO(Fd(A) * F(E) * Fd(B) * F(Jo))
    bra = Fd(i) * F(a); ket = Fd(b) * F(j)

    K = symbols('K', below_fermi=True); Cv = symbols('Cv', above_fermi=True)
    # Fov
    O_f = fF[K, Cv] * NO(Fd(K) * F(Cv))
    print("--- S_ea route, Fov ---")
    print(contract(bra * (S_ea * O_f - O_f * S_ea) * ket))
    # V2 (full 2-body AntiSymmetricTensor)
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    O_v = Rational(1, 4) * v * NO(Fd(p) * Fd(q) * F(t) * F(r))
    print("\n--- S_ea route, 2-body V2 ---")
    print(contract(bra * (S_ea * O_v - O_v * S_ea) * ket))


if __name__ == "__main__":
    main()
