#!/usr/bin/env python3
"""2-body route per intermediate via DESIGNATED-index AntiSymmetricTensor (no
general-Dummy merge, correct antisymmetry).  Wooov = w[(k,l),(i,c)] (kl,i occ, c vir),
Wvovv = w[(a,k),(c,d)] (a occ? no: a vir,k occ,c,d vir).  Derive & print.

Run: wsl python3 script/steom_gphph_sympy3.py
"""
from sympy import symbols, Rational, Add
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
    from sympy import IndexedBase
    s = IndexedBase('s')
    M = symbols('M', below_fermi=True); I = symbols('I', below_fermi=True)
    J = symbols('J', below_fermi=True); B = symbols('B', above_fermi=True)
    S_ip = Rational(1, 2) * s[M, I, J, B] * NO(Fd(M) * F(I) * Fd(B) * F(J))
    bra = Fd(i) * F(a); ket = Fd(b) * F(j)

    # Wooov intermediate: w[(K,L),(I2,C)]  K,L,I2 occ ; C vir ; operator {K†L† C I2}
    K = symbols('K', below_fermi=True); L = symbols('L', below_fermi=True)
    I2 = symbols('I2', below_fermi=True); C = symbols('C', above_fermi=True)
    w1 = AntiSymmetricTensor('w', (K, L), (I2, C))
    O1 = Rational(1, 4) * w1 * NO(Fd(K) * Fd(L) * F(C) * F(I2))
    print("--- Wooov route ---")
    print(contract(bra * (S_ip * O1 - O1 * S_ip) * ket))

    # Wvovv intermediate: w[(A,K),(C,D)]  A vir,K occ ; C,D vir ; operator {A†K† D C}
    A = symbols('A', above_fermi=True); K2 = symbols('K2', below_fermi=True)
    C2 = symbols('C2', above_fermi=True); D = symbols('D', above_fermi=True)
    w2 = AntiSymmetricTensor('w', (A, K2), (C2, D))
    O2 = Rational(1, 4) * w2 * NO(Fd(A) * Fd(K2) * F(D) * F(C2))
    print("\n--- Wvovv route ---")
    print(contract(bra * (S_ip * O2 - O2 * S_ip) * ket))


if __name__ == "__main__":
    main()
