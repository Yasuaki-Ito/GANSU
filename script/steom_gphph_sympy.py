#!/usr/bin/env python3
"""Symbolic Wick derivation of the g_phph same-spin route using
sympy.physics.secondquant.  Ms[i,a,j,b] = <HF| Fd(i)F(a) [S_ip, O_block] Fd(b)F(j) |HF>
fully contracted (Fermi vacuum).  Validate on Wvovv (known -0.5), then Fov/Wooov.

S_ip = Σ_{m i j b} s[m,i,j,b] NO(Fd(m)F(i)Fd(b)F(j))   (m,i,j occ ; b vir)

Run: wsl python3 script/steom_gphph_sympy.py
"""
from sympy import symbols, Rational, Dummy, IndexedBase, latex
from sympy.physics.secondquant import (F, Fd, wicks, NO, AntiSymmetricTensor,
                                        substitute_dummies, evaluate_deltas, Commutator)

# index conventions: below_fermi = occupied, above_fermi = virtual
pretty = dict(above_fermi='abcdef', below_fermi='ijklmn')


def contract(expr):
    e = wicks(expr, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    e = evaluate_deltas(e)
    e = substitute_dummies(e, new_indices=True, pretty_indices=pretty)
    return e


def main():
    # external (fixed) indices
    i, j, m, k, l = symbols('i j m k l', below_fermi=True)
    a, b, c, d = symbols('a b c d', above_fermi=True)
    s = IndexedBase('s')       # s[m,i,j,b], antisym in (i,j)
    w = IndexedBase('w')

    # ---- S_ip operator (Nooijen normal-ordered form; NO handles the delta-corrections) ----
    M = symbols('M', below_fermi=True); I = symbols('I', below_fermi=True)
    J = symbols('J', below_fermi=True); B = symbols('B', above_fermi=True)
    S_ip = Rational(1, 2) * s[M, I, J, B] * NO(Fd(M) * F(I) * Fd(B) * F(J))

    A = symbols('A', above_fermi=True); K = symbols('K', below_fermi=True)
    L = symbols('L', below_fermi=True); I2 = symbols('I2', below_fermi=True)
    Cv = symbols('C', above_fermi=True); D = symbols('D', above_fermi=True)
    fF = IndexedBase('f')

    bra = Fd(i) * F(a); ket = Fd(b) * F(j)

    def do(O, label):
        expr = bra * (S_ip * O - O * S_ip) * ket
        print(f"\n--- Ms[i,a,j,b]  ({label}) ---")
        print(contract(expr))

    # Fov: f[k,c] NO(Fd(k) F(c))   (ov block)
    do(fF[K, Cv] * NO(Fd(K) * F(Cv)), "Fov")
    # Wooov: w[k,l,i,c] {k† l† c i} = NO(Fd(k)Fd(l)F(c)F(i))   (ooov block)
    do(w[K, L, I2, Cv] * NO(Fd(K) * Fd(L) * F(Cv) * F(I2)), "Wooov")
    # Wvovv: w[a,k,c,d] {a† k† d c} = NO(Fd(a)Fd(k)F(d)F(c))   (vovv block)
    do(w[A, K, Cv, D] * NO(Fd(A) * Fd(K) * F(D) * F(Cv)), "Wvovv")


if __name__ == "__main__":
    main()
