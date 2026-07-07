#!/usr/bin/env python3
"""Enumerate the v-blocks (o/v index patterns) and amplitude combos appearing in
the QUADRATIC g_phph route expression, to determine which dressed Hbar blocks the
C++ needs (is Woooo/Wvvvv required, or only ooov/vovv/ovov/oovv?).

Run:  wsl python3 script/steom_gphph_quad_blocks.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from collections import Counter
from sympy import symbols, Rational, Dummy, IndexedBase, Add, Mul
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor, KroneckerDelta
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE


def below(sym):
    return bool(sym.assumptions0.get('below_fermi'))


def main():
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4)*v*NO(Fd(p)*Fd(q)*F(t)*F(r))
    ss = IndexedBase('s'); sea = IndexedBase('se')
    def Sip(tg):
        M, I, J = symbols(f'M{tg} I{tg} J{tg}', below_fermi=True); B = symbols(f'B{tg}', above_fermi=True)
        return Rational(1, 2)*ss[M, I, J, B]*NO(Fd(M)*F(I)*Fd(B)*F(J))
    def Sea(tg):
        E, A, B = symbols(f'E{tg} A{tg} B{tg}', above_fermi=True); J = symbols(f'J{tg}', below_fermi=True)
        return Rational(1, 2)*sea[E, J, A, B]*NO(Fd(A)*F(E)*Fd(B)*F(J))

    combos = [("ip.ip", Sip('1'), Sip('2')), ("ea.ea", Sea('1'), Sea('2')),
              ("ip.ea", Sip('1'), Sea('2')), ("ea.ip", Sea('1'), Sip('2'))]
    total = Counter()
    for name, SA, SB in combos:
        inner = SB*V2 - V2*SB; nq = Rational(1, 2)*(SA*inner - inner*SA)
        expr = WE.contract(Fd(i)*F(a)*nq*Fd(b)*F(j))
        cnt = Counter(); nterms = 0
        for term in Add.make_args(expr):
            if term == 0: continue
            nterms += 1
            for f in Mul.make_args(term):
                if isinstance(f, AntiSymmetricTensor):
                    sig = "".join('o' if below(s) else 'v' for s in list(f.upper)+list(f.lower))
                    cnt[sig] += 1
        total.update(cnt)
        print(f"{name}: {nterms} terms, v-blocks: {dict(cnt)}")
    print("\nTOTAL v-block usage:", dict(total))


if __name__ == "__main__":
    main()
