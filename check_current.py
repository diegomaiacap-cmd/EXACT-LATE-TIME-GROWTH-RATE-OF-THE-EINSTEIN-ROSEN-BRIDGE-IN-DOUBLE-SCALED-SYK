"""Proposition 1' in numbers: the current <J> = d<n>/dt converges to the rate as t^-2, while <n>/t converges as t^-1.
Also checks the corrected moment bound of the thermofield double, |<n|e^{-beta H/2}|0>| <= e^{beta B} (beta B)^n / n!.
-> data/current_check.json"""
import os, sys, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from scipy.linalg import eigh_tridiagonal, expm
from module_dssyk import lanczos_b
from module_rate import rate_beta0

out = dict(beta0=[], bound=[])
for lam in (0.5, 2.0):
    nmax = 3000; b = lanczos_b(lam, nmax); e, U = eigh_tridiagonal(np.zeros(nmax), b)
    c = U[0].copy(); exact = rate_beta0(lam)
    for t in (10, 25, 50, 100, 200):
        p = U @ (np.exp(-1j * e * t) * c)
        Jp = np.zeros(nmax, complex); Jp[:-1] += b * p[1:]; Jp[1:] -= b * p[:-1]
        cur = lam * float(np.real(np.vdot(p, 1j * Jp))); ell = lam * float(np.arange(nmax) @ np.abs(p) ** 2)
        out['beta0'].append(dict(lam=lam, t=t, current=cur, ell_over_t=ell / t, exact=exact, dev_current=cur / exact - 1,
                                 dev_ell_over_t=ell / t / exact - 1, edge=float(np.sum(np.abs(p[-50:]) ** 2))))
        print(out['beta0'][-1], flush=True)
lam = 1.0; q = np.exp(-lam); B = 1 / np.sqrt(lam * (1 - q)); nmax = 200
H = np.diag(lanczos_b(lam, nmax), 1); H = H + H.T
for beta in (1.0, 5.0):
    v = expm(-beta * H / 2)[:, 0]
    for n in (0, 1, 5, 20):
        old = (beta * B) ** n / math.factorial(n)
        out['bound'].append(dict(lam=lam, beta=beta, n=n, value=float(abs(v[n])), paper_bound=old, corrected_bound=old * math.exp(beta * B)))
        print(out['bound'][-1], flush=True)
json.dump(out, open('data/current_check.json', 'w'), indent=1)
