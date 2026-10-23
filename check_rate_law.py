"""Checks of the t^-2 law and of its coefficient (Theorem 2 of the addendum).  -> data/rate_law.json
(1) identity d^2 ell/dt^2 = 2 <q^n>, by a centred finite difference of lambda <J>;
(2) residual of d ell/dt for psi = F(H)|0>, against
    v_inf - d ell/dt = 2 lambda/(pi (alpha+2)) Gamma((alpha+4)/4)^2 (kappa_0 + kappa_pi) B^(-alpha/2) t^(-alpha/2-1),
    where d mu_psi/d theta ~ kappa theta^alpha at each edge (alpha = 2: lambda (kappa_0+kappa_pi) / (8 B t^2))."""
import sys, json
from math import gamma, pi
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / 'src'))
from module_rate import consts, rho_fourier


def chain(lam, nmax):
    q, B = consts(lam); b = B * np.sqrt(1 - q ** np.arange(1, nmax))
    w, U = np.linalg.eigh(np.diag(b, 1) + np.diag(b, -1)); return q, B, b, w, U


def current(p, b):  # <J>, J = i[H, n]
    return -2 * np.sum(b * np.imag(np.conj(p[:-1]) * p[1:]))


def case(lam, label, F, alpha, ts, nmax=3000):
    q, B, b, w, U = chain(lam, nmax); n = np.arange(nmax)
    c = U[0] * F(w, B); c /= np.linalg.norm(c)
    th = np.linspace(0, np.pi, 200001)            # even, periodic, smooth integrand: trapezoid is spectrally accurate
    dens = F(2 * B * np.cos(th), B) ** 2 * rho_fourier(th, lam); N = np.trapezoid(dens, th)
    vinf = 2 * lam * B * np.trapezoid(dens * np.sin(th), th) / N
    def kappa(edge, h):                           # d mu/d theta / theta^alpha at distance h from the edge
        x = np.array([h, 2 * h]); tt = x if edge == 0 else np.pi - x
        g = F(2 * B * np.cos(tt), B) ** 2 * rho_fourier(tt, lam) / N / x ** alpha
        return float((4 * g[0] - g[1]) / 3)       # Richardson, error O(h^4)
    kap = [kappa(0, 0.01), kappa(1, 0.01)]
    C = 2 * lam / (pi * (alpha + 2)) * gamma((alpha + 4) / 4) ** 2 * sum(kap) * B ** (-alpha / 2)
    rows = []
    for t in ts:
        p = U @ (np.exp(-1j * w * t) * c); h = 1e-3
        Jm, Jp = (current(U @ (np.exp(-1j * w * (t + s)) * c), b) for s in (-h, h))
        d2, qn = lam * (Jp - Jm) / (2 * h), float(np.sum(q ** n * abs(p) ** 2))
        res = vinf - lam * current(p, b)
        rows.append(dict(t=t, residual=res, ratio=res * t ** (alpha / 2 + 1) / C, d2l_fd=d2, two_qn=2 * qn,
                         edge=float(abs(p[-50:]).max())))
        print(f"{label} t={t:5.0f} residual={res:.3e} ratio={rows[-1]['ratio']:.5f} d2l={d2:.5e} 2<q^n>={2*qn:.5e}", flush=True)
    return dict(lam=lam, state=label, alpha=alpha, v_inf=vinf, kappa=kap, C=C, rows=rows)


tfd = lambda beta: (lambda E, B: np.exp(-beta * E / 2))
cases = [(1.0, 'TFD beta=0', tfd(0.0), 2, [25, 50, 100, 200, 400]),
         (0.5, 'TFD beta=2', tfd(2.0), 2, [50, 100, 200, 400]),
         (2.0, 'TFD beta=10', tfd(10.0), 2, [50, 100, 200, 400, 800]),
         (1.0, 'F=sqrt(4B^2-E^2)', lambda E, B: np.sqrt(np.maximum(4 * B * B - E * E, 0)), 4, [50, 100, 200, 400]),
         (1.0, 'F=4B^2-E^2', lambda E, B: 4 * B * B - E * E, 6, [50, 100, 200])]
res = [case(*c) for c in cases]


def C_tfd(lam, beta):
    """lambda (kappa_0 + kappa_pi) / (8 B) for the thermofield double, relative to the rate of Eq. (5)."""
    from module_dssyk import rho_theta
    from module_rate import rate
    q, B = consts(lam); qp = np.prod(1 - q ** np.arange(1, int(60 / lam) + 50))
    phi = np.concatenate([np.geomspace(1e-9, 1e-2, 20000), np.linspace(1e-2, np.pi, 200000)[1:]]); th = np.pi - phi
    r = rho_theta(th, q); E = 2 * B * np.cos(th)
    Zs = np.trapezoid(r * np.exp(-beta * (E + 2 * B)), phi) / np.trapezoid(r, phi)        # Z(beta) e^{-2 beta B}
    C = lam * qp ** 3 * 0.5 * (1 + np.exp(-4 * beta * B)) / (2 * np.pi * B * Zs)          # = lam qp^3 cosh(2 beta B) / (2 pi B Z)
    return C / rate(lam, beta), B


# comparison with the stored evolutions of Tables B1 and B2 (relative deviations of the current from Eq. (5))
cc = json.load(open(ROOT / 'data' / 'current_check.json'))['beta0']
b1 = []
for r in cc:
    c, _ = C_tfd(r['lam'], 0.0); b1.append(dict(lam=r['lam'], t=r['t'], observed=r['dev_current'], predicted=-c / r['t'] ** 2))
b2 = []
for r in json.load(open(ROOT / 'data' / 'window_study.json')):
    c, B = C_tfd(r['lam'], r['beta'])
    b2.append(dict(lam=r['lam'], beta=r['beta'], T=r['T'], observed_T=r['dev_current_T'], predicted_T=-c / r['T'] ** 2,
                   observed_T2=r['dev_current_T2'], predicted_T2=-4 * c / r['T'] ** 2))
    print(f"B2 lam={r['lam']} bJ={r['beta']} T={r['T']}: obs {r['dev_current_T']:.2e} pred {-c / r['T'] ** 2:.2e} | T/2 obs {r['dev_current_T2']:.2e} pred {-4 * c / r['T'] ** 2:.2e}")
json.dump(dict(cases=res, table_B1=b1, table_B2=b2), open(ROOT / 'data' / 'rate_law.json', 'w'), indent=1)
