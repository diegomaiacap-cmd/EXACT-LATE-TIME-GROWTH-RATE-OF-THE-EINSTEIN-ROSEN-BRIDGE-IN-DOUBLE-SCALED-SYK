"""Convergence of the Bessel series, Eq. (9), with the number of terms and with numerical precision
(referee point 2.1).  -> data/bessel_terms.json

1. Truncation: the theta sum is cut at |m| <= M and the Bessel sum at |j| <= Jmax.  For each (lambda, beta J)
   the error relative to the extended-precision value is recorded as M and Jmax grow.
2. Cancellation: ratio between the sum of absolute values of the terms and the result, numerator and denominator.
3. Precision: the extended-precision evaluation is repeated with 30 and 60 extra digits.
4. Independent reference: direct quadrature of Eq. (5)."""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from scipy.special import ive
from module_rate import rate
from module_bessel import sin_avg_bessel
from module_bessel_hp import sin_avg_bessel_hp


def sin_avg_trunc(lam, beta, M, Jmax):
    q = np.exp(-lam); B = 1 / np.sqrt(lam * (1 - q)); x = 2 * beta * B
    m = np.arange(-M, M + 1); c = (-1.0) ** m * np.exp(-lam * m * (m + 1) / 2)
    j = np.arange(-Jmax, Jmax + 1); I = ive(2 * np.abs(j), x)
    D = (c[None, :] / (1 - 4.0 * (m[None, :] + j[:, None]) ** 2)).sum(1)
    num_terms = (2 / np.pi) * I * D; den_terms = c * ive(2 * np.abs(m), x)
    val = num_terms.sum() / den_terms.sum()
    canc = max(np.abs(num_terms).sum() / abs(num_terms.sum()), np.abs(den_terms).sum() / abs(den_terms.sum()))
    return float(val), float(canc), 2 * lam * B


out = dict(truncation=[], precision=[])
for lam in (0.5, 1.0, 2.0, 4.0, 8.0):
    for bJ in (2.0, 50.0, 200.0):
        s_hp, v = sin_avg_bessel_hp(lam, bJ, extra=40)
        rq = rate(lam, bJ)
        x = 2 * bJ * v / (2 * lam)
        rows = []
        for M in (1, 2, 3, 5, 8, 12, 20):
            Jdef = int(np.sqrt(25 * max(x, 1.0))) + 10
            s, canc, _ = sin_avg_trunc(lam, bJ, M, Jdef)
            rows.append(dict(M=M, Jmax=Jdef, rel=abs(s / s_hp - 1), cancellation=canc))
        jrows = []
        for Jmax in (5, 10, 20, 40, 80, 160):
            s, canc, _ = sin_avg_trunc(lam, bJ, 20, Jmax)
            jrows.append(dict(M=20, Jmax=Jmax, rel=abs(s / s_hp - 1)))
        s_def = sin_avg_bessel(lam, bJ)
        out['truncation'].append(dict(lam=lam, bJ=bJ, x=x, hp=s_hp * v, quad=rq, rel_quad_hp=abs(rq / (s_hp * v) - 1),
                                      rel_default_double=abs(s_def / s_hp - 1), M_scan=rows, J_scan=jrows))
        print(f"lam={lam} bJ={bJ}: quad vs hp {abs(rq / (s_hp * v) - 1):.1e}, double default vs hp {abs(s_def / s_hp - 1):.1e}, "
              f"M scan {[f'{r['rel']:.0e}' for r in rows]}, J scan {[f'{r['rel']:.0e}' for r in jrows]}, cancellation {rows[-1]['cancellation']:.1e}", flush=True)
for lam, bJ in ((0.01, 2.0), (0.01, 50.0), (0.05, 10.0), (0.05, 200.0), (0.2, 50.0), (0.2, 1000.0)):
    t0 = time.time()
    a, v = sin_avg_bessel_hp(lam, bJ, extra=30); b, _ = sin_avg_bessel_hp(lam, bJ, extra=60); rq = rate(lam, bJ)
    s_d, canc, _ = sin_avg_trunc(lam, bJ, int(np.sqrt(2 * 45 / lam)) + 3, int(np.sqrt(25 * max(2 * bJ * v / (2 * lam), 1.0))) + 10)
    out['precision'].append(dict(lam=lam, bJ=bJ, hp30=a * v, hp60=b * v, quad=rq, rel_30_60=abs(a / b - 1), rel_quad=abs(rq / (b * v) - 1),
                                 rel_double=abs(s_d / b - 1), cancellation=canc))
    print(f"lam={lam} bJ={bJ}: hp30 vs hp60 {abs(a / b - 1):.1e}, quad vs hp60 {abs(rq / (b * v) - 1):.1e}, double vs hp60 {abs(s_d / b - 1):.1e}, "
          f"cancellation {canc:.1e}  {time.time() - t0:.0f}s", flush=True)
json.dump(out, open('data/bessel_terms.json', 'w'), indent=1)
