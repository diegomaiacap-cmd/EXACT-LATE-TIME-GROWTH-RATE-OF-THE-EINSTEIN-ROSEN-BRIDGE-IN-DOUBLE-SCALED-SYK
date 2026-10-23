"""Origin of the t^-2 residual (Appendix B.5).

Heuristic: the residual of d<n>/dt comes from the slow modes at the band edges.  If the spectral density of the
state vanishes as theta^alpha at the edges (theta = 0 and pi), modes with velocity below ~ t^(-1/2) are not yet
resolved from the wall; their weight is ~ t^(-(alpha+1)/2) and each shifts <n> by ~ t^(1/2).  Hence
<n>(t) = vbar t + c + O(t^(-alpha/2)) and d<n>/dt - vbar = O(t^(-alpha/2 - 1)).
The q-Hermite measure vanishes as theta^2 at both edges (factor |1 - e^{2i theta}|^2), so alpha = 2 and the
residual is t^-2.

Test: change alpha with states psi_p = (v(H)/2B)^(p-1)|0>, whose spectral density is that of |0> times
(sin theta)^(2p-2), so alpha = 2p.  Predicted exponents: -2, -3, -4 for p = 1, 2, 3.  Done for the free chain
(all b_n = B) and for the DSSYK chain at lambda = 1 and lambda = 0.3.  The exact rate is the spectral average
of v on the same truncated chain, so the comparison has no discretization error.
-> data/edge_exponent.json"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from scipy.linalg import eigh_tridiagonal
from module_dssyk import lanczos_b

ts = np.array([25, 50, 100, 200, 400])
out = []
for model, lam in (('free', None), ('dssyk', 1.0), ('dssyk', 0.3)):
    nmax = 6000
    if model == 'free':
        B = 1.0; b = np.full(nmax - 1, B)
    else:
        b = lanczos_b(lam, nmax); B = 1 / np.sqrt(lam * (1 - np.exp(-lam)))
    e, U = eigh_tridiagonal(np.zeros(nmax), b)
    v = np.sqrt(np.clip(4 * B ** 2 - e ** 2, 0, None))
    for p in (1, 2, 3):
        c = U[0] * (v / (2 * B)) ** (p - 1); c /= np.linalg.norm(c)
        vbar = float(np.sum(c ** 2 * v))
        dev = []
        for t in ts:
            a = U @ (np.exp(-1j * e * t) * c)
            Ja = np.zeros(nmax, complex); Ja[:-1] += b * a[1:]; Ja[1:] -= b * a[:-1]
            cur = float(np.real(np.vdot(a, 1j * Ja)))
            edge = float(np.sum(np.abs(a[-200:]) ** 2))
            dev.append(cur / vbar - 1)
        dev = np.array(dev)
        slopes = np.log(np.abs(dev[1:] / dev[:-1])) / np.log(ts[1:] / ts[:-1])
        rec = dict(model=model, lam=lam, p=p, alpha=2 * p, predicted=-(p + 1), t=ts.tolist(), dev=dev.tolist(),
                   local_exponents=slopes.tolist(), edge_weight=edge)
        out.append(rec)
        print(f"{model:5} lam={lam} p={p} (alpha={2 * p}): predicted {-(p + 1)}, local exponents {np.round(slopes, 2).tolist()}, "
              f"dev {['%.1e' % x for x in dev]}, edge {edge:.0e}", flush=True)
json.dump(out, open('data/edge_exponent.json', 'w'), indent=1)
