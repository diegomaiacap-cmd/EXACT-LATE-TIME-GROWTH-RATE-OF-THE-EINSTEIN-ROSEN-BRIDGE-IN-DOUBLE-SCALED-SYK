"""Finite temperature.  Writes data/finite_beta.json with
 points : exact energy-representation evolution of e^{-beta H/2}|0> vs Eq. (5)
 curves : Eq. (5), single saddle and the shifted triple-scaled formula over beta J in [0.1, 1e3]
 ts     : convergence of Eq. (5) to the triple-scaled function R(b) at fixed b as lambda -> 0
 cross  : beta J at which the exact rate crosses 2 pi / beta, and the largest saddle ratio
 deepir : Eq. (5) over the fixed-lambda asymptote 4 lambda sqrt(B / (pi beta))
Usage: python scripts/run_finite_beta.py [points|curves|all]"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from scipy.optimize import brentq
from module_dssyk import evolve_spectral
from module_rate import rate, saddle_rate, rate_ts, R_ts, rate_deep_ir, consts
mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
f = 'data/finite_beta.json'
out = json.load(open(f)) if os.path.exists(f) else {}

if mode in ('points', 'all'):
    pts = []
    # (lambda, beta, nmax, theta window lower edge or None, final time, slope window)
    P = ((0.05, 2.0, 1800, np.pi - 1.2, 40, 5), (0.05, 5.0, 1500, np.pi - 1.0, 40, 5), (0.05, 20.0, 1200, np.pi - 0.8, 80, 10),
         (0.01, 50.0, 5000, np.pi - 0.5, 80, 10), (0.5, 2.0, 800, None, 80, 10), (0.5, 10.0, 800, np.pi - 1.5, 80, 10),
         (0.2, 50.0, 1100, np.pi - 0.8, 400, 50), (0.5, 20.0, 1100, np.pi - 1.2, 400, 50), (1.0, 30.0, 900, np.pi - 1.2, 400, 50),
         (2.0, 100.0, 1200, np.pi - 1.0, 1500, 100))
    for lam, beta, nmax, win, T, dw in P:
        t0 = time.time(); tt = np.concatenate([np.linspace(0, T - dw, 40), [T]])
        m, nrm, r5, Lcl = evolve_spectral(lam, beta, tt, nmax, window=(win,) if win else None, npts=40000)
        sl = float(lam * (m[-1] - m[-2]) / dw); sl_sc = float((Lcl[-1] - Lcl[-2]) / dw)
        pts.append(dict(lam=lam, beta=beta, T=T, slope_exact=sl, eq5=rate(lam, beta), eq5_spectral=float(r5),
                        slope_sc_finite_t=sl_sc, saddle=saddle_rate(lam, beta), norm_dev=float(np.abs(nrm - 1).max()),
                        t=tt.tolist(), ell=(lam * m).tolist()))
        print('point', lam, beta, sl, rate(lam, beta), sl * beta / 2 / np.pi, sl_sc, np.abs(nrm - 1).max(), round(time.time() - t0), flush=True)
    out['points'] = pts; json.dump(out, open(f, 'w'))

if mode in ('curves', 'all'):
    bJ = np.geomspace(0.1, 1000, 61)
    out['curves'] = {}
    for lam in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0):
        out['curves'][str(lam)] = dict(beta=bJ.tolist(), eq5=[rate(lam, b) for b in bJ], saddle=[saddle_rate(lam, b) for b in bJ],
                                       ts=[rate_ts(lam, b) for b in bJ])
        print('curve', lam, flush=True)
    out['ts'] = []
    for lam in (0.04, 0.02, 0.01):
        q, B = consts(lam)
        for b in (0.25, 1.0, 4.0):
            beta = b / (lam ** 2 * B); r = rate(lam, beta)
            out['ts'].append(dict(lam=lam, b=b, beta=beta, ratio_R=r / (2 * np.pi / beta * R_ts(b)), ratio_shift=r / rate_ts(lam, beta)))
            print('ts', out['ts'][-1], flush=True)
    out['cross'] = []
    for lam in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0):
        g = lambda lb: rate(lam, np.exp(lb)) * np.exp(lb) / (2 * np.pi) - 1
        bc = float(np.exp(brentq(g, np.log(0.05), np.log(2000))))
        sr = max(saddle_rate(lam, b) * b / (2 * np.pi) for b in np.geomspace(0.01, 1e4, 200))
        out['cross'].append(dict(lam=lam, beta_cross=bc, lam_beta_cross=lam * bc, max_saddle_ratio=sr))
        print('cross', out['cross'][-1], flush=True)
    out['deepir'] = [dict(lam=lam, beta=b, ratio=rate(lam, b) / rate_deep_ir(lam, b)) for lam in (0.5, 1.0, 4.0) for b in (1e2, 1e3, 1e4)]
    print(out['deepir'])
    json.dump(out, open(f, 'w'))
