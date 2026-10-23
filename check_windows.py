"""Convergence study for the ten exact evolutions of Table 3 (referee point 2.2).

For each point (lambda, beta) of Table 3 it separates the four approximations behind the reported slope:
  window    : slope over dw (as in Table 3), dw/2 and 2 dw, and the instantaneous current <J> (no window);
  final time: the same quantities at T/2 and T (a t^-2 residual shows up as a factor ~4);
  truncation: nmax multiplied by 1.25;
  precision : quadrature points multiplied by 1.5, and the theta window widened by 0.3.
All values are compared with Eq. (5) (direct quadrature).  -> data/window_study.json
Usage: python scripts/check_windows.py   (about 20-30 min)"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from module_current import evolve_current
from module_rate import rate

P = ((0.05, 2.0, 1800, np.pi - 1.2, 40, 5), (0.05, 5.0, 1500, np.pi - 1.0, 40, 5), (0.05, 20.0, 1200, np.pi - 0.8, 80, 10),
     (0.01, 50.0, 5000, np.pi - 0.5, 80, 10), (0.5, 2.0, 800, None, 80, 10), (0.5, 10.0, 800, np.pi - 1.5, 80, 10),
     (0.2, 50.0, 1100, np.pi - 0.8, 400, 50), (0.5, 20.0, 1100, np.pi - 1.2, 400, 50), (1.0, 30.0, 900, np.pi - 1.2, 400, 50),
     (2.0, 100.0, 1200, np.pi - 1.0, 1500, 100))
out = []
for lam, beta, nmax, win, T, dw in P:
    t0 = time.time(); r5 = rate(lam, beta)
    ts = [T / 2 - dw, T / 2, T - 2 * dw, T - dw, T - dw / 2, T]
    b = evolve_current(lam, beta, ts, nmax, win, 40000)
    m = b['mean']; c = lam * b['current']
    rec = dict(lam=lam, beta=beta, T=T, dw=dw, eq5=r5,
               slope_dw=lam * (m[5] - m[3]) / dw, slope_half=lam * (m[5] - m[4]) / (dw / 2), slope_2dw=lam * (m[5] - m[2]) / (2 * dw),
               slope_dw_T2=lam * (m[1] - m[0]) / dw, current_T=c[5], current_T2=c[1],
               norm_dev=float(np.abs(b['norm'] - 1).max()), edge=b['edge'])
    var = {}
    for name, kw in (('npts_x1.5', dict(nmax=nmax, window=win, npts=60000)),
                     ('nmax_x1.25', dict(nmax=int(nmax * 1.25), window=win, npts=40000)),
                     ('window_wider', dict(nmax=nmax, window=(max(win - 0.3, 1e-9) if win else None), npts=40000))):
        if name == 'window_wider' and win is None:
            continue
        v = evolve_current(lam, beta, [T - dw, T], kw['nmax'], kw['window'], kw['npts'])
        var[name] = dict(slope_dw=lam * (v['mean'][1] - v['mean'][0]) / dw, current_T=lam * v['current'][1])
    rec['variants'] = var
    rec['max_variant_change'] = max(max(abs(x['slope_dw'] / rec['slope_dw'] - 1), abs(x['current_T'] / rec['current_T'] - 1)) for x in var.values())
    for k in ('slope_dw', 'slope_half', 'slope_2dw', 'slope_dw_T2', 'current_T', 'current_T2'):
        rec['dev_' + k] = rec[k] / r5 - 1
    # Richardson extrapolation assuming a t^-2 residual: dev(inf) = dev(T) - (dev(T/2) - dev(T)) / 3
    rec['dev_current_richardson'] = rec['dev_current_T'] - (rec['dev_current_T2'] - rec['dev_current_T']) / 3
    out.append(rec)
    print(f"lam={lam} bJ={beta}: dev slope_dw {rec['dev_slope_dw']:+.2e} half {rec['dev_slope_half']:+.2e} 2dw {rec['dev_slope_2dw']:+.2e} "
          f"current(T) {rec['dev_current_T']:+.2e} current(T/2) {rec['dev_current_T2']:+.2e}  variants {rec['max_variant_change']:.1e}  "
          f"norm {rec['norm_dev']:.1e}  {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open('data/window_study.json', 'w'), indent=1)
