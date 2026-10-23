"""Crossing of 2 pi / beta (Section 5.3, referee point 2.3).

Inside the triple-scaling form, Eq. (11), and for b << pi^2 (so that erf(pi/sqrt b) = 1 up to e^{-pi^2/b}),
  (beta/2pi) rate = (b / b~) (1 + b~ / 2 pi^2),   b~ = b + 2 lambda.
Setting it to 1 gives b^2 + 2 lambda b - 4 pi^2 lambda = 0, i.e.
  b_x = sqrt(lambda^2 + 4 pi^2 lambda) - lambda,   beta_x J = b_x / (lambda^2 B),
whose leading term is beta_x J = 2 pi / sqrt(lambda) (Eq. (14)).  Neglected: O(lambda^2) corrections to Eq. (11)
and e^{-pi^2/b_x} from the error function.  This script compares both estimates with the exact crossing of Eq. (5).
-> data/crossing_check.json"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from scipy.optimize import brentq
from module_rate import rate, rate_ts, consts

out = []
for lam in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
    q, B = consts(lam)
    lead = 2 * np.pi / np.sqrt(lam); lo, hi = np.log(0.05), np.log(max(1.5 * lead, 50))
    exact = float(np.exp(brentq(lambda lb: rate(lam, np.exp(lb)) * np.exp(lb) / (2 * np.pi) - 1, lo, hi)))
    eq11 = float(np.exp(brentq(lambda lb: rate_ts(lam, np.exp(lb)) * np.exp(lb) / (2 * np.pi) - 1, lo, hi)))
    bx = np.sqrt(lam ** 2 + 4 * np.pi ** 2 * lam) - lam
    closed = bx / (lam ** 2 * B)
    out.append(dict(lam=lam, exact=exact, eq11_numeric=eq11, closed=float(closed), leading=float(lead), b_x=float(bx),
                    erf_neglect=float(np.exp(-np.pi ** 2 / bx)),
                    err_closed=float(closed / exact - 1), err_leading=float(lead / exact - 1), err_eq11=float(eq11 / exact - 1)))
    print(f"lambda={lam:<7} exact {exact:9.3f}  Eq.(11) {eq11:9.3f} ({eq11 / exact - 1:+.3f})  closed {closed:9.3f} ({closed / exact - 1:+.3f})  "
          f"2pi/sqrt(lambda) {lead:9.3f} ({lead / exact - 1:+.3f})  b_x {bx:.3f}  e^(-pi^2/b_x) {np.exp(-np.pi ** 2 / bx):.0e}", flush=True)
json.dump(out, open('data/crossing_check.json', 'w'), indent=1)
