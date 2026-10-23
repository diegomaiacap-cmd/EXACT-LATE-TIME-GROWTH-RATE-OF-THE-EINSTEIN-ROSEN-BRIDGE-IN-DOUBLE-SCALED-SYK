"""Table 1 - late-time rate d ell/dt at infinite temperature (units of J, ell = lambda * C_K):
exact chord-chain evolution (slope between t = 100 and 120), closed theta series Eq. (7), quadrature of Eq. (5),
two-term series of Alfinito-Beccaria and the ballistic estimate 2 lambda B.  Writes data/beta0.json."""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from module_dssyk import Chain
from module_rate import rate_beta0, rate, ab_series, consts
rows = []
for lam, nmax in ((0.05, 7000), (0.5, 1500), (1.0, 1500), (2.0, 1200), (4.0, 1000), (8.0, 1000)):
    q, B = consts(lam)
    c = Chain(lam, nmax); m, _, edge = c.evolve(np.array([100.0, 120.0]))
    rows.append(dict(lam=lam, exact=float(lam * (m[1] - m[0]) / 20), theta=rate_beta0(lam), quad=rate(lam, 0.0),
                     ab=ab_series(lam), ballistic=float(2 * lam * B), edge=float(edge.max())))
    print(rows[-1], flush=True)
json.dump(rows, open('data/beta0.json', 'w'), indent=1)
