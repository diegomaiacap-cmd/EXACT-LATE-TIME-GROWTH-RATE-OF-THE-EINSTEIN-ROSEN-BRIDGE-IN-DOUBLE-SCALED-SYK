import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np
from module_bessel import rate_bessel
from module_rate import rate, theta_series, rate_beta0
out = []
for lam in [0.01, 0.05, 0.2, 0.5, 1, 2, 4, 8]:
    for bJ in [0, 0.5, 2, 10, 50, 200, 1000, 5000]:
        t0 = time.time(); rb = rate_bessel(lam, bJ); rq = rate(lam, bJ) if bJ > 0 else rate_beta0(lam)
        out.append(dict(lam=lam, bJ=bJ, bessel=rb, quad=rq, rel=abs(rb/rq-1)))
        print(f"lam={lam:<5} bJ={bJ:<6} bessel={rb:.10g} quad={rq:.10g} rel={abs(rb/rq-1):.1e} {time.time()-t0:.2f}s", flush=True)
json.dump(out, open('data/bessel_check.json', 'w'), indent=1)
