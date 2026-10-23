import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
from module_bessel_hp import sin_avg_bessel_hp
from module_rate import rate
out = []
for lam, bJ in [(0.5, 50), (0.2, 50), (0.2, 1000), (0.2, 5000), (0.05, 10), (0.05, 200), (0.01, 2), (0.01, 50)]:
    t0 = time.time(); s, v = sin_avg_bessel_hp(lam, bJ); rb = s * v; rq = rate(lam, bJ)
    out.append(dict(lam=lam, bJ=bJ, bessel_hp=rb, quad=rq, rel=abs(rb / rq - 1)))
    print(f"lam={lam} bJ={bJ} bessel_hp={rb:.12g} quad={rq:.12g} rel={abs(rb/rq-1):.1e} {time.time()-t0:.1f}s", flush=True)
json.dump(out, open('data/bessel_hp_check.json', 'w'), indent=1)
