"""Checks every number quoted in the paper against the code and the stored data. Usage: python scripts/verify.py"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np, warnings; warnings.simplefilter('ignore')
from scipy.integrate import quad
from module_dssyk import rho_theta
from module_rate import rho_modular, rho_fourier, theta_series, rate, saddle_rate, R_ts, rate_deep_ir, ab_series, consts
n = [0, 0]; _lines = []
def check(name, cond, val=''):
    n[0] += 1; n[1] += bool(cond); line = f"{'PASS' if cond else 'FAIL'}  {name} {val}"; _lines.append(line); print(line)
def _finish():
    tail = f'{n[1]}/{n[0]} PASS'; print('\n' + tail)
    open('data/verify_report.txt', 'w').write('\n'.join(_lines) + '\n\n' + tail + '\n')
    sys.exit(0 if n[0] == n[1] else 1)
th = np.linspace(0.02, np.pi - 0.02, 50)
for lam in (0.3, 1.0, 4.0):
    e = np.abs(rho_fourier(th, lam) - rho_theta(th, np.exp(-lam))).max()
    check(f'Eq. (6) Fourier series = q-Hermite product, lambda = {lam}', e < 1e-12, f'{e:.1e}')
for lam in (0.05, 0.5, 1, 2, 4, 8):
    e = abs(theta_series(lam) - rate(lam, 0) / (2 * lam * consts(lam)[1]))
    check(f'Eq. (7) theta series = quadrature of Eq. (5), lambda = {lam}', e < 1e-11, f'{e:.1e}')
for lam in (0.05, 0.5, 2.0):
    r0 = rho_theta(th, np.exp(-lam)); e = np.abs(rho_modular(th, lam) / r0 - 1).max()
    check(f'Eq. (8) Poisson-resummed measure = q-Hermite product, lambda = {lam} (relative)', e < 1e-10, f'{e:.1e}')
check('q -> 0: <sin theta> -> 8/3pi', abs(theta_series(60) - 8 / (3 * np.pi)) < 1e-12)
check('q -> 1: <sin theta> -> 1', abs(theta_series(1e-4) - 1) < 1e-3, theta_series(1e-4))
d0 = json.load(open('data/beta0.json'))
for r in d0:
    rel = abs(r['exact'] / r['theta'] - 1)
    check(f"Table 1 exact chain vs Eq. (7), lambda = {r['lam']}", rel < 1e-4 and r['edge'] < 1e-25, f'{rel:.1e}')
for r in d0:
    rel = abs(r['ab'] / r['theta'] - 1)
    if r['lam'] <= 0.5: check(f"A-B two-term series vs Eq. (7), lambda = {r['lam']}", rel < 2e-8, f'{rel:.1e}')
rel = {r['lam']: abs(r['ab'] / r['theta'] - 1) for r in d0}
check('A-B series error 2e-5 (lambda=1), 4% (lambda=4), 42% (lambda=8)', 1e-5 < rel[1.0] < 3e-5 and 0.035 < rel[4.0] < 0.045 and 0.40 < rel[8.0] < 0.43, [f'{v:.2g}' for v in rel.values()])
bal = {r['lam']: r['ballistic'] / r['theta'] - 1 for r in d0}
check('ballistic excess 0.6% ... 17.8%', abs(bal[0.05] - 0.0062) < 5e-4 and abs(bal[8.0] - 0.178) < 2e-3, [f'{v:.3f}' for v in bal.values()])
d = json.load(open('data/finite_beta.json'))
P = d['points']
rels = [abs(p['slope_exact'] / p['eq5'] - 1) for p in P]
check('Table 3: finite-difference slopes vs Eq. (5), all points < 1e-3', max(rels) < 1e-3, f'max {max(rels):.1e}')
check('Table 3: norm of the evolved state conserved', max(p['norm_dev'] for p in P) < 1e-5, f"{max(p['norm_dev'] for p in P):.1e}")
above = [p for p in P if p['slope_exact'] * p['beta'] / (2 * np.pi) > 1.1]
check('exact evolution exceeds 2pi/beta at >= 5 points', len(above) >= 5, [round(p['slope_exact'] * p['beta'] / (2 * np.pi), 2) for p in above])
p2 = [p for p in P if p['lam'] == 2.0][0]
check('lambda = 2, beta J = 100: ratio 6.27, gap to Eq. (5) < 1e-3 at t = 1500', abs(p2['slope_exact'] * 100 / (2 * np.pi) - 6.27) < 0.01 and abs(p2['slope_exact'] / p2['eq5'] - 1) < 1e-3)
check('single saddle never exceeds 2pi/beta', max(c['max_saddle_ratio'] for c in d['cross']) < 1)
for c in d['curves'].values():
    b = np.array(c['beta']); s = np.array(c['saddle']) * b / (2 * np.pi)
    if (s > 1 + 1e-12).any(): check('saddle curve <= 1', False)
for b in (0.1, 0.5, 2.0, 10.0, 40.0):
    sh = lambda s: 0.5 * (np.exp(2 * np.pi * s - b * s ** 2 - np.pi ** 2 / b) - np.exp(-2 * np.pi * s - b * s ** 2 - np.pi ** 2 / b))
    up = np.pi / b + 40 / np.sqrt(b)
    num = quad(lambda s: s ** 2 * sh(s), 0, up, limit=400, epsabs=0, epsrel=1e-13)[0]
    den = quad(lambda s: s * sh(s), 0, up, limit=400, epsabs=0, epsrel=1e-13)[0]
    e = abs(b * num / den / np.pi - R_ts(b))
    check(f'Eq. (10) closed form vs quadrature, b = {b}', e < 1e-8, f'{e:.1e}')
check('R(b) > 1 for all b (grid)', (R_ts(np.geomspace(1e-3, 1e3, 2000)) > 1).all())
check('R(b) asymptotes', abs(R_ts(1e-3) - 1 - 1e-3 / (2 * np.pi ** 2)) < 1e-9 and abs(R_ts(1e5) / (2 * np.sqrt(1e5) / np.pi ** 1.5) - 1) < 1e-3)
ts = {(r['lam'], r['b']): r for r in d['ts']}
for b in (0.25, 1.0, 4.0):
    e = [abs(1 - ts[(l, b)]['ratio_shift']) for l in (0.04, 0.02, 0.01)]
    e0 = [abs(1 - ts[(l, b)]['ratio_R']) for l in (0.04, 0.02, 0.01)]
    check(f'Eq. (11) residual shrinks faster than O(lambda), b = {b}', e[0] > e[1] > e[2] and e[0] / e[2] > 8 and e0[0] / e0[2] < 4.5, [f'{x:.1e}' for x in e])
check('Eq. (11) at lambda = 0.01 within 3e-4', max(abs(1 - ts[(0.01, b)]['ratio_shift']) for b in (0.25, 1, 4)) < 3e-4)
for r in d['deepir']:
    if r['beta'] == 1e4: check(f"Eq. (12) deep IR, lambda = {r['lam']}, beta J = 1e4", abs(r['ratio'] - 1) < 5e-4, f"{r['ratio'] - 1:.1e}")
for c in d['cross']:
    if c['lam'] <= 0.2:
        e = c['beta_cross'] / (2 * np.pi / np.sqrt(c['lam'])) - 1
        check(f"crossing beta J ~ 2pi/sqrt(lambda), lambda = {c['lam']}", abs(e) < 0.12, f'{e:+.3f}')
cl = {c['lam']: c['beta_cross'] for c in d['cross']}
check('crossing values 61.7, 26.9, 12.7, 7.47, 4.85, 3.08, 1.99', all(abs(cl[l] - v) < 0.06 for l, v in ((0.01, 61.7), (0.05, 26.9), (0.2, 12.7), (0.5, 7.47), (1.0, 4.85), (2.0, 3.08), (4.0, 1.99))))
x = rate(0.01, 50) * 50 / (2 * np.pi)
check('lambda = 0.01, beta J = 50: ratio 0.987', abs(x - 0.987) < 1e-3, x)

# ---- Eq. (9): Bessel series at any temperature
from module_bessel import rate_bessel
from module_rate import rate_beta0
from module_bessel_hp import sin_avg_bessel_hp
for lam in (0.5, 1.0, 2.0, 4.0, 8.0):
    for bJ in (0.5, 2.0, 10.0, 50.0, 200.0):
        e = abs(rate_bessel(lam, bJ) / rate(lam, bJ) - 1)
        check(f'Eq. (9) double precision vs Eq. (5), lambda = {lam}, beta J = {bJ}', e < 2e-7, f'{e:.1e}')
check('Eq. (9) at beta = 0 equals Eq. (7)', abs(rate_bessel(0.3, 0.0) / rate_beta0(0.3) - 1) < 1e-13)
for lam, bJ in ((0.01, 2), (0.01, 50), (0.05, 10), (0.2, 1000)):
    s_, v_ = sin_avg_bessel_hp(lam, bJ); e = abs(s_ * v_ / rate(lam, bJ) - 1)
    check(f'Eq. (9) extended precision vs Eq. (5), lambda = {lam}, beta J = {bJ}', e < 1e-6, f'{e:.1e}')

# ---- Proposition 1' (Appendix A): the current converges as t^-2, <n>/t as t^-1; corrected moment bound
cc = json.load(open('data/current_check.json'))
for lam in (0.5, 2.0):
    rows = {r['t']: r for r in cc['beta0'] if r['lam'] == lam}
    check(f"current at t = 200 matches Eq. (7), lambda = {lam}", abs(rows[200]['dev_current']) < (2e-9 if lam == 0.5 else 3e-6), f"{rows[200]['dev_current']:.1e}")
    rc = [rows[t]['dev_current'] / rows[2 * t]['dev_current'] for t in (25, 50, 100)]
    check(f"current residual falls as t^-2 (factor 4 per doubling), lambda = {lam}", all(3.5 < x < 4.6 for x in rc), [f'{x:.2f}' for x in rc])
    rl = [rows[t]['dev_ell_over_t'] / rows[2 * t]['dev_ell_over_t'] for t in (25, 50, 100)]
    check(f"ell/t residual falls as t^-1 (factor 2 per doubling), lambda = {lam}", all(1.8 < x < 2.2 for x in rl), [f'{x:.2f}' for x in rl])
check("paper's old bound (beta B)^n/n! fails at n = 0 (value > 1)", all(r['value'] > r['paper_bound'] for r in cc['bound'] if r['n'] == 0))
check("corrected bound e^(beta B)(beta B)^n/n! holds at every tested n", all(r['value'] <= r['corrected_bound'] for r in cc['bound']))

# ---- Table 3: window, final time, truncation and precision (Appendix B)
ws = json.load(open('data/window_study.json'))
check('Table 3: |current(T) / Eq. (5) - 1| < 1e-3 at all ten points', max(abs(r['dev_current_T']) for r in ws) < 1e-3, f"{max(abs(r['dev_current_T']) for r in ws):.1e}")
check('Table 3: slopes over dw/2, dw, 2dw within 1e-3 of Eq. (5)', max(max(abs(r['dev_slope_half']), abs(r['dev_slope_dw']), abs(r['dev_slope_2dw'])) for r in ws) < 1e-3)
check('Table 3: truncation and quadrature variants change the result by < 2e-4', max(r['max_variant_change'] for r in ws) < 2e-4, f"{max(r['max_variant_change'] for r in ws):.1e}")
lt = [r for r in ws if r['T'] >= 400]
check('Table 3, T >= 400: residual at T/2 is 4 times that at T (t^-2)', all(3.5 < r['dev_current_T2'] / r['dev_current_T'] < 4.5 for r in lt), [f"{r['dev_current_T2'] / r['dev_current_T']:.2f}" for r in lt])
check('Table 3, T >= 400: Richardson-extrapolated current within 1e-5 of Eq. (5)', max(abs(r['dev_current_richardson']) for r in lt) < 1e-5, f"{max(abs(r['dev_current_richardson']) for r in lt):.1e}")
pt = [r for r in ws if r['lam'] == 2.0 and r['beta'] == 100.0][0]
check('lambda = 2, beta J = 100: residual 0.22% at T = 750 and 0.055% at T = 1500', abs(pt['dev_current_T2'] + 2.2e-3) < 1e-4 and abs(pt['dev_current_T'] + 5.5e-4) < 3e-5, f"{pt['dev_current_T2']:.2e}, {pt['dev_current_T']:.2e}")

# ---- Eq. (9): truncation, cancellation and precision (Appendix B)
bt = json.load(open('data/bessel_terms.json'))
check('Eq. (9) double precision vs extended precision, lambda >= 0.5, beta J <= 200: < 2e-10', max(r['rel_default_double'] for r in bt['truncation']) < 2e-10, f"{max(r['rel_default_double'] for r in bt['truncation']):.1e}")
check('direct quadrature of Eq. (5) vs extended-precision Eq. (9): < 2e-7', max(r['rel_quad_hp'] for r in bt['truncation']) < 2e-7, f"{max(r['rel_quad_hp'] for r in bt['truncation']):.1e}")
check('Eq. (9) converged at M = 12 terms of the theta sum, lambda >= 0.5', all(r['M_scan'][5]['rel'] < 1e-9 for r in bt['truncation']))
check('extended precision stable under 30 more digits', max(r['rel_30_60'] for r in bt['precision']) < 1e-14)
check('double precision fails for lambda <= 0.05 (cancellation > 1e16)', all(r['rel_double'] > 1 and r['cancellation'] > 1e16 for r in bt['precision'] if r['lam'] <= 0.05))

# ---- crossing of 2 pi / beta (Section 5.3)
cx = json.load(open('data/crossing_check.json'))
sm = [r for r in cx if r['lam'] <= 0.2]
check('crossing: closed form from Eq. (11) within 2% for lambda <= 0.2', max(abs(r['err_closed']) for r in sm) < 0.021, [f"{r['err_closed']:+.3f}" for r in sm])
check('crossing: leading estimate 2pi/sqrt(lambda) within 11% for lambda <= 0.2', max(abs(r['err_leading']) for r in sm) < 0.11, [f"{r['err_leading']:+.3f}" for r in sm])

# ---- origin of the t^-2 residual (Appendix B.5): exponent -(alpha/2) - 1 for edge vanishing theta^alpha
ee = json.load(open('data/edge_exponent.json'))
for r in ee:
    if r['model'] == 'dssyk' and r['lam'] == 0.3 and r['p'] > 1:
        continue                                   # residual below double precision
    last = r['local_exponents'][-2:]
    name = 'free chain' if r['model'] == 'free' else f"chord chain, lambda = {r['lam']}"
    check(f"edge exponent, {name}, alpha = {r['alpha']}: residual ~ t^{r['predicted']}", all(abs(x - r['predicted']) < 0.05 for x in last), [f'{x:.2f}' for x in last])

# ---- growth at fixed entanglement (Section 6; spread/PREREG.md, registered before the runs)
sp = json.load(open('spread/spread.json')); st = [r for r in sp['states'] if r.get('solved')]
check('fixed entanglement: matched states', len(st) == 40, len(st))
check('fixed entanglement: entropy matched within 3e-4', max(abs(r['dS']) for r in st) < 3e-4, f"{max(abs(r['dS']) for r in st):.1e}")
for r in sp['D1']:
    check(f"Prop. 1' for a non-thermal state ({r['kind']}, lambda = 0.3): slope = 2 lambda B <sin theta>", r['rel_diff'] < 1e-10, f"{r['rel_diff']:.1e}")
check('fixed entanglement: every matched rate differs from the thermal one by > 1e-3 (relative)', min(r['rel_dR'] for r in st) > 1e-3, f"{min(r['rel_dR'] for r in st):.1e}")
check('fixed entanglement: largest relative rate difference (quoted 0.93)', abs(max(r['rel_dR'] for r in st) - 0.926) < 0.005, f"{max(r['rel_dR'] for r in st):.3f}")
for lam, tol in ((0.05, 0.05), (0.3, 0.2)):
    e = max(r['rel_err_L'] for r in st if r['lam'] == lam)
    check(f'spread law dR = -2 lambda B kappa dH within {tol:.0%}, lambda = {lam}', e <= tol, f'{e:.3f}')
e1 = max(r['rel_err_L'] for r in st if r['lam'] == 1.0)
check('spread law at lambda = 1: largest error (quoted 0.19)', 0.18 < e1 < 0.19, f'{e1:.3f}')
check('lambda = 1: dR and dH of opposite sign in every state', all(r['dR'] * r['dH'] < 0 for r in st if r['lam'] == 1.0))
g = [r for r in st if r['lam'] == 0.05 and r['beta'] == 1.0]
rh = [abs(r['dR'] / r['dH']) for r in g]; rc = [abs(r['dR'] / r['dC']) for r in g]
check('lambda = 0.05, beta = 1: dR/dH varies by < 10% across families (quoted 3.7%)', max(rh) / min(rh) - 1 < 0.10, f'{max(rh) / min(rh) - 1:.3f}')
check('lambda = 0.05, beta = 1: dR/dC varies by > factor 2 (quoted 3.3)', max(rc) / min(rc) > 2, f'{max(rc) / min(rc):.2f}')
check('every matched state is less spread and faster than the thermal one', all(r['dH'] < 0 and r['dR'] > 0 for r in st))
# affine relation sin(theta) ~ c1 + kappa log rho over thermal states (quoted 1-2% at lambda = 0.05, 10-24% at lambda = 1)
def _aff(lam, beta):
    q, B = consts(lam); tt = np.linspace(1e-6, np.pi - 1e-6, 40001); rr = np.maximum(rho_theta(tt, q), 1e-300); rr /= np.trapezoid(rr, tt)
    EE = 2 * B * np.cos(tt); w = np.exp(np.log(rr) - beta * (EE - EE.min())); mu = w / np.trapezoid(w, tt); m = mu > 1e-14 * mu.max()
    W = mu[m] * np.gradient(tt)[m]; x, y = np.log(rr[m]), np.sin(tt[m])
    k = np.cov(x, y, aweights=W)[0, 1] / np.cov(x, aweights=W); c = np.average(y - k * x, weights=W)
    return np.sqrt(np.average((y - c - k * x) ** 2, weights=W)) / np.sqrt(np.cov(y, aweights=W))
a005 = [_aff(0.05, b) for b in (1, 5, 10, 20)]; a1 = [_aff(1.0, b) for b in (1, 5, 10, 20)]
check('affine residual of sin(theta) on log rho, lambda = 0.05, beta J = 1-20: 1-2%', 0.005 < min(a005) and max(a005) < 0.025, [f'{x:.3f}' for x in a005])
check('affine residual, lambda = 1, beta J = 1-20: 10-24%', 0.09 < min(a1) and max(a1) < 0.245, [f'{x:.3f}' for x in a1])
tab = {(0.05, 1.0, 'gauss', 0.05): (0.009, 1.025), (0.05, 1.0, 'bimodal', 0.1): (0.019, 1.011), (0.05, 1.0, 'uniform', 0.05): (0.028, 1.004),
       (0.3, 5.0, 'gauss', 0.05): (0.051, 1.112), (0.3, 5.0, 'bimodal', 0.1): (0.132, 1.034), (0.3, 5.0, 'uniform', 0.05): (0.221, 0.998),
       (1.0, 5.0, 'gauss', 0.05): (0.285, 0.970), (1.0, 5.0, 'bimodal', 0.4): (0.569, 0.858), (1.0, 5.0, 'uniform', 0.05): (0.926, 0.815)}
for key, (rel, ratio) in tab.items():
    r = [x for x in st if (x['lam'], x['beta'], x['kind'], x['width']) == key][0]
    check(f'Table 4 row {key}: dR/R and Eq. (17)/dR', abs(r['dR'] / r['ref']['R'] - rel) < 6e-4 and abs(r['pred_L'] / r['dR'] - ratio) < 6e-4, f"{r['dR'] / r['ref']['R']:.4f}, {r['pred_L'] / r['dR']:.4f}")
# ---- Proposition 2 and Theorem 2 (Section 3, Appendix C): identity, t^-2 law and its coefficient
rl = json.load(open('data/rate_law.json'))
for c in rl['cases']:
    big = [r for r in c['rows'] if r['two_qn'] > 1e-9]          # below that the finite difference is limited by rounding
    if big:
      e = max(abs(r['d2l_fd'] / r['two_qn'] - 1) for r in big)
      check(f"Prop. 2: d2 ell/dt2 = 2<q^n> by finite difference, {c['state']}, lambda = {c['lam']}", e < 1e-3, f'{e:.1e}')
    last = c['rows'][-1]
    check(f"Thm. 2: residual / leading term at t = {last['t']}, {c['state']} (alpha = {c['alpha']}), lambda = {c['lam']}", abs(last['ratio'] - 1) < 2.5e-3, f"{last['ratio']:.5f}")
    check(f"Prop. 2: residual positive (rate approached from below), {c['state']}", all(r['residual'] > 0 for r in c['rows']))
for lam in (0.5, 2.0):
    r = [x for x in rl['table_B1'] if x['lam'] == lam and x['t'] == 200][0]
    check(f'Thm. 2 vs Table B1, beta = 0, lambda = {lam}, t = 200', abs(r['observed'] / r['predicted'] - 1) < 3e-3, f"{r['observed'] / r['predicted']:.4f}")
ok = [r for r in rl['table_B2'] if r['lam'] >= 0.2]
e = max(abs(r['observed_T'] / r['predicted_T'] - 1) for r in ok); e2 = max(abs(r['observed_T2'] / r['predicted_T2'] - 1) for r in ok)
check('Thm. 2 vs Table B2, six points with lambda >= 0.2: within 3% at T and 7% at T/2', e < 0.03 and e2 < 0.07, f'{e:.3f}, {e2:.3f}')
bad = [r for r in rl['table_B2'] if r['lam'] <= 0.05]
check('Table B2, four points with lambda <= 0.05: leading term not yet reached at T (observed > 5x predicted)', all(abs(r['observed_T']) > 5 * abs(r['predicted_T']) for r in bad))
r = [x for x in rl['table_B2'] if x['lam'] == 2.0][0]
check('lambda = 2, beta J = 100: predicted 0.219% at t = 750 and 0.0548% at t = 1500 (quoted 0.22%, 0.055%)', abs(-r['predicted_T2'] - 2.19e-3) < 1e-5 and abs(-r['predicted_T'] - 5.48e-4) < 3e-6, f"{-r['predicted_T2']:.3e}, {-r['predicted_T']:.3e}")
q = np.exp(-0.7); qp = np.prod(1 - q ** np.arange(1, 400))
H = [1.0, 2.0]                                     # H_n(1|q): H_{n+1} = 2 H_n - (1 - q^n) H_{n-1}
for k in range(1, 300): H.append(2 * H[-1] - (1 - q ** k) * H[-2])
qn = np.cumprod(np.concatenate([[1.0], 1 - q ** np.arange(1, 301)]))
S = sum(q ** k * H[k] ** 2 / qn[k] for k in range(300))
check('Mehler: sum q^n H_n(1|q)^2/(q;q)_n = 1/((1-q)(q;q)^3), q = e^-0.7', abs(S * (1 - q) * qp ** 3 - 1) < 1e-12, f'{S * (1 - q) * qp ** 3 - 1:.1e}')
check('bound H_n(1|q) <= (n+1)/(q;q)^2, n < 300', all(H[k] <= (k + 1) / qp ** 2 for k in range(300)))
_finish()
