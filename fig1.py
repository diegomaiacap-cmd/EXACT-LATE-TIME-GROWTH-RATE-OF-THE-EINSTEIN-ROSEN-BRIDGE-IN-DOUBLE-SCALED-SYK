"""Figure 1. Usage: python scripts/fig1.py pt|en"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker as tk
from module_rate import theta_series, ab_series, consts, R_ts, rate
lang = sys.argv[1] if len(sys.argv) > 1 else 'pt'
B_, O, TX, MU, SF, GR = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e1'
COL = {'0.01': '#0e2a4f', '0.05': '#1f5aa6', '0.2': '#4f93e0', '0.5': '#f2a27d', '1.0': O, '2.0': '#a8401a'}
T = dict(pt=dict(dec=',', a='(a) Temperatura infinita', ax='λ', ay='⟨sin θ⟩₀ = (dℓ/dt) / 2λB',
                 th='série teta, Eq. (7)', ab='Alfinito–Beccaria, dois termos', ex='evolução exata da cadeia', sc='semicírculo, 8/3π',
                 b='(b) Temperatura finita', bx='βJ', by='(dℓ/dt) · β/2π', eq='Eq. (5)', sd='sela única', ts='Eq. (11), triplo escalonamento',
                 c='(c) Limite de triplo escalonamento', cx='b = λ²Bβ', cy='(dℓ/dt) · β/2π', R='R(b), Eq. (10)', a1='1 + b/2π²', a2=r'$2\sqrt{b}/\pi^{3/2}$',
                 out='figures/pt/fig1.png',
                 foot='(b) Linhas cheias: Eq. (5); tracejadas: sela única; pontilhadas: Eq. (11); círculos: evolução exata do duplo termocampo. (c) Eq. (5) com λ = 0,04 e 0,01 em função de b, convergindo para R(b).'),
         en=dict(dec='.', a='(a) Infinite temperature', ax='λ', ay='⟨sin θ⟩₀ = (dℓ/dt) / 2λB',
                 th='theta series, Eq. (7)', ab='Alfinito–Beccaria, two terms', ex='exact chain evolution', sc='semicircle, 8/3π',
                 b='(b) Finite temperature', bx='βJ', by='(dℓ/dt) · β/2π', eq='Eq. (5)', sd='single saddle', ts='Eq. (11), triple scaling',
                 c='(c) Triple-scaling limit', cx='b = λ²Bβ', cy='(dℓ/dt) · β/2π', R='R(b), Eq. (10)', a1='1 + b/2π²', a2=r'$2\sqrt{b}/\pi^{3/2}$',
                 out='figures/en/fig1.png',
                 foot='(b) Solid: Eq. (5); dashed: single saddle; dotted: Eq. (11); circles: exact evolution of the thermofield double. (c) Eq. (5) at λ = 0.04 and 0.01 as a function of b, converging to R(b).'))[lang]
fc = lambda v, _: f'{v:g}'.replace('.', T['dec'])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.7), facecolor=SF)
for a in ax:
    a.set_facecolor(SF); a.grid(color=GR, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
# (a)
a = ax[0]; L = np.geomspace(0.01, 12, 200)
norm = lambda l: 2 * l * consts(l)[1]
a.plot(L, [theta_series(l) for l in L], '-', color=B_, lw=2, label=T['th'])
a.plot(L, [ab_series(l) / norm(l) for l in L], '--', color=O, lw=1.6, label=T['ab'])
d0 = json.load(open('data/beta0.json'))
a.plot([r['lam'] for r in d0], [r['exact'] / norm(r['lam']) for r in d0], 'o', color=TX, ms=5, label=T['ex'])
a.axhline(8 / (3 * np.pi), color=MU, lw=1, ls=(0, (4, 2)), label=T['sc'])
a.set_xscale('log'); a.set_ylim(0.6, 1.02); a.set_xlabel(T['ax']); a.set_ylabel(T['ay']); a.set_title(T['a'], loc='left', fontsize=10.5)
a.legend(frameon=False, fontsize=8, loc='lower left')
# (b)
a = ax[1]; d = json.load(open('data/finite_beta.json'))
for lam, c in d['curves'].items():
    b = np.array(c['beta']); f = b / (2 * np.pi)
    a.plot(b, np.array(c['eq5']) * f, '-', color=COL[lam], lw=2, label=f'λ = {lam.rstrip("0").rstrip(".")}'.replace('.', T['dec']))
    a.plot(b, np.array(c['saddle']) * f, '--', color=COL[lam], lw=1.1)
    if float(lam) <= 0.2: a.plot(b, np.array(c['ts']) * f, ':', color=COL[lam], lw=1.4)
for p in d['points']:
    k = str(p['lam']) if str(p['lam']) in COL else None
    a.plot(p['beta'], p['slope_exact'] * p['beta'] / (2 * np.pi), 'o', mfc='none', mec=COL.get(k, TX), mew=1.5, ms=6)
a.axhline(1, color=TX, lw=1)
a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(0.08, 30); a.set_xlabel(T['bx']); a.set_ylabel(T['by']); a.set_title(T['b'], loc='left', fontsize=10.5)
h, l = a.get_legend_handles_labels()
h += [plt.Line2D([], [], color=MU, ls='--'), plt.Line2D([], [], color=MU, ls=':', lw=1.4)]; l += [T['sd'], T['ts']]
a.legend(h, l, frameon=False, fontsize=8, loc='upper left')
# (c)
a = ax[2]; bb = np.geomspace(0.05, 60, 200)
a.plot(bb, R_ts(bb), '-', color=TX, lw=2.2, label=T['R'])
a.plot(bb[bb < 8], 1 + bb[bb < 8] / (2 * np.pi ** 2), ':', color=MU, lw=1.2, label=T['a1'])
a.plot(bb[bb > 1], 2 * np.sqrt(bb[bb > 1]) / np.pi ** 1.5, '--', color=MU, lw=1.2, label=T['a2'])
for lam, c in ((0.04, '#7fb0ea'), (0.01, B_)):
    Bc = consts(lam)[1]; bs = np.geomspace(0.05, 30, 40); be = bs / (lam ** 2 * Bc)
    a.plot(bs, [rate(lam, x) * x / (2 * np.pi) for x in be], 'o', color=c, ms=3.5, label=f'Eq. (5), λ = {lam:g}'.replace('.', T['dec']))
a.set_xscale('log'); a.set_xlabel(T['cx']); a.set_ylabel(T['cy']); a.set_title(T['c'], loc='left', fontsize=10.5)
a.legend(frameon=False, fontsize=8, loc='upper left')
for a_ in ax:
    a_.yaxis.set_major_formatter(tk.FuncFormatter(fc)); a_.xaxis.set_major_formatter(tk.FuncFormatter(fc))
fig.text(0.01, 0.01, T['foot'], fontsize=7.5, color=MU)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(T['out'], dpi=200); print('ok')
