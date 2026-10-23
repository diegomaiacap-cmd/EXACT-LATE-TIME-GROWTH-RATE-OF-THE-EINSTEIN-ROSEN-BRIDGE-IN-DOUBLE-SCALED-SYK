Exact late-time growth rate of the Einstein–Rosen bridge in double-scaled SYK — code and data

This repository reproduces every number, table and figure of the paper

> D. Diniz Maia, *Exact late-time growth rate of the Einstein–Rosen bridge in double-scaled SYK* (2026).

In double-scaled SYK the renormalized length of the Einstein–Rosen bridge is ℓ = λn, with n the chord number. The paper proves that dℓ/dt converges to the group velocity 2λB sin θ averaged over the spectral measure of the state, and hence so does ℓ(t)/t (Proposition 1, Appendix A), proves that the limit is approached from below as t^(−α/2−1) with an exact coefficient (Proposition 2 and Theorem 2, Section 3.1, Appendix C), evaluates the canonical average in closed form (a theta series at β = 0, Eq. (7); a ratio of Bessel series at any β, Eq. (9)), derives the triple-scaling limit and its finite-λ shift (Eqs. (10)–(11)), the deep-infrared rate (Eq. (12)), and checks everything against exact evolution of the thermofield double (Eq. (15)).

Contents

| Path | What it does |
| --- | --- |
| `src/module_dssyk.py` | chord chain, q-Hermite measure, exact energy-representation evolution, Eq. (15) |
| `src/module_rate.py` | exact rate (5), theta series (7), Poisson form (8), R(b) (10), shifted formula (11), deep IR (12), single saddle (13) |
| `src/module_bessel.py` | Bessel series (9) in double precision (reliable for λ ≥ 0.5) |
| `src/module_bessel_hp.py` | Bessel series (9) in extended precision with Python's `decimal` (about 2.2/λ digits) |
| `scripts/run_beta0.py` | Table 1 (β = 0) → `data/beta0.json` |
| `scripts/run_finite_beta.py all` | Tables 2–3, crossings of 2π/β, deep IR, curves of Figure 1 → `data/finite_beta.json` |
| `src/module_current.py` | exact evolution in the energy representation, chunked, returning the current ⟨J⟩ = d⟨n⟩/dt of Proposition 1 |
| `scripts/check_bessel.py`, `scripts/check_bessel_hp.py` | Eq. (9) against direct quadrature of Eq. (5) → `data/bessel_*check.json` |
| `scripts/check_current.py` | Appendix B.1 (current vs ℓ/t at β = 0) and the corrected moment bound of Section 3 → `data/current_check.json` |
| `scripts/check_windows.py` | Appendix B.2: window, final time, truncation and quadrature for the ten points of Table 3 → `data/window_study.json` (about 2 min) |
| `scripts/check_bessel_terms.py` | Appendix B.3: truncation, cancellation and precision of Eq. (9) → `data/bessel_terms.json` |
| `scripts/check_crossing.py` | Section 5.3 and Appendix B.4: crossing of 2π/β, closed form and leading estimate → `data/crossing_check.json` |
| `scripts/check_rate_law.py` | Section 3.1, Appendix B.5: identity d²ℓ/dt² = 2⟨q^n⟩ and residual / leading term of Theorem 2 for five states; comparison with Tables B1–B2 → `data/rate_law.json` |
| `scripts/check_edge_exponent.py` | Appendix B.5: residual exponent −α/2 − 1 for states whose spectral density vanishes as θ^α at the band edges → `data/edge_exponent.json` |
| `scripts/fig1.py en` | Figure 1 → `figures/en/fig1.png` (`pt` for the Portuguese version) |
| `spread/` | Section 7, growth at fixed entanglement: registration with results appended (`PREREG.md`), `run_spread.py` → `spread.json` (about 10 min) |
| `scripts/verify.py` | 151 checks of every quoted number against the code and the stored data → `data/verify_report.txt` |

Requirements

Python ≥ 3.10 with NumPy ≥ 2.0, SciPy ≥ 1.13 and Matplotlib ≥ 3.8 (tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Quick check

```bash
python scripts/verify.py        # 151/151 PASS, under a minute (reads the stored data)
```

What the 151 checks verify

`scripts/verify.py` prints one line per check (PASS or FAIL, the name, the value) and writes the same lines to `data/verify_report.txt`. The tolerances are those of the statements in the paper.

| Paper | What is checked | Checks | Tolerance |
| --- | --- | --- | --- |
| Sec. 3, App. A | current d⟨n⟩/dt converges as t⁻², ⟨n⟩/t as t⁻¹, at β = 0 (Proposition 1) | 6 | deviation at t = 200 below 2 × 10⁻⁹ (λ = 0.5) and 3 × 10⁻⁶ (λ = 2); decay factor per doubling of t in 3.5–4.6 (current) and 1.8–2.2 (ℓ/t) |
| Sec. 3 | moment bound of the thermofield double: the old bound fails at n = 0, the corrected one holds | 2 | exact inequalities |
| Sec. 4 | Fourier series (6) and Poisson form (8) against the q-Hermite product | 6 | 10⁻¹² (absolute) and 10⁻¹⁰ (relative) |
| Sec. 4 | theta series (7) against quadrature of Eq. (5); limits q → 0 and q → 1 | 8 | 10⁻¹¹; 10⁻¹² and 10⁻³ |
| Sec. 4, Table 1 | exact chain at β = 0 against Eq. (7); weight at the truncation edge | 6 | 10⁻⁴; 10⁻²⁵ |
| Sec. 4, Table 1 | two-term expansion of [6] and ballistic estimate: the errors quoted in the text | 4 | as quoted |
| Sec. 4 | Bessel series (9) against quadrature of Eq. (5): 25 points in double precision, 4 in extended precision, β = 0 limit | 30 | 2 × 10⁻⁷ (double, λ ≥ 0.5, βJ ≤ 200); 10⁻⁶ (extended) |
| Sec. 5.1, Table 2 | R(b) against quadrature, R(b) > 1, its asymptotes; Eqs. (10)–(11) at fixed b | 11 | 10⁻⁸; quoted ratios; 3 × 10⁻⁴ for Eq. (11) at λ = 0.01 |
| Sec. 5.2 | deep-infrared rate (12) at βJ = 10⁴ | 3 | 5 × 10⁻⁴ |
| Sec. 5.3 | single saddle never exceeds 2π/β; crossing values; 2π/√λ within 12% | 6 | exact; 0.06 in βJ; 12% |
| Sec. 6, Table 3 | finite-difference slopes against Eq. (5); norm conservation; ratios above 2π/β | 4 | 10⁻³; 10⁻⁵ |
| App. B.2 | Table 3: current at T, window, truncation and quadrature, t⁻² ratio, Richardson extrapolation | 6 | 10⁻³; 2 × 10⁻⁴; ratio 3.5–4.5; 10⁻⁵ |
| App. B.3 | Eq. (9): double vs extended precision, quadrature error, 12 terms, +30 digits, failure of double precision at small λ | 5 | 2 × 10⁻¹⁰; 2 × 10⁻⁷; 10⁻⁹; 10⁻¹⁴ |
| App. B.4 | crossing: closed form within 2%, leading estimate within 11%, for λ ≤ 0.2 | 2 | 2.1%; 11% |
| App. B.5 | residual exponent −α/2 − 1 for α = 2, 4, 6 on the free and chord chains | 7 | last two local exponents within 0.05 |
| Sec. 3.1, App. B.5, C | Proposition 2 (finite difference), Theorem 2 (residual / leading term, five states), comparison with Tables B1–B2, the λ = 2, βJ = 100 numbers, Mehler sum, bound on Hₙ(1\|q) | 21 | 10⁻³; 2.5 × 10⁻³; 3 × 10⁻³ (B1), 3% at T and 7% at T/2 (B2); 10⁻⁵; 10⁻¹² |
| Sec. 7, Table 4 | growth at fixed entanglement: matching, Proposition 1 for non-thermal states, spread law (17), capacity, affine residuals, the nine rows of Table 4 | 24 | registered tolerances (spread/PREREG.md); 6 × 10⁻⁴ for the table |

Full reproduction

Every script can be run from any directory; paths are resolved from the repository root.

```bash
python scripts/run_beta0.py              # Table 1, about 1 min
python scripts/run_finite_beta.py all    # Tables 2-3 and the curves of Figure 1, about 10 min
python scripts/check_bessel.py           # Eq. (9), double precision
python scripts/check_bessel_hp.py        # Eq. (9), extended precision (the slowest step)
python scripts/check_current.py          # Appendix B.1
python scripts/check_windows.py          # Appendix B.2
python scripts/check_bessel_terms.py     # Appendix B.3
python scripts/check_crossing.py         # Section 5.3, Appendix B.4
python scripts/check_edge_exponent.py    # Appendix B.5
python scripts/check_rate_law.py         # Section 3.1, Appendices B.5 and C
python scripts/fig1.py en                # Figure 1
python spread/run_spread.py              # Section 7, about 10 min
python scripts/verify.py                 # all checks
```

Notes

Some comments and docstrings are in Portuguese. Equation, table and figure numbers refer to the paper.

License

Code: MIT (`LICENSE`). Data, figures and documentation: CC BY 4.0 (`LICENSE-DATA.md`).

Citation

See `CITATION.cff`. Please cite the paper when using this code or data.

