#!/usr/bin/env python3
"""Sanity-check the Tobit NLL functions before the full training run.

Two-tier:
  1) float64 math check of _log_phi_cdf vs scipy over a wide range
     (validates the Mills expansion itself).
  2) float32 check over the REALISTIC alpha range of the model
     ((C-f)/sigma with C=-10, f~-12..-8, sigma~0.5-2 => alpha in ~[-3, +6])
     plus finite-gradient checks in both branches.
"""

import common  # re-roots CWD to repo root



import warnings; warnings.filterwarnings('ignore')
import pathlib
import torch, numpy as np
from scipy.stats import norm

src = open(pathlib.Path(__file__).resolve().parent / 'tobit_censored.py').read()
a = src.index('def _log_phi_cdf')
b = src.index('best_va, best_state')
ns = {'torch': torch, 'np': np, 'C': -10.0}
exec(compile(src[a:b], 'fns', 'exec'), ns)
_log_phi_cdf = ns['_log_phi_cdf']
nll_tobit = ns['nll_tobit']

# 1) float64: math of the expansion
xs = np.linspace(-30, 10, 401)
worst = 0.0
for x in xs:
    got = _log_phi_cdf(torch.tensor([x], dtype=torch.float64)).item()
    ref = norm.logcdf(x)
    # direct branch: float64 torch.erf intrinsic precision ~1e-9 abs in Phi
    # => ~1e-8 in log Phi around here; Mills branch: 3/x^4
    tol = 1e-8 if x >= -6 else 3.0 / (x ** 4)
    assert abs(got - ref) < tol, (x, got, ref, tol)
    worst = max(worst, abs(got - ref))
print(f'1) float64 math check OK over [-30, 10], worst abs err = {worst:.2e}')

# 2) float32 over the realistic alpha range
r32 = np.linspace(-3, 6, 91)
worst32 = 0.0
for x in r32:
    got = _log_phi_cdf(torch.tensor([x], dtype=torch.float32)).item()
    ref = norm.logcdf(x)
    tol = 1e-4
    assert abs(got - ref) < tol, (x, got, ref)
    worst32 = max(worst32, abs(got - ref))
print(f'2) float32 realistic range [-3, +6] OK, worst abs err = {worst32:.2e}')

# 3) NLL direction: high f on a censored row -> large NLL; low f -> ~0
C = -10.0
l_hi = nll_tobit(torch.tensor(5.0), torch.tensor(-10.0), torch.tensor(1.0)).item()
l_lo = nll_tobit(torch.tensor(-15.0), torch.tensor(-10.0), torch.tensor(1.0)).item()
print(f'3) censored row: f=+5 -> nll={l_hi:.2f} (large); f=-15 -> nll={l_lo:.4f} (~0)')
assert l_hi > 5 and l_lo < 0.05

# 4) uncensored: zero residual -> nll = log(s) = 0 (formula omits const 0.5log2pi)
l_u = nll_tobit(torch.tensor(-8.0), torch.tensor(-8.0), torch.tensor(1.0)).item()
print(f'4) uncensored zero-residual: nll={l_u:.4f} (expect 0.0)')
assert abs(l_u) < 1e-6

# 5) grad of NLL flows into f (both branches), finite in float32
f = torch.tensor([5.0, -15.0, -10.0, -8.0], requires_grad=True)
y = torch.tensor([-10.0, -10.0, -10.0, -9.0])
nll_tobit(f, y, torch.tensor(1.0)).backward()
assert torch.isfinite(f.grad).all() and (f.grad.abs() > 0).all(), f.grad
print(f'5) NLL gradients finite and nonzero in both branches OK: {f.grad.tolist()}')
print('ALL SANITY CHECKS PASSED')
