import numpy as np
SIGMA_F, S_GEO = (0.45, 1.3)
BETA_MEAN, BETA_SD = (0.4, 0.15)
P_FLOW = 0.5
N_CURVE = 225
SLOPE_K = 1.0

def _g(x):
    return (1.0 - np.exp(-x)) / x

def beta_stretch_to_arrest(L, Wi, rho=2.0):
    tau = 2.0 * L / Wi
    return L * (1.0 - _g(tau) / _g(tau / rho)) / np.log(rho)

def beta_relax_before_arrest(L, dt_over_lam, rho=2.0):
    return L * (1.0 - np.exp(-dt_over_lam * (1.0 - 1.0 / rho))) / np.log(rho)

def beta_check_by_integration(L, tau, rho=2.0, T=1.0, n=4000):
    from scipy.optimize import brentq
    lam = T / tau
    tt = np.linspace(0.0, T, n)
    M = lambda e, l: np.trapezoid(e * np.exp(-(T - tt) / l), tt)
    target = M(2 * L / T, lam)
    e2 = brentq(lambda e: M(e, rho * lam) - target, 1e-08, 1000000.0)
    return (L - e2 * T / 2) / np.log(rho)

def stretch_from_measured_jet(d0=18800.0, d1=245.0, w0=0.965, w1=0.2545, rho0=0.9504, rho1=1.0995):
    v0v = 1.0 / ((1 - w0) * rho0) / (1.0 / ((1 - w1) * rho1))
    ln_r = np.log(d0 / d1)
    theta_p = 2 * ln_r - np.log(v0v)
    return (theta_p / 2.0, theta_p / (2 * ln_r))
