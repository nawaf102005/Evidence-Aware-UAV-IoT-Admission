"""Revision 12: explicit ordered Riccati forgetting and radial error bounds.

Thompson contraction is established theory (Bougerol 1993; Lawson--Lim 2007).
This module implements elementary, verifiable constants for the manuscript's
ordered endpoints and a sharp 2-D radial transfer bound. It does not assert
uniform exponential forgetting on arbitrary all-loss histories.
"""
import math
import numpy as np
from scipy.linalg import eigvalsh


def ordered_log_ratio(lower, upper):
    """log min{c: upper <= c lower}; requires 0 < lower <= upper."""
    L, U = np.asarray(lower, float), np.asarray(upper, float)
    if L.shape != U.shape or L.shape != (2, 2):
        raise ValueError('Two matching 2 by 2 matrices required.')
    if np.linalg.eigvalsh(L).min() <= 0 or np.linalg.eigvalsh(U-L).min() < -1e-10:
        raise ValueError('Ordered positive definite endpoints required.')
    return max(0., float(np.log(eigvalsh(U, L).max())))


def sharp_radial_gap(log_ratio):
    """Sharp sup of psi_R(U)-psi_R(L) when L <= U <= exp(d)*L.

    The expression uses log1p/expm1; its continuous value at d=0 is zero.
    Equality is attainable with isotropic endpoints and a suitable scale.
    """
    d = np.asarray(log_ratio, float)
    if np.any(d < 0) or np.any(~np.isfinite(d)):
        raise ValueError('Finite nonnegative log ratios required.')
    h = -np.expm1(-d)
    inverse_term = np.zeros_like(d)
    np.divide(d, np.expm1(np.minimum(d, 700.)), out=inverse_term, where=d > 0)
    inverse_term = np.where(d > 700., 0., inverse_term)
    return h*np.exp(-inverse_term)


def prediction_log_ratio(d, upper_eigenvalue, process_floor):
    """Nonlinear ordered contraction log[(b+q)/(b exp(-d)+q)]."""
    if d < 0 or upper_eigenvalue <= 0 or process_floor <= 0:
        raise ValueError('Nonnegative distance and positive b,q required.')
    b, q = upper_eigenvalue, process_floor
    return float(np.log1p(b/q)-np.log1p((b/q)*np.exp(-d)))


def step_log_bound(d, upper_eigenvalue, process_floor, information_floor=0.):
    """Prediction followed by a common information update.

    Q >= q I; both predicted endpoints >= q I; S >= s I. Information
    inversion then addition gives the extra factor 1/(1+q*s).
    """
    if information_floor < 0:
        raise ValueError('Nonnegative information floor required.')
    return prediction_log_ratio(d, upper_eigenvalue, process_floor)/(1+process_floor*information_floor)


def information_moment(action, process_floor):
    """E[1/(1+q lambda_min(S_E))] under the declared reception law."""
    if process_floor <= 0:
        raise ValueError('Strictly positive process floor required.')
    minimum = np.maximum(np.linalg.eigvalsh(action.information)[:, 0], 0.)
    return float(action.probabilities @ (1/(1+process_floor*minimum)))


def memory_width_bound(initial_log_ratio, moments, terms=128):
    """Expected ordered-angular enclosure width, exact arithmetic.

    Initial distance must be a deterministic valid bound. Product moments
    require independent patterns/fixed actions, or uniform conditional bounds.
    Ordinary floating-point safety allowances are added by the risk kernel.
    """
    if initial_log_ratio < 0 or terms < 2:
        raise ValueError('Valid distance and angular grid required.')
    eta = np.asarray(moments, float)
    if np.any((eta < 0) | (eta > 1+1e-12)):
        raise ValueError('Contraction moments must lie in [0,1].')
    return min(1., initial_log_ratio*float(np.prod(eta))/math.e+1/terms)


def sufficient_depth(initial_log_ratio, rho, tolerance, terms=128):
    """Uniform bounded-covariance depth; caller must establish b and rho."""
    if initial_log_ratio < 0 or not 0 < rho < 1 or terms < 2:
        raise ValueError('Valid distance, strict contraction, and grid required.')
    if tolerance <= 1/terms:
        return None  # This quadrature-width bound cannot certify the request.
    if initial_log_ratio/math.e <= tolerance-1/terms:
        return 1
    return 1+int(math.ceil(math.log(initial_log_ratio/(math.e*(tolerance-1/terms)))/(-math.log(rho))))


def all_loss_endpoints(steps, initial_upper, process):
    """Exact counterexample: lower=wq I, upper=(b+wq) I, no observations."""
    if steps < 1 or initial_upper <= 0 or process <= 0:
        raise ValueError('Positive steps, b, and q required.')
    return steps*process*np.eye(2), (initial_upper+steps*process)*np.eye(2)
