"""Information-dependent Riccati contraction and scale-sensitive risk bounds.

R14 companion to the frozen R10--R13 engines. The finite-history covariance
sandwich has prior art (Rohr, Marelli and Fu, CDC-ECC 2011). These functions
do not assert a new general Riccati contraction theorem. In particular,
Hilbert's projective distance cannot certify absolute covariance scale.
All numerical enclosures use explicit floating-point allowances, not interval
arithmetic. Exact-arithmetic statements and assumptions are in paper.tex.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from memory_certificates import EYE, GUARD, posterior, radial_bounds


def sym(x):
    return (x + np.swapaxes(x, -1, -2))/2


def ordered_distance(lower, upper):
    """Ordered Thompson distance; both endpoints must be positive definite."""
    L, U = np.asarray(lower, float), np.asarray(upper, float)
    vals, vecs = np.linalg.eigh(L)
    if np.any(vals <= 0):
        raise ValueError('Positive definite lower endpoint required.')
    root = (vecs * (1/np.sqrt(vals))[..., None, :]) @ np.swapaxes(vecs, -1, -2)
    ratio = np.linalg.eigvalsh(sym(root @ U @ root))[..., -1]
    return np.maximum(0., np.log(np.maximum(ratio, 1.)))


def global_coefficient(a):
    """Sharp global scalar coefficient, also a valid matrix upper bound."""
    a = np.asarray(a, float)
    if np.any(a < 0):
        raise ValueError('Information index must be nonnegative.')
    return 1/(np.sqrt(1+a)+np.sqrt(a))**2


def information_index(Q, S):
    """Conservative floating-point evaluation of lambda_min(Q^1/2 S Q^1/2)."""
    Q, S = sym(np.asarray(Q, float)), sym(np.asarray(S, float))
    lam, V = np.linalg.eigh(Q)
    tol = 2e-11 * np.maximum(1., np.max(np.abs(lam), axis=-1))
    if np.any(lam[..., 0] < -tol):
        raise ValueError('Effective process covariance is not PSD.')
    root = (V*np.sqrt(np.maximum(lam, 0))[..., None, :]) @ np.swapaxes(V, -1, -2)
    ev = np.linalg.eigvalsh(sym(root @ S @ root))
    # Subtraction protects small spurious eigenvalues after block products.
    return np.maximum(0., ev[..., 0]-2e-11*np.maximum(1., np.max(np.abs(ev), axis=-1)))


def block_map(processes, informations):
    """Return A,Q_eff,S_eff for a composed Riccati fractional-linear map.

    F(P)=A[(P+Q_eff)^-1+S_eff]^-1 A.T. Broadcasting is allowed
    over reception patterns. Processes are covariance matrices, not scalars.
    """
    if len(processes) != len(informations):
        raise ValueError('Mismatched block lengths.')
    A, B, C, D = EYE.copy(), np.zeros((2, 2)), np.zeros((2, 2)), EYE.copy()
    for Q, S in zip(processes, informations):
        Q, S = np.asarray(Q), np.asarray(S)
        Ap, Bp = A+Q@C, B+Q@D
        Cp, Dp = S@A+(EYE+S@Q)@C, S@B+(EYE+S@Q)@D
        A, B, C, D = Ap, Bp, Cp, Dp
    Qe = sym(np.linalg.solve(A, B))
    Se = sym(np.swapaxes(A, -1, -2)@C)
    return A, Qe, Se


def block_coefficient(processes, informations):
    _, Qe, Se = block_map(processes, informations)
    return global_coefficient(information_index(Qe, Se))


def scaled_radial_gap(lower, distance, radius, terms=128):
    """Upper bound on U_N(U)-D_N(L) when L<=U<=exp(distance)*L.

    Uses the actual variance/radius scale and actual quadrature gap of L.
    Can be used with an a posteriori distance or an independently proved bound.
    """
    L = np.asarray(lower, float)
    d = np.asarray(distance, float)
    if np.any(d < 0):
        raise ValueError('Nonnegative distance required.')
    lam = np.linalg.eigvalsh(sym(L))
    if np.any(lam[..., 0] < -1e-10):
        raise ValueError('PSD lower endpoint required.')
    v = np.maximum(lam[..., 0], 0)[..., None] + (lam[..., 1]-lam[..., 0])[..., None]*np.cos(np.linspace(0, np.pi/2, terms+1))**2
    x = np.full_like(v, np.inf)
    np.divide(radius**2/2, v, out=x, where=v > 0)
    f = np.exp(-x)
    # exp(-x*exp(-d)), with d finite; no cancellation-sensitive subtraction
    # between two almost equal tail probabilities.
    enlarged = np.exp(-x*np.exp(-d)[..., None])
    diff = np.maximum(enlarged-f, 0.)
    return np.minimum(1., diff[..., :-1].mean(axis=-1)+(f[..., 0]-f[..., -1])/terms+8*GUARD)


def chernoff_fewer_than(probabilities, k):
    """Chernoff upper bound for K<k, K a sum of independent Bernoullis.

    Any theta>=0 gives a valid bound; bounded scalar optimization only tightens
    it. It never serves as evidence of temporal independence in measured data.
    """
    p = np.asarray(probabilities, float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid Bernoulli probability.')
    if k <= 0: return 0.
    if k > len(p): return 1.
    with np.errstate(divide='ignore'):
        l0, l1 = np.log1p(-p), np.log(p)
    def log_bound(theta):
        return theta*(k-1)+np.logaddexp(l0, l1-theta).sum()
    fit = minimize_scalar(log_bound, bounds=(0., 80.), method='bounded')
    return float(min(1., math.exp(min(0., log_bound(fit.x), log_bound(80.)))))


def radial_chunked(P, radius, terms, chunk=2048):
    """Avoid an exponential-history by angular-grid temporary allocation."""
    lo, hi = np.empty(len(P)), np.empty(len(P))
    for start in range(0, len(P), chunk):
        stop = min(start+chunk, len(P))
        lo[start:stop], hi[start:stop] = radial_bounds(P[start:stop], radius, terms)
    return lo, hi


class LazyWindowOracle:
    """Exact selected-window enumeration, without a dense A**w action table.

    This changes computation, not the certificate. It does NOT make the global
    action-sequence problem polynomial. Local-search plans must be labelled so.
    """
    def __init__(self, original):
        self.o = original
        self.cache = {}
        self.patterns_evaluated = 0
        self.queries = 0

    def local(self, t, ids, diagnostics=False):
        ids = tuple(ids)
        key = (t, ids, bool(diagnostics))
        if key in self.cache: return self.cache[key]
        o = self.o; s = t+1-len(ids)
        L = (o.initial if s == 0 else np.zeros((2, 2)))[None].copy()
        U = o.no_observation[s][None].copy()
        weights = np.ones(1)
        # Diagnostics also retain pattern-specific information compositions.
        AA = BB = CC = DD = None
        for j, idx in zip(range(s, t+1), ids):
            a = o.catalogs[j][idx]
            L = posterior(L[:, None]+o.process[j]*EYE, a.information[None]).reshape(-1, 2, 2)
            U = posterior(U[:, None]+o.process[j]*EYE, a.information[None]).reshape(-1, 2, 2)
            weights = (weights[:, None]*a.probabilities[None]).ravel()
        D, _ = radial_chunked(L, o.radius, o.terms)
        _, V = radial_chunked(U, o.radius, o.terms)
        result = dict(lower=max(0., float(weights@D)-GUARD),
                      upper=min(1., float(weights@V)+GUARD),
                      patterns=len(weights))
        if diagnostics:
            ap = np.empty(len(weights)); pred = np.empty(len(weights)); kappas = np.empty(len(weights))
            dboot = float(ordered_distance(o.process[s]*EYE, o.no_observation[s]+o.process[s]*EYE))
            if s == 0: dboot = 0.
            # Pattern-wise block factors, chunked to bound memory.
            radices = [len(o.catalogs[j][idx].probabilities) for j, idx in zip(range(s,t+1),ids)]
            for start in range(0, len(weights), 2048):
                stop = min(start+2048, len(weights)); indices = np.arange(start,stop)
                codes = np.array(np.unravel_index(indices, radices))
                infos = [o.catalogs[j][idx].information[codes[k]] for k,(j,idx) in enumerate(zip(range(s,t+1),ids))]
                qs = [np.zeros((2,2))]+[o.process[j]*EYE for j in range(s+1,t+1)]
                kap = block_coefficient(qs, infos)
                actual_d = ordered_distance(L[start:stop], U[start:stop])
                # The initial mission window has identical endpoints.
                predicted_d = np.zeros_like(kap) if s == 0 else kap*dboot
                pred[start:stop] = scaled_radial_gap(L[start:stop], predicted_d, o.radius, o.terms)
                ap[start:stop] = scaled_radial_gap(L[start:stop], actual_d, o.radius, o.terms)
                kappas[start:stop] = kap
                if np.any(actual_d > predicted_d+2e-6):
                    raise ArithmeticError('Block contraction failed numerical check.')
            result.update(width=float(weights@(V-D)), posterior_bound=float(weights@ap),
                          block_bound=float(weights@pred), mean_kappa=float(weights@kappas),
                          strict_block_probability=float(weights[kappas < 1-1e-8].sum()))
        self.cache[key] = result; self.queries += 1; self.patterns_evaluated += len(weights)
        return result

    def check(self, path, memory, diagnostics=False):
        rows = [self.local(t,path[max(0,t+1-memory):t+1],diagnostics) for t in range(len(path))]
        return sum(x['lower'] for x in rows), sum(x['upper'] for x in rows), rows
