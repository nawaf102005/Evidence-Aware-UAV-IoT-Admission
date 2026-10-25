"""Finite-memory covariance certificates and bounded risk-grid planning.

Revision 10: independent epoch loss laws, a fixed shared route, affine 2-D KF.
No fault-detection/RAIM implementation is claimed. The Gaussian radius law,
Riccati monotonicity, and risk allocation are established tools.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
import numpy as np
from scipy.stats import binom
from secure_drop_research import Config, inv2
from wireless_extensions import bler_normal_approx

EYE = np.eye(2)
SYMBOLS = np.array([320, 640, 1280, 2560, 5120])
GUARD = 2e-13


def posterior(covariance, information):
    """PSD-safe Riccati update; accepts a zero lower endpoint."""
    C, S = np.broadcast_arrays(np.asarray(covariance, float), np.asarray(information, float))
    value = inv2(EYE + C @ S) @ C
    return (value + np.swapaxes(value, -1, -2)) / 2


@lru_cache(maxsize=12)
def _cosines(terms):
    return np.cos(np.linspace(0, np.pi / 2, terms + 1)) ** 2


def radial_bounds(covariance, radius, terms=128):
    """Ordered angular sums for the known centered 2-D Gaussian/Hoyt law.

    Also valid for singular PSD covariances by continuity. The returned
    bracket includes a floating-point safety allowance, not interval arithmetic.
    """
    if terms < 2 or radius <= 0:
        raise ValueError('Positive radius and at least two angular cells required.')
    P = np.asarray(covariance, float)
    tr = P[..., 0, 0] + P[..., 1, 1]
    disc = np.hypot(P[..., 0, 0] - P[..., 1, 1], 2 * P[..., 0, 1])
    lo, hi = (tr - disc) / 2, (tr + disc) / 2
    if np.any(lo < -1e-9) or np.any(~np.isfinite(P)):
        raise ValueError('Covariance must be finite and positive semidefinite.')
    v = np.maximum(lo, 0)[..., None] + disc[..., None] * _cosines(terms)
    exponent = np.full_like(v, -np.inf)
    np.divide(-radius ** 2 / 2, v, out=exponent, where=v > 0)
    values = np.exp(exponent)
    lower = values[..., 1:].mean(axis=-1)
    upper = values[..., :-1].mean(axis=-1)
    return np.maximum(0., lower - GUARD), np.minimum(1., upper + GUARD)


def scalar_risk_bound(P, radius, criterion):
    tr = np.trace(P, axis1=-2, axis2=-1)
    if criterion == 'moment':
        return np.minimum(1., tr / radius ** 2)
    if criterion == 'protection':
        disc = np.hypot(P[..., 0, 0]-P[..., 1, 1], 2*P[..., 0, 1])
        maximum = np.maximum((tr+disc)/2, 1e-300)
        return np.exp(-radius**2/(2*maximum))
    raise ValueError(criterion)


@dataclass
class RadioAction:
    support: tuple
    power_w: float
    H: np.ndarray
    variance: np.ndarray
    loss: np.ndarray
    information: np.ndarray
    probabilities: np.ndarray
    masks: np.ndarray
    cost: float
    energy_j: float
    airtime_s: float


@dataclass
class ServiceChoice:
    symbols: int
    attempts: int
    success: float
    failure: float
    expected_packets: float


def queue_service(active, gamma, frames, packets, cfg):
    """ACK-confirmed finite-queue service, including each frame's overhead.

    A lost ACK consumes an attempt. Independent attempts are a declared model,
    not a measured protocol. Data carries 512 useful + 96 header bits.
    """
    if frames < 1 or packets < 0:
        raise ValueError('At least one service frame and a nonnegative queue required.')
    if packets == 0:
        return ServiceChoice(320, 0, 1., 0., 0.)
    first = max(cfg.frame_s-active*cfg.pilot_s-40e-6, 0.)
    rest = max(cfg.frame_s-40e-6, 0.)
    attempts = (np.floor((first+1e-15)/(SYMBOLS/cfg.bandwidth_hz+10e-6))
                +(frames-1)*np.floor((rest+1e-15)/(SYMBOLS/cfg.bandwidth_hz+10e-6))).astype(int)
    ack_success = 1-float(bler_normal_approx(gamma, int(round(10e-6*cfg.bandwidth_hz)), 48))
    success = (1-bler_normal_approx(gamma, SYMBOLS, 608))*ack_success
    failure = binom.cdf(packets-1, attempts, success)
    # E[min(D, Bin(N,s))] from its survival probabilities; avoids truncation bias.
    expected = np.sum(binom.sf(np.arange(1, packets+1)[:, None]-1, attempts, success), axis=0)
    index = min(range(len(SYMBOLS)), key=lambda i:(failure[i], -expected[i], SYMBOLS[i]))
    return ServiceChoice(int(SYMBOLS[index]), int(attempts[index]), float(success[index]),
                         float(failure[index]), float(expected[index]))


def build_catalog(instance, node, cfg, csi_db=0., support_sizes=(0, 1, 2, 3, 4, 5, 6), powers=(.02, .08, .32)):
    """Shared nested E-opt forward-selection catalog, not an exhaustive support search.

    All tested verifiers/planners receive exactly these candidates.
    """
    delta = instance.nodes[node] - instance.anchors
    ranges = np.sqrt(np.sum(delta**2, axis=1)+cfg.altitude_m**2)
    H = delta/ranges[:, None]
    lower_gain = instance.gains[node]*10**(-csi_db/10)
    gamma_per_w = lower_gain/(cfg.noise_w+instance.interference[node])
    beta = cfg.bandwidth_hz/np.sqrt(12.)
    xi = cfg.ranging_efficiency*299792458.**2/(8*np.pi**2*beta**2*cfg.bandwidth_hz*(cfg.pilot_s-cfg.pilot_control_s))
    ref_var = cfg.range_floor_m**2+xi/(.08*gamma_per_w)
    ref_loss = .02+.98*bler_normal_approx(.08*gamma_per_w, int(round(cfg.bandwidth_hz*cfg.pilot_control_s)), 48)
    atoms = ((1-ref_loss)/ref_var)[:, None, None]*H[:, :, None]*H[:, None, :]
    selected, subsets, total = [], ({0: ()} if 0 in support_sizes else {}), np.zeros((2, 2))
    for n in range(1, min(max(support_sizes), cfg.anchors)+1):
        candidates = [j for j in range(cfg.anchors) if j not in selected]
        j = max(candidates, key=lambda k:(round(float(np.linalg.eigvalsh(total+atoms[k])[0]), 12), np.trace(total+atoms[k]), -k))
        selected.append(j); total += atoms[j]
        if n in support_sizes:
            subsets[n] = tuple(sorted(selected))
    actions = []
    for n, support in subsets.items():
        h = H[list(support)]
        masks = np.array(list(product((0, 1), repeat=n)), int)  # 1 = received
        for power in (powers if n else (0.,)):
            gamma = power*gamma_per_w[list(support)]
            variance = cfg.range_floor_m**2+xi/gamma
            loss = .02+.98*bler_normal_approx(gamma, int(round(cfg.bandwidth_hz*cfg.pilot_control_s)), 48)
            probabilities = np.prod(np.where(masks, 1-loss, loss), axis=1)
            information = np.einsum('km,mij->kij', masks,
                              h[:, :, None]*h[:, None, :]/variance[:, None, None])
            energy = cfg.pilot_s*n*power
            actions.append(RadioAction(support, power, h, variance, loss, information,
                               probabilities, masks, n+cfg.pilot_energy_price_per_j*energy,
                               energy, n*cfg.pilot_s))
    data_gamma = cfg.data_power_w*lower_gain.max()/(cfg.noise_w+instance.interference[node])
    return actions, float(data_gamma)


class MemoryOracle:
    """Local finite-memory risk factors, including upper/lower endpoint bounds."""
    def __init__(self, catalogs, process, initial, radius, terms=128, max_histories=None):
        self.catalogs, self.process = catalogs, np.asarray(process, float)
        self.initial, self.radius, self.terms = np.asarray(initial), radius, terms
        self.max_histories = max_histories
        self.no_observation = [self.initial+sum(self.process[:s])*EYE for s in range(len(process)+1)]
        self.cache = {}

    def local(self, t, action_ids, criterion='radial'):
        key = (t, tuple(action_ids), criterion)
        if key in self.cache:
            return self.cache[key]
        s = t+1-len(action_ids)
        if s < 0:
            raise ValueError('Window extends before the mission.')
        upper = self.no_observation[s][None].copy()
        lower = (self.initial if s == 0 else np.zeros((2, 2)))[None].copy()
        weights, omitted = np.ones(1), 0.
        for epoch, index in zip(range(s, t+1), action_ids):
            action = self.catalogs[epoch][index]
            upper = posterior(upper[:, None]+self.process[epoch]*EYE, action.information[None]).reshape(-1, 2, 2)
            lower = posterior(lower[:, None]+self.process[epoch]*EYE, action.information[None]).reshape(-1, 2, 2)
            weights = (weights[:, None]*action.probabilities[None]).ravel()
            if self.max_histories and len(weights) > self.max_histories:
                keep = np.argsort(weights, kind='stable')[-self.max_histories:]
                omitted += float(weights.sum()-weights[keep].sum())
                weights, upper, lower = weights[keep], upper[keep], lower[keep]
        D, _ = radial_bounds(lower, self.radius, self.terms)
        for name in ('radial', 'protection', 'moment'):
            values = radial_bounds(upper, self.radius, self.terms)[1] if name == 'radial' else scalar_risk_bound(upper, self.radius, name)
            self.cache[(t, tuple(action_ids), name)] = (float(max(0., weights@D-GUARD)),
                                           float(min(1., weights@values+omitted+GUARD)), omitted)
        return self.cache[key]

    def factors(self, memory, criterion='radial'):
        result = []
        for t in range(len(self.process)):
            indices = range(max(0, t+1-memory), t+1)
            shape = tuple(len(self.catalogs[j]) for j in indices)
            values = np.empty(shape)
            for ids in product(*(range(x) for x in shape)):
                values[ids] = self.local(t, ids, criterion)[1]
            result.append(values)
        return result

    def check(self, path, memory, criterion='radial'):
        values = [self.local(t, path[max(0, t+1-memory):t+1], criterion) for t in range(len(path))]
        return sum(v[0] for v in values), sum(v[1] for v in values)


@dataclass
class GridPlan:
    path: tuple | None
    cost: float
    lower_cost: float
    bound: float
    units: int
    status: str
    gap: float


def _grid_dp(factors, costs, valid, alpha, bins, upward=True, equal_epoch=False):
    """Shortest path over (last w-1 actions, risk units)."""
    delta = alpha/bins
    # Sparse labels with Pareto pruning by ascending risk after every epoch.
    labels = {((), 0):(0., ())}
    for t, factor in enumerate(factors):
        future = {}
        memory = max(x.ndim for x in factors)
        for (history, units), (cost, path) in labels.items():
            for action in np.flatnonzero(valid[t]):
                ids = (history+(int(action),))[-factor.ndim:]
                risk = float(factor[ids])
                if equal_epoch and risk > alpha/len(factors):
                    continue
                increment = int(math.ceil(risk/delta)) if upward else int(math.floor(risk/delta))
                total = units+increment
                if total > bins:
                    continue
                tail = (history+(int(action),))[-(memory-1):] if memory > 1 else ()
                key = (tail, total)
                value = (cost+costs[t][action], path+(int(action),))
                if key not in future or value[0] < future[key][0]:
                    future[key] = value
        labels = {}
        grouped = {}
        for (tail, units), value in future.items():
            grouped.setdefault(tail, []).append((units, value))
        for tail, group in grouped.items():
            best = np.inf
            for units, value in sorted(group):
                if value[0] < best-1e-12:
                    labels[(tail, units)] = value
                    best = value[0]
        if not labels:
            return None
    key, value = min(labels.items(), key=lambda item:(item[1][0], item[0][1], item[1][1]))
    return value[1], value[0], key[1]


def plan_grid(factors, catalogs, valid, alpha, bins=2000, equal_epoch=False):
    if not 0 < alpha < 1 or bins < 1:
        raise ValueError('Risk alpha in (0,1) and a positive integer grid required.')
    costs = [np.array([a.cost for a in group]) for group in catalogs]
    lower = _grid_dp(factors, costs, valid, alpha, bins, False, equal_epoch)
    upper = _grid_dp(factors, costs, valid, alpha, bins, True, equal_epoch)
    if upper is None:
        return GridPlan(None, np.nan, lower[1] if lower else np.inf, np.nan, 0,
                      'rounding_unresolved' if lower else 'certificate_infeasible', np.nan)
    path, cost, units = upper
    bound = sum(float(f[path[max(0, t+1-f.ndim):t+1]]) for t, f in enumerate(factors))
    if bound > alpha+GUARD:
        raise ArithmeticError('A rounded solution failed unrounded verification.')
    lower_cost = lower[1]
    return GridPlan(path, cost, lower_cost, bound, units, 'certified', max(0., (cost-lower_cost)/cost))


def service_mask(catalogs, gammas, events, stops, frames, packets, cfg, data_risk=.01):
    choices, masks = [], []
    for actions, gamma, node in zip(catalogs, gammas, events):
        options = [queue_service(len(a.support), gamma, frames, packets if 1 <= node <= stops else 0, cfg) for a in actions]
        choices.append(options)
        masks.append(np.array([v.failure <= data_risk/stops for v in options]))
    return masks, choices


def simulate_plan(oracle, path, cfg, seed, repetitions=2048, actual_csi_db=0., assumed_csi_db=0.):
    """Shared anchor-index randomness, actual error dynamics, Joseph covariance.

    actual_csi_db is attenuation relative to the nominal map. The assumed
    filter R is the action's conservative variance. Modeled covariance need
    not equal actual covariance; both are propagated explicitly.
    """
    rng = np.random.default_rng(seed+70_000_000)
    H = len(path); M = cfg.anchors; N = repetitions
    uniforms = rng.random((H, N, M))
    measurements = rng.normal(size=(H, N, M))
    process_noise = rng.normal(size=(H, N, 2))
    error = rng.multivariate_normal(np.zeros(2), oracle.initial, N)
    model = np.broadcast_to(oracle.initial, (N, 2, 2)).copy()
    actual = model.copy()
    failures = np.zeros(N, bool); sum_marginal = 0.; marginals = []
    for t, index in enumerate(path):
        action = oracle.catalogs[t][index]
        q = oracle.process[t]
        error += np.sqrt(q)*process_noise[t]
        model += q*EYE; actual += q*EYE
        relative = 10**((actual_csi_db-assumed_csi_db)/10)
        variance = cfg.range_floor_m**2+(action.variance-cfg.range_floor_m**2)*relative
        # Recover gamma from modeled R; use the same declared control-loss proxy.
        beta = cfg.bandwidth_hz/np.sqrt(12.)
        xi = cfg.ranging_efficiency*299792458.**2/(8*np.pi**2*beta**2*cfg.bandwidth_hz*(cfg.pilot_s-cfg.pilot_control_s))
        true_gamma = xi/(variance-cfg.range_floor_m**2)
        loss = .02+.98*bler_normal_approx(true_gamma, int(round(cfg.bandwidth_hz*cfg.pilot_control_s)), 48)
        for j, anchor in enumerate(action.support):
            keep = uniforms[t, :, anchor] >= loss[j]
            if not np.any(keep):
                continue
            h = action.H[j]
            C = model[keep]; A = actual[keep]
            Ch = C@h
            denominator = np.einsum('ni,i->n', Ch, h)+action.variance[j]
            K = Ch/denominator[:, None]
            T = EYE-K[:, :, None]*h[None, None, :]
            noise = np.sqrt(variance[j])*measurements[t, keep, anchor]
            error[keep] = np.einsum('nij,nj->ni', T, error[keep])+K*noise[:, None]
            model[keep] = T@C@np.swapaxes(T, -1, -2)+action.variance[j]*K[:, :, None]*K[:, None, :]
            actual[keep] = T@A@np.swapaxes(T, -1, -2)+variance[j]*K[:, :, None]*K[:, None, :]
        failed = np.sum(error**2, axis=1) > cfg.radius_m**2
        failures |= failed
        lower, upper = radial_bounds(actual, cfg.radius_m, 512)
        marginal = float(((lower+upper)/2).mean())
        marginals.append(marginal); sum_marginal += marginal
    return dict(violation=float(failures.mean()), failures=int(failures.sum()),
                repetitions=N, marginal_sum=sum_marginal,
                max_marginal=max(marginals), model_domination_min=float(np.linalg.eigvalsh(model-actual).min()))
