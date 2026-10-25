"""Selection-valid Bernoulli calibration for reception-window risk checks.

All streams and local dyadic looks are registered before observing probes.
The CP and Hoeffding constructions spend the same family/time error budget.
This module adds no model for unmeasured bias or unbounded deployment drift.
"""
from dataclasses import replace
from functools import cached_property
import math
import numpy as np
from scipy.stats import beta
from memory_certificates import MemoryOracle, EYE, GUARD, posterior, radial_bounds


def interval(k, n, error, method='cp'):
    if not (0 <= k <= n and n > 0 and 0 < error < 1):
        raise ValueError('Invalid binomial count or error allocation.')
    if method == 'cp':
        lo = 0. if k == 0 else beta.ppf(error/2, k, n-k+1)
        hi = 1. if k == n else beta.isf(error/2, k+1, n-k)
    elif method == 'hoeffding':
        r = math.sqrt(math.log(2/error)/(2*n))
        lo, hi = k/n-r, k/n+r
    else:
        raise ValueError(method)
    return max(0., float(lo)-8*GUARD), min(1., float(hi)+8*GUARD)


class Evidence:
    def __init__(self, streams, delta=.01, initial=32, method='cp'):
        if streams < 1 or initial < 1 or not 0 < delta < 1:
            raise ValueError('Invalid registry or confidence budget.')
        self.K, self.delta, self.initial, self.method = streams, delta, initial, method
        self.n = np.zeros(streams, int); self.k = self.n.copy()
        self.look = np.full(streams, -1, int)
        self.lower = np.zeros(streams); self.upper = np.ones(streams)

    @property
    def mean(self):
        return np.divide(self.k, self.n, out=np.zeros(self.K), where=self.n > 0)

    def update(self, i, successes_lost, samples):
        j = int(self.look[i]+1); target = self.initial*2**j
        if self.n[i]+samples != target or not 0 <= successes_lost <= samples:
            raise ValueError('Only the next registered local dyadic look is allowed.')
        self.n[i] = target; self.k[i] += successes_lost; self.look[i] = j
        d = self.delta/(self.K*(j+1)*(j+2))
        lo, hi = interval(int(self.k[i]), target, d, self.method)
        self.lower[i] = max(self.lower[i], lo)
        self.upper[i] = min(self.upper[i], hi)
        if self.lower[i] > self.upper[i]:
            raise ArithmeticError('Empty confidence intersection: withhold admission.')


class Registry:
    def __init__(self, oracle):
        self.keys = sorted({(t, int(m), float(a.power_w))
            for t,g in enumerate(oracle.catalogs) for a in g for m in a.support})
        lookup = {k:i for i,k in enumerate(self.keys)}
        self.ids = [[tuple(lookup[(t,int(m),float(a.power_w))] for m in a.support)
                     for a in g] for t,g in enumerate(oracle.catalogs)]

    def with_losses(self, oracle, losses):
        groups = []
        for t,g in enumerate(oracle.catalogs):
            actions = []
            for a, ids in zip(g, self.ids[t]):
                p = np.asarray(losses)[list(ids)]
                weights = np.prod(np.where(a.masks, 1-p, p), axis=1)
                actions.append(replace(a, loss=p.copy(), probabilities=weights))
            groups.append(actions)
        return MemoryOracle(groups, oracle.process, oracle.initial, oracle.radius, oracle.terms)


class Polynomial:
    """Multiaffine window expectation; each axis is 0=received, 1=lost."""
    def __init__(self, ids, values):
        self.ids = tuple(ids)
        self.values = np.asarray(values).reshape((2,)*len(ids))

    def evaluate(self, p):
        value = self.values
        for i in self.ids:
            value = np.tensordot(np.array([1-p[i], p[i]]), value, axes=(0,0))
        return float(value)

    def gradient(self, p):
        result = []
        for j,i in enumerate(self.ids):
            value = np.take(self.values,1,axis=j)-np.take(self.values,0,axis=j)
            for k in self.ids[:j]+self.ids[j+1:]:
                value = np.tensordot(np.array([1-p[k],p[k]]),value,axes=(0,0))
            result.append((i,max(0.,float(value))))
        return result


class PlanCheck:
    def __init__(self, oracle, registry, path, memory):
        self.path, self.memory, self.registry = tuple(path), memory, registry
        self.windows = []
        for t in range(len(path)):
            s = max(0,t+1-memory)
            P = oracle.no_observation[s][None].copy(); ids=[]
            for j in range(s,t+1):
                a = oracle.catalogs[j][path[j]]
                # RadioAction.masks lists received bits in ascending order.
                # Reverse the information array to obtain 0=receive,1=loss.
                P = posterior(P[:,None]+oracle.process[j]*EYE,
                              a.information[None,::-1]).reshape(-1,2,2)
                ids.extend(registry.ids[j][path[j]])
            upper = np.concatenate([radial_bounds(P[k:k+4096],oracle.radius,oracle.terms)[1]
                                    for k in range(0,len(P),4096)])
            self.windows.append(Polynomial(ids,upper))

    def evaluate(self, p):
        return sum(x.evaluate(p) for x in self.windows)+len(self.path)*GUARD

    def gradient(self, p):
        value = np.zeros(len(self.registry.keys))
        for x in self.windows:
            for i,d in x.gradient(p): value[i] += d
        return value

    @cached_property
    def influences(self):
        value = np.zeros(len(self.registry.keys))
        for x in self.windows:
            for j,i in enumerate(x.ids):
                d = np.take(x.values,1,axis=j)-np.take(x.values,0,axis=j)
                if d.min(initial=0.) < -1e-10:
                    raise ArithmeticError('Window tail is not increasing in lost links.')
                value[i] += max(0., float(d.max(initial=0.)))
        return value

    @cached_property
    def used(self):
        return np.flatnonzero(self.influences > 1e-15)


def sufficient_uniform_look(influences, margin, registered, delta=.01, initial=32):
    """CP/Hoeffding high-probability sufficient sample count for a fixed plan.

    On the simultaneous event, U_i-p_i <= 2 r_i. Includes dyadic error
    spending. Positive margin is essential; no admission promise otherwise.
    The returned count excludes compulsory initial probes of unused streams.
    """
    influences = np.asarray(influences, float)
    G = float(np.sum(influences)); used = int(np.count_nonzero(influences > 1e-15))
    if margin <= 0: return dict(look=-1, per_stream=math.inf, total=math.inf)
    for j in range(100):
        n = initial*2**j
        inflation = 2*G*math.sqrt(math.log(2*registered*(j+1)*(j+2)/delta)/(2*n))
        if inflation+8*GUARD*G+16*GUARD <= margin:
            return dict(look=j, per_stream=n, total=used*n,
                        inflation=inflation, sum_influence=G)
    raise ArithmeticError('Could not find finite sufficient look.')


def no_loss_barrier(threshold, registered, delta=.01, initial=32):
    """First local look at which even the best record k=0 can pass CP U<=tau."""
    if not 0 < threshold < 1: raise ValueError('Threshold must be strictly between zero and one.')
    for j in range(100):
        n=initial*2**j; error=delta/(registered*(j+1)*(j+2))
        best = -math.expm1(math.log(error/2)/n)
        if best+8*GUARD <= threshold: return n
    raise ArithmeticError('No finite zero-loss barrier found.')


def break_even(probe_seconds, reference_per_mission, planned_per_mission):
    saving = reference_per_mission-planned_per_mission
    return math.inf if saving <= 0 else max(1, math.ceil(probe_seconds/saving))
