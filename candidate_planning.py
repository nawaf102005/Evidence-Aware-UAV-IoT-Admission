"""Probability-reusable, upper-only window checking and bounded candidate search.

The covariance coefficients are identical to PlanCheck.  Reusing them across
declared loss envelopes changes neither the verifier nor the feasible set.
This is engineering reuse for a controlled experiment, not a complexity claim.
"""
from dataclasses import dataclass
import numpy as np
from memory_certificates import EYE, GUARD, posterior, radial_bounds
from evidence_certificates import Polynomial
from benchmark_certificates import grid_plan

ALPHA = .1
MARGIN = 2e-11


class GeometryFactors:
    def __init__(self, oracle, registry, memory=6):
        self.o, self.reg, self.memory = oracle, registry, memory
        self.cache = {}; self.patterns = 0

    def polynomial(self, t, suffix):
        key = (int(t), tuple(suffix))
        if key in self.cache:
            return self.cache[key]
        s = t + 1 - len(suffix)
        cov = self.o.no_observation[s][None].copy(); ids = []
        for j, aidx in zip(range(s, t+1), suffix):
            a = self.o.catalogs[j][aidx]
            cov = posterior(cov[:, None] + self.o.process[j]*EYE,
                            a.information[None, ::-1]).reshape(-1, 2, 2)
            ids.extend(self.reg.ids[j][aidx])
        v = np.concatenate([radial_bounds(cov[k:k+2048], self.o.radius,
                                         self.o.terms)[1]
                            for k in range(0, len(cov), 2048)])
        self.cache[key] = Polynomial(ids, v)
        self.patterns += len(v)
        return self.cache[key]

    def window(self, t, suffix, law):
        return self.polynomial(t, suffix).evaluate(law) + GUARD

    def check(self, path, law, memory=None):
        w = self.memory if memory is None else memory
        return sum(self.window(t, path[max(0, t+1-w):t+1], law)
                   for t in range(len(path)))

    def factors2(self, law):
        A = len(self.o.catalogs[0]); out = []
        for t in range(len(self.o.catalogs)):
            if t == 0:
                out.append(np.array([self.window(t, (a,), law) for a in range(A)]))
            else:
                out.append(np.array([[self.window(t, (a, b), law) for b in range(A)]
                                     for a in range(A)]))
        return out


def cost(o, path):
    return float(sum(g[a].cost for g, a in zip(o.catalogs, path)))


def coordinate(factors, law, valid, initial, passes=3):
    if initial is None:
        return None
    o = factors.o; path = tuple(initial)
    if factors.check(path, law) > ALPHA-MARGIN:
        raise ValueError('Infeasible coordinate initializer')
    for _ in range(passes):
        changed = False
        for t, g in enumerate(o.catalogs):
            for a in sorted(map(int, np.flatnonzero(valid[t])),
                            key=lambda a: (g[a].cost, a)):
                if g[a].cost >= g[path[t]].cost-1e-12:
                    break
                trial = path[:t] + (a,) + path[t+1:]
                if factors.check(trial, law) <= ALPHA-MARGIN:
                    path = trial; changed = True; break
        if not changed:
            break
    return path


@dataclass
class Proposal:
    path: tuple | None
    queries: int = 0
    candidates: int = 0


def genetic(factors, law, valid, initial, seed, population=16,
            generations=20, factor_budget=768):
    if initial is None:
        return Proposal(None)
    o = factors.o; H = len(valid); w = factors.memory
    rng = np.random.default_rng(seed)
    choices = [tuple(map(int, np.flatnonzero(v))) for v in valid]
    queried = set(); cache = {}; best = None

    def evaluate(path):
        nonlocal best
        path = tuple(path)
        if path in cache:
            return cache[path]
        keys = {(t, path[max(0, t+1-w):t+1]) for t in range(H)}
        if len(queried | keys) > factor_budget:
            return None
        queried.update(keys)
        risk = factors.check(path, law); c = cost(o, path)
        rank = (0, c, risk, path) if risk <= ALPHA-MARGIN else (1, risk, c, path)
        cache[path] = rank
        if rank[0] == 0 and (best is None or rank < cache[best]):
            best = path
        return rank

    def mutate(parent):
        p = list(parent); loci = list(np.flatnonzero(rng.random(H) < 1/H))
        if rng.random() < .25:
            s = int(rng.integers(H)); loci += list(range(s, min(H, s+int(rng.integers(2, 5)))))
        if not loci:
            loci = [int(rng.integers(H))]
        for t in set(loci):
            alt = [a for a in choices[t] if a != p[t]]
            if alt:
                p[t] = int(rng.choice(alt))
        return tuple(p)

    initial = tuple(initial)
    if evaluate(initial) is None or best is None:
        return Proposal(None, len(queried), len(cache))
    pool = [initial]; exhausted = False
    for _ in range(population-1):
        p = mutate(initial)
        if evaluate(p) is None:
            exhausted = True; break
        pool.append(p)
    for _ in range(generations):
        if exhausted:
            break
        pool.sort(key=lambda p: cache[p]); new = pool[:2]
        while len(new) < population:
            def tournament():
                return min((pool[int(rng.integers(len(pool)))] for _ in range(3)),
                           key=lambda p: cache[p])
            a, b = tournament(), tournament()
            p = tuple(x if choose else y for x, y, choose in
                      zip(a, b, rng.random(H) < .5)) if rng.random() < .8 else a
            p = mutate(p)
            if evaluate(p) is None:
                exhausted = True; break
            new.append(p)
        if not exhausted:
            pool = new
    return Proposal(best, len(queried), len(cache))


def initialize(factors, law, valid, extra=()):
    dp = grid_plan(factors.factors2(law), factors.o.catalogs, valid, ALPHA, 2000)
    rich = tuple(len(g)-1 for g in factors.o.catalogs)
    candidates = [p for p in (dp.path, rich, *extra) if p is not None]
    candidates = [tuple(p) for p in candidates
                  if all(valid[t][a] for t, a in enumerate(p))
                  and factors.check(p, law) <= ALPHA-MARGIN]
    return (min(candidates, key=lambda p: (cost(factors.o, p), p))
            if candidates else None), dp.path


def lifetime(factors, path, upper, rate=.001, maximum=200):
    """Last valid reuse index; -1 explicitly denotes an empty valid set."""
    if path is None or factors.check(path, upper) > ALPHA-MARGIN:
        return -1
    lo, hi = 0, maximum
    while lo < hi:
        mid = (lo+hi+1)//2
        if factors.check(path, np.minimum(1, upper+rate*mid)) <= ALPHA-MARGIN:
            lo = mid
        else:
            hi = mid-1
    return lo
