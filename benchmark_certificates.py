"""Independent full-history and stronger-reference algorithms, revision 12.

The exact tree uses the full posterior at every history. Its constrained MDP
reference minimizes expected radio cost subject to the *sum of certified
marginal tails*, over randomized history policies on the stated finite catalog.
That is not optimization of the true union probability. The finite-family
PAC comparator instead bounds a Monte Carlo mission event with a simultaneous
Clopper--Pearson bound. These are explicitly different sufficient contracts.
"""
from dataclasses import dataclass
from itertools import product
import math
import numpy as np
from scipy.stats import beta
from memory_certificates import EYE, GUARD, _grid_dp, posterior, radial_bounds


@dataclass
class CheckedPlan:
    path: tuple | None
    objective: float
    certificate: float
    lower_objective: float
    status: str


def grid_plan(factors, catalogs, valid, alpha=.1, bins=4000, equal=False, objectives=None):
    """Safe wrapper, including negative utility and feasible zero radio cost."""
    costs = ([np.asarray([a.cost for a in group]) for group in catalogs]
             if objectives is None else objectives)
    low = _grid_dp(factors, costs, valid, alpha, bins, False, equal)
    high = _grid_dp(factors, costs, valid, alpha, bins, True, equal)
    if high is None:
        return CheckedPlan(None, np.nan, np.nan, low[1] if low else np.inf,
                           'rounding_unresolved' if low else 'certificate_infeasible')
    path, cost, _ = high
    risk = sum(float(f[path[max(0, t+1-f.ndim):t+1]]) for t, f in enumerate(factors))
    if risk > alpha+1e-11:
        raise ArithmeticError('Grid solution failed unrounded verification.')
    return CheckedPlan(path, float(cost), risk, float(low[1]), 'certified')


def enumerate_plans(factors, catalogs, valid, alpha):
    """Unrounded exhaustive action-sequence search; independent of grid DP."""
    best = None
    for path in product(*(np.flatnonzero(v) for v in valid)):
        risk = sum(float(f[path[max(0, t+1-f.ndim):t+1]]) for t, f in enumerate(factors))
        cost = sum(catalogs[t][a].cost for t, a in enumerate(path))
        if risk <= alpha and (best is None or (cost, path) < (best.objective, best.path)):
            best = CheckedPlan(tuple(path), cost, risk, np.nan, 'certified')
    return best or CheckedPlan(None, np.nan, np.nan, np.inf, 'certificate_infeasible')


def discrete_risk_redistribution(factors, catalogs, valid, alpha):
    """IRA-inspired finite-catalog slack redistribution, not continuous IRA.

    Start at the cheapest action under equal budgets. Repeatedly exchange
    unspent global risk for the largest single-stage cost decrease. Every
    accepted swap is rechecked. It can stop at a local optimum or fail its
    equal-budget initialization; no convex-IRA optimality claim applies.
    """
    if any(f.ndim != 1 for f in factors):
        raise ValueError('The discrete redistribution comparator uses w=1.')
    path = []
    for f, g, v in zip(factors, catalogs, valid):
        feasible = [a for a in np.flatnonzero(v) if f[a] <= alpha/len(factors)]
        if not feasible:
            return CheckedPlan(None, np.nan, np.nan, np.nan, 'equal_initialization_infeasible')
        path.append(min(feasible, key=lambda a: (g[a].cost, int(a))))
    while True:
        risk = sum(float(f[a]) for f, a in zip(factors, path))
        moves = []
        for t, (f, g, v) in enumerate(zip(factors, catalogs, valid)):
            for a in np.flatnonzero(v):
                saving = g[path[t]].cost-g[a].cost
                new_risk = risk-f[path[t]]+f[a]
                if saving > 1e-12 and new_risk <= alpha:
                    moves.append((-saving, new_risk, t, int(a)))
        if not moves:
            break
        _, _, t, a = min(moves); path[t] = a
    return CheckedPlan(tuple(path), sum(g[a].cost for g, a in zip(catalogs, path)),
                       sum(float(f[a]) for f, a in zip(factors, path)), np.nan, 'certified')


class FullHistoryTree:
    """Every action/reception prefix, without covariance truncation or pruning."""
    def __init__(self, oracle, max_nodes=2_000_000):
        self.oracle = oracle
        self.levels = [np.asarray(oracle.initial)[None]]
        self.edges = []; self.nodes = 1
        for t, group in enumerate(oracle.catalogs):
            parents = self.levels[-1]; count = len(parents)
            next_count = count*sum(len(a.probabilities) for a in group)
            if self.nodes+next_count > max_nodes:
                raise ValueError('Declared exact-tree node limit exceeded.')
            offspring, edges, offset = [], [], 0
            for a in group:
                B = len(a.probabilities)
                P = posterior(parents[:, None]+oracle.process[t]*EYE, a.information)
                D, U = radial_bounds(P, oracle.radius, oracle.terms)
                indices = offset+np.arange(count*B).reshape(count, B)
                edges.append((indices, a.probabilities, D, U, a.cost))
                offspring.append(P.reshape(-1, 2, 2)); offset += count*B
            self.levels.append(np.concatenate(offspring)); self.edges.append(edges)
            self.nodes += next_count

    def bellman(self, multiplier, upper=True):
        """Exact finite-tree Lagrangian solution and its expected cost/risk."""
        C = np.zeros(len(self.levels[-1])); R = C.copy()
        for edges in reversed(self.edges):
            candidate_C, candidate_R = [], []
            for indices, p, D, U, cost in edges:
                candidate_C.append(cost+C[indices]@p)
                candidate_R.append(((U if upper else D)+R[indices])@p)
            CC, RR = np.stack(candidate_C), np.stack(candidate_R)
            chosen = np.argmin(CC+multiplier*RR, axis=0)
            columns = np.arange(CC.shape[1]); C, R = CC[chosen, columns], RR[chosen, columns]
        return float(C[0]), float(R[0])

    def constrained(self, alpha=.1, upper=True, iterations=55):
        """Primal mixture and Lagrangian lower bound for randomized policies.

        Mixing occurs once at mission start between two deterministic policies.
        The returned dual/primal gap certifies expected-cost suboptimality for
        this finite-tree marginal-sum constraint, to numerical precision.
        """
        low_C, low_R = self.bellman(0., upper)
        if low_R <= alpha:
            return dict(status='certified', expected_cost=low_C, risk=low_R,
                        dual_lower=low_C, gap=0., multiplier=0., mixture=0.)
        left, right = 0., 1.
        high_C, high_R = self.bellman(right, upper)
        for _ in range(60):
            if high_R <= alpha: break
            right *= 2; high_C, high_R = self.bellman(right, upper)
        else:
            return dict(status='certificate_infeasible', expected_cost=np.nan,
                        risk=high_R, dual_lower=np.nan, gap=np.nan)
        dual = -np.inf
        for _ in range(iterations):
            lam = (left+right)/2; C, R = self.bellman(lam, upper)
            dual = max(dual, C+lam*(R-alpha))
            if R > alpha:
                left, low_C, low_R = lam, C, R
            else:
                right, high_C, high_R = lam, C, R
        fraction = np.clip((low_R-alpha)/(low_R-high_R), 0., 1.)
        cost = fraction*high_C+(1-fraction)*low_C
        risk = fraction*high_R+(1-fraction)*low_R
        return dict(status='certified', expected_cost=cost, risk=risk,
                    dual_lower=dual, gap=max(0., cost-dual),
                    multiplier=right, mixture=fraction)

    def all_open_loop(self):
        """Unrounded D/U marginal sums for every fixed action sequence."""
        codes = np.zeros(1, int); weights = np.ones(1)
        lower = np.zeros(1); upper = lower.copy(); costs = lower.copy()
        number = 1
        for edges in self.edges:
            A = len(edges); number *= A
            new_codes, new_weights, LD, UU = [], [], [], []
            for a, (indices, p, D, U, cost) in enumerate(edges):
                B = len(p)
                new_codes.append(np.repeat(codes*A+a, B))
                new_weights.append((weights[:, None]*p).ravel())
                LD.append(D.ravel()); UU.append(U.ravel())
            codes, weights = np.concatenate(new_codes), np.concatenate(new_weights)
            lower = np.repeat(lower, A)+np.bincount(codes, weights* np.concatenate(LD), minlength=number)
            upper = np.repeat(upper, A)+np.bincount(codes, weights* np.concatenate(UU), minlength=number)
            costs = np.repeat(costs, A)+np.tile([e[-1] for e in edges], number//A)
        output = []
        paths = product(*(range(len(g)) for g in self.oracle.catalogs))
        for code, path in enumerate(paths):
            output.append(dict(path=path, lower=float(lower[code]), upper=float(upper[code]), cost=float(costs[code])))
        return output


def clopper_upper(failures, samples, confidence_error):
    if not 0 < confidence_error < 1 or not 0 <= failures <= samples:
        raise ValueError('Valid count and confidence error required.')
    return 1. if failures == samples else float(beta.ppf(1-confidence_error, failures+1, samples-failures))


def finite_family_pac(candidates, failure_counts, samples, alpha=.1, delta=.01):
    """Simultaneous CP test over a fixed geometry-conditioned family of K plans.

    Candidates must be generated independently of these audit samples. Shared
    random numbers between candidates are allowed by the Bonferroni bound.
    """
    if len(candidates) != len(failure_counts) or not len(candidates):
        raise ValueError('A nonempty fixed candidate family and matching counts required.')
    accepted = []
    for candidate, count in zip(candidates, failure_counts):
        bound = clopper_upper(int(count), samples, delta/len(candidates))
        if bound <= alpha:
            accepted.append((candidate['cost'], tuple(candidate['path']), bound, int(count)))
    if not accepted:
        return dict(status='pac_rejected', path=None, cost=np.nan, bound=np.nan)
    cost, path, bound, count = min(accepted)
    return dict(status='pac_accepted', path=path, cost=cost, bound=bound, failures=count)
