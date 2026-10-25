"""Finite-catalog genetic candidate generation with exact risk verification.

Generic feasibility-first GA, not a reproduction of a DRL/ISAC paper. Search
randomness is independent of evaluation noise. Receptions are never sampled
in the search constraint. No global approximation or learning claim is made.
"""
from dataclasses import dataclass
import numpy as np
from information_certificates import LazyWindowOracle


@dataclass
class SearchResult:
    path: tuple | None
    cost: float
    certificate: float
    status: str
    generations: int
    unique_candidates: int
    failed_candidates: int
    factor_queries: int
    pattern_evaluations: int


def genetic_plan(o, valid, initial, memory=6, alpha=.1, seed=0,
                 population=32, generations=64, factor_budget=2048):
    """Warm-started elitist GA under an explicit unique-factor query budget.

    All genomes satisfy data masks. Feasible individuals dominate infeasible
    ones; feasible fitness is the common radio objective, infeasible fitness
    is the exact upper certificate. Elites retain the feasible initializer.
    Tournament size 3; uniform crossover probability .8; gene mutation 1/H;
    with probability .25 an additional block of 2--4 loci is mutated.
    The initializer is shared with the coordinate comparator, not its output.
    """
    if population < 4 or generations < 1 or factor_budget < 1:
        raise ValueError('Invalid search budget.')
    H=len(o.catalogs)
    if H==0 or len(valid)!=H or not (0<alpha<1):
        raise ValueError('Invalid decision problem.')
    choices=[tuple(map(int,np.flatnonzero(v))) for v in valid]
    if any(not c for c in choices):
        return SearchResult(None,np.inf,np.inf,'data_infeasible',0,0,0,0,0)
    if initial is None:
        return SearchResult(None,np.inf,np.inf,'missing_initializer',0,0,0,0,0)
    initial=tuple(map(int,initial))
    if len(initial)!=H or any(a not in c for a,c in zip(initial,choices)):
        raise ValueError('Initializer violates a queue mask.')
    rng=np.random.default_rng(seed);oracle=LazyWindowOracle(o)
    cache={};failed=0;best=None

    def evaluate(path):
        nonlocal failed,best
        path=tuple(map(int,path))
        if path in cache:return cache[path]
        required={(t,path[max(0,t+1-memory):t+1],False) for t in range(H)}
        new=sum(k not in oracle.cache for k in required)
        if oracle.queries+new>factor_budget:return None
        _,risk,_=oracle.check(path,memory)
        cost=float(sum(g[a].cost for g,a in zip(o.catalogs,path)))
        feasible=risk<=alpha-2e-11
        rank=(0,cost,risk) if feasible else (1,risk,cost)
        value=(rank,cost,risk);cache[path]=value
        if feasible and (best is None or rank<cache[best][0]):best=path
        if not feasible:failed+=1
        return value

    init=evaluate(initial)
    if init is None:
        return SearchResult(None,np.inf,np.inf,'initializer_exceeds_query_budget',0,0,0,0,0)
    if best is None:
        return SearchResult(None,np.inf,init[2],'initial_certificate_infeasible',0,1,1,oracle.queries,oracle.patterns_evaluated)

    def mutate(parent,force=False):
        x=list(parent)
        loci=list(np.flatnonzero(rng.random(H)<1/H))
        if rng.random()<.25:
            start=int(rng.integers(H));length=int(rng.integers(2,5))
            loci+=list(range(start,min(H,start+length)))
        if force and not loci:loci=[int(rng.integers(H))]
        for j in set(loci):
            alternatives=[a for a in choices[j] if a!=x[j]]
            if alternatives:x[j]=int(rng.choice(alternatives))
        return tuple(x)

    pool=[initial];exhausted=False
    for _ in range(population-1):
        child=mutate(initial,True)
        if evaluate(child) is None:exhausted=True;break
        pool.append(child)
    completed=0
    for generation in range(generations):
        if exhausted:break
        pool.sort(key=lambda p:cache[p][0]);new_pool=pool[:2]
        while len(new_pool)<population:
            def tournament():
                sampled=[pool[int(rng.integers(len(pool)))] for _ in range(3)]
                return min(sampled,key=lambda p:cache[p][0])
            a,b=tournament(),tournament()
            child=tuple(x if choose else y for x,y,choose in zip(a,b,rng.random(H)<.5)) if rng.random()<.8 else a
            child=mutate(child,True)
            if evaluate(child) is None:exhausted=True;break
            new_pool.append(child)
        if exhausted:break
        pool=new_pool;completed=generation+1
    _,cost,risk=cache[best]
    return SearchResult(best,cost,risk,'factor_budget' if exhausted else 'generation_limit',
        completed,len(cache),failed,oracle.queries,oracle.patterns_evaluated)
