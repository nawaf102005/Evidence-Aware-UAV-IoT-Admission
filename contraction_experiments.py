"""Revision-12 executed benchmarks, with independent seeds and frozen sources.

  python contraction_experiments.py --study all --workers 4

Small exact tree: 100 layouts x H={3,5}, M=2, A=3 (silent, two powers).
Long service: 500 layouts x H={5,10,20}, M=6, A=10, three route families.
IoT front: first 100 long layouts, H=10, alpha={.01,.05,.1,.2}, F={1,2,4}.
MC training is separate from evaluation; all failures/rejections are retained.
This is a simulation benchmark, not a field experiment.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import platform
import time
import numpy as np
from memory_certificates import (EYE, MemoryOracle, build_catalog, posterior,
                                  radial_bounds, queue_service)
from secure_drop_research import Config, generate_instance
from wireless_extensions import bler_normal_approx
from memory_experiments import write_csv
from benchmark_certificates import (FullHistoryTree, grid_plan, enumerate_plans,
    discrete_risk_redistribution, finite_family_pac)
from contraction_certificates import information_moment, memory_width_bound

ROOT = Path(__file__).resolve().parent
SOURCES = ['contraction_certificates.py', 'benchmark_certificates.py',
           'contraction_experiments.py', 'memory_certificates.py',
           'secure_drop_research.py', 'wireless_extensions.py']
LONG_SEEDS = range(1600001, 1600501)
SMALL_SEEDS = range(1700001, 1700101)
N = 2048


def flight_route(inst, cfg, events, family):
    """Fixed telemetry-collection order, no parcel mass; check flight/deadlines.

    The long-horizon application collects sensor data, without delivery payload.
    The complete route returns to depot; prefix studies certify their reported
    service horizon. No path or anchor placement is optimized here.
    """
    previous = 0; lengths = []; elapsed = energy = 0.; valid = True
    for node in events:
        d = float(np.linalg.norm(inst.nodes[node]-inst.nodes[previous])); lengths.append(d)
        energy += d*cfg.flight_j_per_m; elapsed += d/cfg.speed_m_s
        if node:
            elapsed += cfg.service_s
            valid &= elapsed <= inst.deadlines_s[node-1]
        previous = node
    valid &= energy <= cfg.capacity_j-cfg.reserve_j
    return SimpleNamespace(events=tuple(events), lengths=np.asarray(lengths),
        flight_j=energy, completion_s=elapsed, family=family, flight_feasible=bool(valid))


def long_problem(seed):
    cfg = replace(Config(), anchors=6, stops=19, area_m=600., pilot_control_s=40e-6)
    inst = generate_instance(seed, cfg)
    family = ('nearest', 'reverse-nearest', 'deadline')[(seed-LONG_SEEDS.start)%3]
    order = inst.orders[{'nearest': 0, 'reverse-nearest': 1, 'deadline': 2}[family]]
    route = flight_route(inst, cfg, list(order)+[0], family)
    catalogs, gammas = zip(*(build_catalog(inst, node, cfg, support_sizes=(0, 1, 2, 3)) for node in route.events))
    o = MemoryOracle(catalogs, cfg.drift_m2_per_m*route.lengths, cfg.initial_sd_m**2*EYE, cfg.radius_m, 128)
    return cfg, inst, route, o, gammas


def prefix(o, H):
    return MemoryOracle(o.catalogs[:H], o.process[:H], o.initial, o.radius, o.terms)


def data_choices(o, gammas, events, cfg, F=4, D=24, require_completion=True):
    deliveries = sum(node != 0 for node in events)
    choices = [[queue_service(len(a.support), gamma, F, D if node else 0, cfg) for a in group]
               for group, gamma, node in zip(o.catalogs, gammas, events)]
    valid = [np.array([v.failure <= .01/max(deliveries, 1) if require_completion else True for v in g])
             for g in choices]
    return valid, choices


def simulate(o, path, cfg, seed, repetitions=N, bias=0., persistent=False,
             nonlinear=False, inst=None, events=None):
    """Paired anchor-index KF errors; no radial integration on audit samples."""
    rng = np.random.default_rng(seed+120_000_000)
    H, M = len(path), cfg.anchors
    u = rng.random((H, repetitions, M)); z = rng.normal(size=(H, repetitions, M))
    d = rng.normal(size=(H, repetitions, 2)); e = rng.multivariate_normal(np.zeros(2), o.initial, repetitions)
    C = np.broadcast_to(o.initial, (repetitions, 2, 2)).copy()
    failed = np.zeros(repetitions, bool); marginal = 0.
    state_rng = np.random.default_rng(seed+121_000_000)
    bad = state_rng.random(repetitions) < .02
    for t, index in enumerate(path):
        a = o.catalogs[t][index]; q = o.process[t]
        C += q*EYE; e += np.sqrt(q)*d[t]
        if persistent and t:
            bad = state_rng.random(repetitions) < np.where(bad, .95, .02*.05/.98)
        for j, anchor in enumerate(a.support):
            loss = np.full(repetitions, a.loss[j])
            if persistent:
                # Same stationary per-link loss; cross-anchor and temporal coupling.
                loss = np.where(bad, 1., np.maximum((a.loss[j]-.02)/.98, 0.))
            keep = u[t, :, anchor] >= loss
            if not np.any(keep): continue
            h = a.H[j]; P = C[keep]; Ph = P@h
            gain = Ph/(np.einsum('ni,i->n', Ph, h)+a.variance[j])[:, None]
            T = EYE-gain[:, :, None]*h[None, None, :]
            innovation = e[keep]@h
            if nonlinear:
                delta = inst.nodes[events[t]]-inst.anchors[anchor]
                true_range = np.sqrt(delta@delta+cfg.altitude_m**2)
                innovation = true_range-np.sqrt(np.sum((delta-e[keep])**2, axis=1)+cfg.altitude_m**2)
            innovation += np.sqrt(a.variance[j])*z[t, keep, anchor]+bias
            e[keep] -= gain*innovation[:, None]
            C[keep] = T@P@T.transpose(0, 2, 1)+a.variance[j]*gain[:, :, None]*gain[:, None, :]
        miss = np.sum(e*e, axis=1) > o.radius**2
        failed |= miss; marginal += float(miss.mean())
    return dict(violation=float(failed.mean()), failures=int(failed.sum()),
                repetitions=repetitions, observed_marginal_sum=marginal)


def row_for(seed, case, H, method, plan, o, choices, route, cfg, simulate_now=True):
    row = dict(seed=seed, case=case, horizon=H, route_family=route.family,
        method=method, admitted=int(plan.path is not None and route.flight_feasible),
        status=plan.status if route.flight_feasible else 'flight_infeasible',
        flight_j=route.flight_j, objective=plan.objective,
        lower_objective=plan.lower_objective, bins=4000, alpha=.1)
    if not row['admitted']: return row
    path = plan.path; actions = [g[a] for g, a in zip(o.catalogs, path)]
    q = [g[a] for g, a in zip(choices, path)]
    row.update(certificate=plan.certificate, path=';'.join(map(str, path)),
        airtime_us=np.mean([a.airtime_s for a in actions])*1e6,
        energy_mj=np.mean([a.energy_j for a in actions])*1000,
        payload_kbit=sum(v.expected_packets*512 for v in q)/1000,
        queue_fail_bound=sum(v.failure for v in q))
    if simulate_now: row.update(simulate(o, path, cfg, seed))
    return row


def contraction_rows(seed, o, path):
    rows = []; rng = np.random.default_rng(seed+122_000_000); t = len(path)-1
    for w in (1, 2, 3, 4, 6):
        s = max(0, t+1-w)
        L = np.broadcast_to(o.initial if s == 0 else np.zeros((2, 2)), (512, 2, 2)).copy()
        U = np.broadcast_to(o.no_observation[s], L.shape).copy()
        eta = []
        for j in range(s, t+1):
            a = o.catalogs[j][path[j]]
            code = rng.choice(len(a.probabilities), 512, p=a.probabilities)
            L = posterior(L+o.process[j]*EYE, a.information[code])
            U = posterior(U+o.process[j]*EYE, a.information[code])
            if j > s: eta.append(information_moment(a, o.process[j]))
        D, UU = radial_bounds(L, o.radius, o.terms)[0], radial_bounds(U, o.radius, o.terms)[1]
        d1 = np.log1p(np.linalg.eigvalsh(o.no_observation[s]).max()/o.process[s]) if s else 0.
        bound = memory_width_bound(d1, eta, o.terms)
        rows.append(dict(seed=seed, horizon=len(path), window=w, sampled_width=float(np.mean(UU-D)),
                         moment_bound=bound, eta_product=float(np.prod(eta)), samples=512,
                         initial_log_bound=d1))
    return rows


def long_layout(seed):
    started = time.perf_counter(); cfg, inst, route, whole, gammas = long_problem(seed)
    f1 = whole.factors(1); f2 = whole.factors(2); fp = whole.factors(2, 'protection')
    factor_ms = (time.perf_counter()-started)*1000
    rows, widths, stress = [], [], []
    for H in (5, 10, 20):
        o = prefix(whole, H); events = route.events[:H]
        valid, choices = data_choices(o, gammas[:H], events, cfg)
        definitions = [('w1-equal', f1[:H], True), ('w1-joint', f1[:H], False),
                       ('w2-equal', f2[:H], True), ('w2-joint', f2[:H], False),
                       ('Gaussian-protection', fp[:H], False)]
        selected = None
        for method, factors, equal in definitions:
            start = time.perf_counter(); plan = grid_plan(factors, o.catalogs, valid, .1, 4000, equal)
            dp_ms = (time.perf_counter()-start)*1000
            row = row_for(seed, 'long', H, method, plan, o, choices, route, cfg)
            row['dp_ms'] = dp_ms
            row['factor_build_all20_ms'] = factor_ms
            rows.append(row)
            if method == 'w2-joint' and row['admitted']: selected = plan
        start = time.perf_counter()
        redistribution = discrete_risk_redistribution(f1[:H], o.catalogs, valid, .1)
        dp_ms = (time.perf_counter()-start)*1000
        row = row_for(seed, 'long', H, 'Discrete-IRA', redistribution, o, choices, route, cfg)
        row.update(dp_ms=dp_ms, factor_build_all20_ms=factor_ms); rows.append(row)
        if selected and H == 20:
            widths.extend(contraction_rows(seed, o, selected.path))
        if selected and H == 10 and seed < LONG_SEEDS.start+200:
            for name, bias, persistent, nonlinear in [('matched', 0., False, False),
                  ('nonlinear', 0., False, True), ('bias-0.25m', .25, False, True),
                  ('bias-1m', 1., False, True), ('persistent-same-marginals', 0., True, False)]:
                stress.append(dict(seed=seed, horizon=H, condition=name,
                    certificate=selected.certificate,
                    **simulate(o, selected.path, cfg, seed, N, bias, persistent, nonlinear, inst, events)))
    return rows, widths, stress


def pareto_layout(seed):
    cfg, inst, route, whole, gammas = long_problem(seed); H = 10; o = prefix(whole, H)
    f = o.factors(2); output = []
    for F in (1, 2, 4):
        valid, choices = data_choices(o, gammas[:H], route.events[:H], cfg, F, 96, False)
        utility = [np.asarray([-v.expected_packets+1e-7*a.cost for v, a in zip(g, cat)])
                   for g, cat in zip(choices, o.catalogs)]
        for alpha in (.01, .05, .1, .2):
            for method, objectives in [('radio-cost', None), ('payload-opt', utility)]:
                plan = grid_plan(f, o.catalogs, valid, alpha, 4000, objectives=objectives)
                row = row_for(seed, 'pareto', H, method, plan, o, choices, route, cfg, False)
                row.update(frames=F, packets=96, alpha=alpha, service_completion_required=0)
                if row['admitted']:
                    row['mean_payload_per_site_kbit'] = row['payload_kbit']/H
                    row['pilot_fraction'] = row['airtime_us']/(1000*F)
                output.append(row)
    return output


def small_problem(seed, H):
    cfg = replace(Config(), anchors=2, stops=H-1, area_m=600., pilot_control_s=40e-6)
    inst = generate_instance(seed, cfg)
    route = flight_route(inst, cfg, list(inst.orders[0])+[0], 'nearest')
    catalogs, gammas = zip(*(build_catalog(inst, node, cfg, support_sizes=(0, 2), powers=(.02, .32))
                            for node in route.events))
    o = MemoryOracle(catalogs, cfg.drift_m2_per_m*route.lengths, cfg.initial_sd_m**2*EYE, cfg.radius_m, 128)
    return cfg, inst, route, o, gammas


def small_layout(seed):
    output = []
    for H in (3, 5):
        cfg, inst, route, o, gammas = small_problem(seed, H)
        start = time.perf_counter(); tree = FullHistoryTree(o)
        candidates = tree.all_open_loop(); tree_ms = (time.perf_counter()-start)*1000
        all_valid, choices = data_choices(o, gammas, route.events, cfg, require_completion=False)
        # No queue-completion exclusion: this audit isolates verification/control.
        feasible = [r for r in candidates if r['upper'] <= .1]
        best = min(feasible, key=lambda r:(r['cost'], r['path'])) if feasible else None
        row = dict(seed=seed, case='exact', horizon=H, method='full-open-loop',
            admitted=int(best is not None), tree_nodes=tree.nodes, tree_ms=tree_ms,
            plans=len(candidates), cost=best['cost'] if best else np.nan,
            certificate=best['upper'] if best else np.nan,
            path=';'.join(map(str, best['path'])) if best else '')
        output.append(row)
        for w in (1, 2, 3):
            start = time.perf_counter(); factors = o.factors(min(w, H))
            exact = enumerate_plans(factors, o.catalogs, all_valid, .1)
            elapsed = (time.perf_counter()-start)*1000
            output.append(dict(seed=seed, case='exact', horizon=H, method=f'window-{w}',
                admitted=int(exact.path is not None), cost=exact.objective,
                certificate=exact.certificate, search_ms=elapsed,
                path=';'.join(map(str, exact.path)) if exact.path else '',
                full_lower=next((r['lower'] for r in candidates if r['path'] == exact.path), np.nan),
                full_upper=next((r['upper'] for r in candidates if r['path'] == exact.path), np.nan)))
        start = time.perf_counter(); cmdp = tree.constrained(.1, iterations=42)
        lower_cmdp = tree.constrained(.1, upper=False, iterations=42)
        output.append(dict(seed=seed, case='exact', horizon=H, method='full-causal-CMDP',
            admitted=int(cmdp['status'] == 'certified'), cost=cmdp['expected_cost'],
            certificate=cmdp['risk'], dual_lower=cmdp['dual_lower'], dual_gap=cmdp['gap'],
            quadrature_lower_cost=lower_cmdp['expected_cost'],
            search_ms=(time.perf_counter()-start)*1000, tree_nodes=tree.nodes,
            mixture=cmdp.get('mixture', np.nan)))
        # Simultaneous finite-family audit, samples independent of family creation.
        counts = [simulate(o, r['path'], cfg, seed+3_000_000, 1024)['failures'] for r in candidates]
        pac = finite_family_pac(candidates, counts, 1024, .1, .01)
        pacrow = dict(seed=seed, case='exact', horizon=H, method='finite-family-PAC',
            admitted=int(pac['path'] is not None), cost=pac['cost'], certificate=pac['bound'],
            training_repetitions=1024, family_size=len(candidates), confidence_error=.01)
        if pac['path']:
            pacrow.update(simulate(o, pac['path'], cfg, seed+4_000_000, 16384))
            pacrow['path'] = ';'.join(map(str, pac['path']))
        output.append(pacrow)
    return output


def contract():
    return dict(revision=12, long_seeds=[LONG_SEEDS.start, LONG_SEEDS.stop-1],
        small_seeds=[SMALL_SEEDS.start, SMALL_SEEDS.stop-1],
        repetitions=N, exact_horizons=[3, 5], long_horizons=[5, 10, 20],
        angular_terms=128, risk_bins=4000, risk=.1, radius=2.,
        long_area_m=600., long_route_families=['nearest', 'reverse-nearest', 'deadline'],
        long_zero_parcel_mass=True, exact_catalog_size=3, long_catalog_size=10,
        pareto_seeds=[LONG_SEEDS.start, LONG_SEEDS.start+99],
        stress_seeds=[LONG_SEEDS.start, LONG_SEEDS.start+199],
        source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
        disclosure='Simulated channels; no new field, hardware, or flight measurement.')


def run(study, workers, pilot=False):
    funcs = {'long': long_layout, 'small': small_layout, 'pareto': pareto_layout}
    seeds = list(SMALL_SEEDS if study == 'small' else LONG_SEEDS)
    if study == 'pareto': seeds = seeds[:100]
    if pilot: seeds = [x+1_000_000 for x in seeds[:2]]
    started = time.time(); rows = []; widths = []; stress = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(funcs[study], seeds), 1):
            if study == 'long':
                r, w, s = result; rows.extend(r); widths.extend(w); stress.extend(s)
            else: rows.extend(result)
            if i % 10 == 0 or i == len(seeds):
                prefix_name = 'v12_pilot_' if pilot else 'v12_'
                write_csv(ROOT/'results'/f'{prefix_name}{study}_records.csv', rows)
                if widths: write_csv(ROOT/'results'/f'{prefix_name}contraction_records.csv', widths)
                if stress: write_csv(ROOT/'results'/f'{prefix_name}stress_records.csv', stress)
                print(json.dumps(dict(study=study, layouts=i, total=len(seeds), rows=len(rows), seconds=round(time.time()-started, 1))), flush=True)
    meta = dict(contract(), study=study, layouts=len(seeds), records=len(rows), workers=workers,
                elapsed_s=time.time()-started, pilot=pilot, python=platform.python_version(),
                numpy=np.__version__, completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (ROOT/'results'/f'v12_{"pilot_" if pilot else ""}{study}_metadata.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--study', choices=['long', 'small', 'pareto', 'all'], default='all')
    p.add_argument('--workers', type=int, default=4); p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    (ROOT/'results'/'v12_contract.json').write_text(json.dumps(contract(), indent=2))
    for name in (['small', 'long', 'pareto'] if args.study == 'all' else [args.study]):
        run(name, args.workers, args.pilot)
