"""Predeclared revision-10 paired experiments; no historical CSV is modified.

Run: python3 memory_experiments.py --workers 4
Main: seeds 1200001..1200200, 2048 simulations per admitted policy/layout.
All tables use these raw records; there are no manually entered outcome cells.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import time
import numpy as np
from scipy.stats import binomtest, t as student_t
from secure_drop_research import Config, generate_instance
from robust_experiments import route_for
from memory_certificates import (MemoryOracle, build_catalog, plan_grid, service_mask,
                                 simulate_plan)

ROOT = Path(__file__).resolve().parent
SOURCE = ['memory_certificates.py', 'memory_experiments.py', 'secure_drop_research.py',
          'wireless_extensions.py', 'test_memory_certificates.py']
METHODS = [('No-observation radial', 1, 'radial', False),
           ('Memory radial', 2, 'radial', False),
           ('Gaussian protection', 2, 'protection', False),
           ('Moment bound', 2, 'moment', False),
           ('Equal epoch radial', 2, 'radial', True)]


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    temporary.replace(path)


def problem(seed, anchors=6, csi=0., terms=128, sizes=(0, 1, 2, 3, 4, 5, 6), drift=.002, initial=.3):
    cfg = replace(Config(), anchors=anchors, pilot_control_s=40e-6,
                  drift_m2_per_m=drift, initial_sd_m=initial)
    inst = generate_instance(seed, cfg); route = route_for(inst, cfg)
    if route is None:
        return cfg, inst, None, None, None
    catalogs, gammas = zip(*(build_catalog(inst, node, cfg, csi_db=csi, support_sizes=sizes)
                            for node in route.events))
    oracle = MemoryOracle(catalogs, cfg.drift_m2_per_m*route.lengths,
                          np.eye(2)*cfg.initial_sd_m**2, cfg.radius_m, terms)
    return cfg, inst, route, oracle, gammas


def one_plan(seed, cfg, route, oracle, gammas, method, frames=4, packets=24, alpha=.1,
             case='main', repetitions=0, csi=0., actual_csi=0., bins=2000,
             memory=None, criterion=None, equal_epoch=False):
    definition = next((x for x in METHODS if x[0] == method), None)
    if definition:
        _, memory, criterion, equal_epoch = definition
    row = dict(seed=seed, case=case, method=method, frames=frames, packets=packets,
               alpha=alpha, anchors=cfg.anchors, csi_design_db=csi, csi_actual_db=actual_csi,
               radius=cfg.radius_m, drift=cfg.drift_m2_per_m, initial_sd=cfg.initial_sd_m,
               terms=oracle.terms if oracle else 128, admitted=0, status='route_infeasible')
    if route is None:
        return row
    start = time.perf_counter()
    mask, choices = service_mask(oracle.catalogs, gammas, route.events, cfg.stops, frames, packets, cfg)
    factor_start = time.perf_counter()
    factors = oracle.factors(memory, criterion)
    oracle_ms = (time.perf_counter()-factor_start)*1000
    dp_start = time.perf_counter()
    plan = plan_grid(factors, oracle.catalogs, mask, alpha, bins, equal_epoch)
    row.update(status=plan.status, horizon=len(route.events),
               planning_ms=1000*(time.perf_counter()-start),
               oracle_ms=oracle_ms, dp_ms=1000*(time.perf_counter()-dp_start),
               lower_cost=plan.lower_cost, rounding_gap=plan.gap,
               data_feasible=int(all(np.any(x) for x in mask)))
    if plan.path is None:
        return row
    selected = [oracle.catalogs[t][a] for t, a in enumerate(plan.path)]
    queue = [choices[t][a] for t, a in enumerate(plan.path)]
    deliveries = [t for t, node in enumerate(route.events) if 1 <= node <= cfg.stops]
    lower, check = oracle.check(plan.path, memory, criterion)
    lo1, noobs = oracle.check(plan.path, 1, 'radial')
    row.update(admitted=1, cost=plan.cost, certificate=check, marginal_lower=lower,
        same_plan_noobs=noobs, airtime_us=float(np.mean([a.airtime_s for a in selected])*1e6),
        pilot_fraction=float(np.mean([a.airtime_s for a in selected])/(frames*cfg.frame_s)),
        energy_mj=float(np.mean([a.energy_j for a in selected])*1000),
        queue_fail_bound=sum(queue[t].failure for t in deliveries),
        payload_kbit=sum(queue[t].expected_packets*512 for t in deliveries)/1000,
        attempts_data=float(np.mean([queue[t].attempts for t in deliveries])),
        path=';'.join(str(a) for a in plan.path),
        active_links=float(np.mean([len(a.support) for a in selected])))
    if repetitions:
        row.update(simulate_plan(oracle, plan.path, cfg, seed, repetitions,
                                 actual_csi_db=actual_csi, assumed_csi_db=csi))
    return row


def main_layout(arguments):
    seed, reps = arguments; rows = []
    cfg, inst, route, oracle, gammas = problem(seed)
    for method, _, _, _ in METHODS:
        rows.append(one_plan(seed, cfg, route, oracle, gammas, method, repetitions=reps))
    # Same layout/channel/route catalogs throughout these paired sensitivities.
    for frames, packets in [(1, 12), (1, 24), (2, 24), (4, 48), (8, 48)]:
        for method in ('No-observation radial', 'Memory radial'):
            rows.append(one_plan(seed, cfg, route, oracle, gammas, method, frames, packets,
                                 case='service'))
    for alpha in (.05, .01):
        for method in ('No-observation radial', 'Memory radial', 'Gaussian protection'):
            rows.append(one_plan(seed, cfg, route, oracle, gammas, method, alpha=alpha, case='risk',
                                 repetitions=reps if alpha == .01 else 0))
    # Fixed-geometry radius sweep: mutate only the risk-query radius, not channels.
    if oracle:
        for radius in (1., 3.):
            o = MemoryOracle(oracle.catalogs, oracle.process, oracle.initial, radius, oracle.terms)
            cc = replace(cfg, radius_m=radius)
            for method in ('No-observation radial', 'Memory radial'):
                rows.append(one_plan(seed, cc, route, o, gammas, method, case='radius'))
    return rows


def extra_layout(arguments):
    seed, reps = arguments; rows = []
    # Independent seeds for deployment pool, variance inflation, and depth studies.
    for anchors in (6, 12, 20, 50):
        cfg, inst, route, oracle, gammas = problem(seed, anchors=anchors)
        rows.append(one_plan(seed, cfg, route, oracle, gammas, 'Memory radial', case='pool'))
    for csi in (0., 2., 4.):
        cfg, inst, route, oracle, gammas = problem(seed, csi=csi)
        rows.append(one_plan(seed, cfg, route, oracle, gammas, 'Memory radial', case='csi',
                    repetitions=reps, csi=csi, actual_csi=2.))
    # Fixed A=10 catalog for all three depth policies; no expanding-superset claim.
    cfg, inst, route, oracle, gammas = problem(seed, sizes=(0, 1, 2, 3))
    for memory in (1, 2, 3):
        rows.append(one_plan(seed, cfg, route, oracle, gammas, 'Window '+str(memory),
             case='depth', memory=memory, criterion='radial'))
    for drift, initial in [(.001, .3), (.004, .3), (.002, .1), (.002, .6)]:
        cfg, inst, route, oracle, gammas = problem(seed, drift=drift, initial=initial)
        rows.append(one_plan(seed, cfg, route, oracle, gammas, 'Memory radial', case='odometry'))
    cfg, inst, route, oracle, gammas = problem(seed)
    if oracle:
        for terms in (32, 64, 512):
            o = MemoryOracle(oracle.catalogs, oracle.process, oracle.initial, cfg.radius_m, terms)
            rows.append(one_plan(seed, cfg, route, o, gammas, 'Memory radial', case='quadrature'))
    return rows


KEYS = ['case','method','frames','packets','alpha','anchors','csi_design_db','radius','drift','initial_sd','terms']
METRICS = ['cost','certificate','marginal_lower','same_plan_noobs','airtime_us','pilot_fraction',
 'energy_mj','queue_fail_bound','payload_kbit','planning_ms','oracle_ms','dp_ms','rounding_gap',
 'violation','marginal_sum','max_marginal','active_links']


def ci(values):
    v = np.array([x for x in values if np.isfinite(x)], float)
    if not len(v):
        return np.nan, np.nan
    return float(v.mean()), float(student_t.ppf(.995,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else np.nan


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in KEYS), []).append(row)
    output = []
    for key, group in groups.items():
        good = [x for x in group if x['admitted']]
        result = dict(zip(KEYS, key), layouts=len(group), admissions=len(good),
             data_feasible=sum(x.get('data_feasible',0) for x in group),
             unresolved=sum(x['status']=='rounding_unresolved' for x in group))
        for metric in METRICS:
            result[metric], result[metric+'_ci99'] = ci([float(x.get(metric,np.nan)) for x in good])
        # Ratios of cohort means; zero/rare empirical risks are not divided.
        if result['violation'] > 0:
            result['certificate_to_observed'] = result['certificate']/result['violation']
            result['union_overlap'] = result['marginal_sum']/result['violation']
        if result['marginal_sum'] > 0:
            result['memory_envelope_factor'] = result['certificate']/result['marginal_sum']
        output.append(result)
    return output


def paired(rows):
    main = [r for r in rows if r['case']=='main']
    by = {name:{r['seed']:r for r in main if r['method']==name} for name,_,_,_ in METHODS}
    output = []
    for other in ('No-observation radial','Gaussian protection','Moment bound','Equal epoch radial'):
        a,b = by['Memory radial'],by[other]
        seeds = sorted(set(a)&set(b)); common=[s for s in seeds if a[s]['admitted'] and b[s]['admitted']]
        wins = sum(a[s]['admitted']>b[s]['admitted'] for s in seeds)
        losses = sum(a[s]['admitted']<b[s]['admitted'] for s in seeds)
        row = dict(first='Memory radial',second=other,common=len(common),wins=wins,losses=losses,
             p_raw=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.)
        for m in ('cost','airtime_us','energy_mj','certificate','violation','payload_kbit'):
            row[m],row[m+'_ci99']=ci([a[s].get(m,np.nan)-b[s].get(m,np.nan) for s in common])
        output.append(row)
    order = sorted(range(len(output)),key=lambda j:output[j]['p_raw']); previous=0.
    for rank,j in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*output[j]['p_raw']))
        output[j]['p_holm']=previous
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--layouts',type=int,default=200);p.add_argument('--extras',type=int,default=24)
    p.add_argument('--repetitions',type=int,default=2048);p.add_argument('--output',default='results')
    args=p.parse_args(); out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    metadata=dict(revision=10,main_seeds=[1200001,1200000+args.layouts],
        extra_seeds=[1300001,1300000+args.extras],arguments=vars(args),
        main_contract=dict(frames=4,queued_packets_per_site=24,alpha=.1,radius_m=2.,data_mission_risk=.01),
        catalog=dict(support_sizes=[0,1,2,3,4,5,6],powers_w=[.02,.08,.32],generator='shared nested expected-information E-opt forward selection'),
        pilot_amendment='Minimum support of two removed before final runs: a prior-informed filter can certify zero/one-anchor actions. Pilot seeds 1000001..1000200 excluded; final cohorts use new seeds.',
        assumptions=['independent loss across epochs/anchors conditional on declared gain map',
          'centered affine Gaussian range errors independent of packet outcomes given gain map',
          'filter uses declared upper range variance in CSI design',
          'nonideal ACK and independent attempt completion proxy; not packet-level PHY validation'],
        sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCE},
        status='running',wall_seconds=None)
    (out/'v10_metadata.json').write_text(json.dumps(metadata,indent=2))
    start=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, group in enumerate(pool.map(main_layout,[(s,args.repetitions) for s in range(1200001,1200001+args.layouts)])):
            rows.extend(group)
            if (index+1)%10==0:
                write_csv(out/'v10_records.csv',rows); print('main',index+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
        for index,group in enumerate(pool.map(extra_layout,[(s,args.repetitions) for s in range(1300001,1300001+args.extras)])):
            rows.extend(group)
            if (index+1)%4==0:
                write_csv(out/'v10_records.csv',rows); print('extras',index+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
    write_csv(out/'v10_records.csv',rows);write_csv(out/'v10_summary.csv',summarize(rows));write_csv(out/'v10_paired.csv',paired(rows))
    metadata.update(status='complete',wall_seconds=time.perf_counter()-start,records=len(rows))
    (out/'v10_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('complete',len(rows),'seconds',round(metadata['wall_seconds'],1),flush=True)


if __name__ == '__main__':
    main()
