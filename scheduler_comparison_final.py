"""R22 final learning/optimization comparison after retained surrogate pilot; all failures retained.

Training 256 geometries 6100001+, validation48 6200001+, test180 6400001+.
Three disjoint search/training seeds. Primary held-out scenarios are fixed
before execution. Fixed512 registered probes give identical inputs/checks.
All methods use 7 geometry-only codewords (0/1/2 links, 3 powers), not routes
or supports optimized using a true packet law. No test-based tuning is allowed.
"""
from pathlib import Path
from dataclasses import replace
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,platform,time
import numpy as np
from scipy.stats import beta
from secure_drop_research import Config,generate_instance
from contraction_experiments import flight_route,data_choices,simulate
from wireless_extensions import bler_normal_approx
from memory_certificates import EYE,RadioAction,MemoryOracle
from evidence_certificates import Registry,PlanCheck,interval
from benchmark_certificates import grid_plan
from information_certificates import LazyWindowOracle
from genetic_search import genetic_plan
from certified_learning import Surrogate,Network,train_ppo,train_ddqn,propose
from memory_experiments import write_csv

ROOT=Path(__file__).resolve().parent
SCENARIOS=((4,10,.03),(6,20,.10),(10,30,.25))
SEEDS=(701,702,703);PROBES=512;ALPHA=.1;DELTA=.01
SOURCES=('scheduler_comparison_final.py','scheduler_comparison.py','certified_learning.py','learning_schedulers.py','evidence_certificates.py',
         'memory_certificates.py','benchmark_certificates.py','genetic_search.py')


def problem(seed,M=6,H=20,gate_max=.1):
    cfg=replace(Config(),anchors=M,stops=H-1,area_m=600.,pilot_control_s=40e-6)
    inst=generate_instance(seed,cfg);family=('nearest','reverse-nearest','deadline')[seed%3]
    order=inst.orders[{'nearest':0,'reverse-nearest':1,'deadline':2}[family]]
    route=flight_route(inst,cfg,list(order)+[0],family)
    groups=[];gammas=[];base={}
    rng=np.random.default_rng(np.random.SeedSequence([seed,622]))
    gates=rng.uniform(0,gate_max,size=(H,M))
    for t,node in enumerate(route.events):
        delta=inst.nodes[node]-inst.anchors
        rr=np.sqrt(np.sum(delta**2,axis=1)+cfg.altitude_m**2);h=delta/rr[:,None]
        per=inst.gains[node]/(cfg.noise_w+inst.interference[node])
        bw=cfg.bandwidth_hz/np.sqrt(12.)
        xi=cfg.ranging_efficiency*299792458.**2/(8*np.pi**2*bw**2*cfg.bandwidth_hz*(cfg.pilot_s-cfg.pilot_control_s))
        v=cfg.range_floor_m**2+xi/(.08*per);atoms=h[:,:,None]*h[:,None,:]/v[:,None,None]
        total=np.zeros((2,2));selected=[];supports=[()]
        for _ in range(2):
            m=max((i for i in range(M) if i not in selected),
                  key=lambda i:(round(np.linalg.eigvalsh(total+atoms[i])[0],12),np.trace(total+atoms[i]),-i))
            selected.append(m);total+=atoms[m];supports.append(tuple(sorted(selected)))
        group=[]
        for support in supports:
            masks=np.array(list(product((0,1),repeat=len(support))),int)
            for power in ((.02,.08,.32) if support else (0.,)):
                hh=h[list(support)];v=cfg.range_floor_m**2+xi/(power*per[list(support)])
                info=np.einsum('km,mij->kij',masks,hh[:,:,None]*hh[:,None,:]/v[:,None,None])
                losses=[]
                for m in support:
                    b=.02+.98*float(bler_normal_approx(power*per[m],800,48))
                    base[t,m,power]=b;losses.append(b+(1-b)*gates[t,m])
                losses=np.array(losses)
                weights=np.prod(np.where(masks,1-losses,losses),axis=1)
                energy=len(support)*cfg.pilot_s*power
                group.append(RadioAction(support,power,hh,v,losses,info,weights,masks,
                       len(support)+1000*energy,energy,len(support)*cfg.pilot_s))
        groups.append(group);gammas.append(cfg.data_power_w*per.max())
    o=MemoryOracle(groups,cfg.drift_m2_per_m*route.lengths,cfg.initial_sd_m**2*EYE,cfg.radius_m,128)
    reg=Registry(o);truth=np.array([o.catalogs[t][-1].loss[0] for t in []])
    truth=np.array([base[t,m,p]+(1-base[t,m,p])*gates[t,m] for t,m,p in reg.keys])
    cap=np.array([base[t,m,p]+(1-base[t,m,p])*gate_max for t,m,p in reg.keys])
    k=np.array([np.random.default_rng(np.random.SeedSequence([seed,623,i])).binomial(PROBES,p) for i,p in enumerate(truth)])
    mean=k/PROBES
    upper=np.array([interval(int(z),PROBES,DELTA/len(k))[1] for z in k])
    empirical=reg.with_losses(o,mean);valid,_=data_choices(o,gammas,route.events,cfg)
    return cfg,inst,route,o,gammas,reg,truth,cap,k,upper,empirical,valid


def prepare_one(job):
    seed,scenario=job;p=problem(seed,*SCENARIOS[scenario])
    cfg,inst,route,o,gamma,reg,truth,cap,k,upper,empirical,valid=p
    return (seed,scenario,reg.with_losses(o,np.minimum(upper,cap)),valid,route.flight_feasible)


def coordinate(o,valid,initial,w=6,passes=3):
    if initial is None:return None,dict(queries=0,patterns=0)
    lo=LazyWindowOracle(o);path=list(initial);_,risk,_=lo.check(path,w)
    if risk>ALPHA-2e-11:return None,dict(queries=lo.queries,patterns=lo.patterns_evaluated)
    for _ in range(passes):
        changed=False
        for t,g in enumerate(o.catalogs):
            options=sorted(np.flatnonzero(valid[t]),key=lambda a:(g[a].cost,int(a)))
            for a in options:
                if g[a].cost>=g[path[t]].cost-1e-12:break
                q=path.copy();q[t]=int(a);_,r,_=lo.check(q,w)
                if r<=ALPHA-2e-11:path=q;changed=True;break
        if not changed:break
    return tuple(path),dict(queries=lo.queries,patterns=lo.patterns_evaluated)


def evaluate_one(job):
    seed,scenario,reps=job;tic=time.perf_counter()
    cfg,inst,route,o,gamma,reg,truth,cap,k,upper,empirical,valid=problem(seed,*SCENARIOS[scenario])
    empirical=reg.with_losses(o,np.minimum(upper,cap))
    rows=[];cache={};K=len(reg.keys)
    common=dict(seed=seed,scenario=scenario,anchors=cfg.anchors,H=len(route.events),
                gate_max=SCENARIOS[scenario][2],registered=K,calibration_probes=K*PROBES,
                true_p_min=float(truth.min()),true_p_max=float(truth.max()),
                initial_counts=';'.join(map(str,k)),repetitions=reps)
    def record(method,s,path,elapsed,probes=K*PROBES,extra=None):
        row=dict(common,method=method,search_seed=s,search_ms=elapsed,
                 calibration_s=probes*cfg.pilot_s,probes=probes,
                 proposed=int(path is not None),admitted=0,status='no_proposal')
        if extra:row.update(extra)
        if path is not None:
            path=tuple(path)
            if path not in cache:
                checker=PlanCheck(o,reg,path,6)
                # Audit all proposed plans, including those refused by CP.
                stats=simulate(o,path,cfg,seed,reps)
                cache[path]=(checker,stats)
            checker,stats=cache[path]
            u=checker.evaluate(cap if method=='fixed-envelope' else np.minimum(upper,cap))
            nominal=checker.evaluate(k/PROBES);actual=checker.evaluate(truth)
            row.update(path=';'.join(map(str,path)),certificate=u,nominal_certificate=nominal,
                       true_certificate=actual,nominal_admitted=int(nominal<=ALPHA-2e-11),
                       admitted=int(u<=ALPHA-2e-11),status='certified' if u<=ALPHA-2e-11 else 'verifier_rejected',
                       objective=sum(g[a].cost for g,a in zip(o.catalogs,path)),
                       ranging_us=np.mean([g[a].airtime_s for g,a in zip(o.catalogs,path)])*1e6,**stats)
        rows.append(row)
    if not route.flight_feasible or any(not np.any(v) for v in valid):
        for m in ('DP-w2','coordinate-w6','GA-w6','PPO-L','DDQN-L','fixed-envelope'):
            for s in (SEEDS if m in ('GA-w6','PPO-L','DDQN-L') else (0,)):
                record(m,s,None,0,probes=0);rows[-1]['status']='flight_or_data_infeasible'
        return rows,time.perf_counter()-tic
    # Empirical initial plan shared by coordinate and genetic search.
    start=time.perf_counter();factors=empirical.factors(2)
    dp=grid_plan(factors,empirical.catalogs,valid,ALPHA,2000)
    init=dp.path
    record('DP-w2',0,init,(time.perf_counter()-start)*1000)
    start=time.perf_counter();cp,extra=coordinate(empirical,valid,init)
    record('coordinate-w6',0,cp,(time.perf_counter()-start)*1000,extra=extra)
    for s in SEEDS:
        start=time.perf_counter()
        result=genetic_plan(empirical,valid,init,memory=6,seed=seed+s,
                            population=16,generations=20,factor_budget=768)
        record('GA-w6',s,result.path,(time.perf_counter()-start)*1000,
               extra=dict(queries=result.factor_queries,patterns=result.pattern_evaluations))
        for name,kind in [('PPO-L','ppo'),('DDQN-L','ddqn')]:
            start=time.perf_counter();net=Network.load(ROOT/'models'/f'r22_final_{kind}_{s}.npz')
            p=propose(net,empirical,valid,kind)
            record(name,s,p,(time.perf_counter()-start)*1000)
    # Hard availability envelope is an assumed specification, never estimated.
    start=time.perf_counter();worst=reg.with_losses(o,cap)
    dpw=grid_plan(worst.factors(2),worst.catalogs,valid,ALPHA,2000)
    wp,extra=coordinate(worst,valid,dpw.path)
    record('fixed-envelope',0,wp,(time.perf_counter()-start)*1000,probes=0,extra=extra)
    return rows,time.perf_counter()-tic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','train','evaluate'],required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--episodes',type=int,default=8192)
    parser.add_argument('--count',type=int,default=60);parser.add_argument('--repetitions',type=int,default=2048)
    args=parser.parse_args();(ROOT/'models').mkdir(exist_ok=True)
    meta=ROOT/'results/v22_final_scheduler_metadata.json'
    if args.stage=='prepare':
        import pickle
        declaration=dict(status='declared_before_execution',scenarios=SCENARIOS,training_seeds=[6100001,6100256],
             validation_seeds=[6200001,6200048],test_seeds=[6400001,6400000+3*args.count],
             per_scenario=args.count,training_runs=SEEDS,episodes=args.episodes,probes=PROBES,
             max_active_links=2,catalog_actions=7,alpha=ALPHA,delta=DELTA,
             candidates='fixed-look CP upper intersected with the common hard operating envelope; exact w2 reward and factor features; original KF w6 final verification; no test tuning',
             sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
        meta.write_text(json.dumps(declaration,indent=2));jobs=[(6100001+i,i%3) for i in range(256)]
        jobs += [(6200001+i,i%3) for i in range(48)]
        with ProcessPoolExecutor(args.workers) as pool:prepared=list(pool.map(prepare_one,jobs))
        with (ROOT/'models/r22_final_training_inputs.pkl').open('wb') as f:pickle.dump(prepared,f)
        print('PREPARED',len(prepared),sum(x[-1] for x in prepared),flush=True)
    elif args.stage=='train':
        import pickle
        prepared=pickle.loads((ROOT/'models/r22_final_training_inputs.pkl').read_bytes())
        training=[Surrogate(o,v) for seed,sc,o,v,ok in prepared[:256] if ok and all(np.any(x) for x in v)]
        for s in SEEDS:
            for kind,fn in [('ppo',train_ppo),('ddqn',train_ddqn)]:
                net,log,info=fn(training,s,args.episodes)
                net.save(ROOT/'models'/f'r22_final_{kind}_{s}.npz',kind=kind,seed=s,**info)
                write_csv(ROOT/'results'/f'v22_final_training_{kind}_{s}.csv',log)
                (ROOT/'results'/f'v22_final_training_{kind}_{s}.json').write_text(json.dumps(info,indent=2))
                print('TRAINED',kind,s,info,flush=True)
        # Validation is an audit of final checkpoints, never a model selector.
        validation=[]
        for seed,sc,o,v,ok in prepared[256:]:
            if not ok or any(not np.any(x) for x in v):continue
            for s in SEEDS:
                for kind in ('ppo','ddqn'):
                    net=Network.load(ROOT/'models'/f'r22_final_{kind}_{s}.npz');p=propose(net,o,v,kind)
                    checker=PlanCheck(o,Registry(o),p,6);losses=[]
                    reg=Registry(o)
                    for t,m,power in reg.keys:
                        a=next(a for a in o.catalogs[t] if m in a.support and a.power_w==power)
                        losses.append(a.loss[a.support.index(m)])
                    validation.append(dict(seed=seed,kind=kind,search_seed=s,
                         empirical_certificate=checker.evaluate(losses),objective=sum(g[a].cost for g,a in zip(o.catalogs,p))))
        write_csv(ROOT/'results/v22_final_learning_validation.csv',validation)
    else:
        outputs=[];tic=time.perf_counter();jobs=[(6400001+i,i//args.count,args.repetitions) for i in range(3*args.count)]
        with ProcessPoolExecutor(args.workers) as pool:
            for rows,secs in pool.map(evaluate_one,jobs):
                outputs.extend(rows);write_csv(ROOT/'results/v22_final_scheduler_records.csv',outputs)
                print('TESTED',rows[0]['seed'],rows[0]['scenario'],round(secs,2),
                      [(r['method'],r['search_seed'],r['admitted']) for r in rows],flush=True)
        declaration=json.loads(meta.read_text());declaration.update(status='completed',
             test_wall_seconds=time.perf_counter()-tic,rows=len(outputs),repetitions=args.repetitions,
             models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'models').glob('r22_final_*.npz'))})
        meta.write_text(json.dumps(declaration,indent=2))


if __name__=='__main__':main()
