"""Independent 200-layout admission confirmation for the final research study.

The registered evidence, selection policy, and original finite-window KF
certificate are unchanged from the audited final study implementation.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,csv,json,time,hashlib
import numpy as np
from scheduler_comparison_final import problem,SCENARIOS,SEEDS,PROBES
from candidate_planning import GeometryFactors,cost,initialize,coordinate,genetic
from evidence_certificates import PlanCheck
from contraction_experiments import simulate
from memory_certificates import EYE,posterior
ROOT=Path(__file__).resolve().parent
LAMBDAS=(0.,*np.geomspace(.001,10.,16))

def write(path,rows):
 fields=sorted({k for r in rows for k in r})
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def fim_candidates(o,reg,law,valid):
 """Restricted-catalog logdet-gain/cost family; no installed-anchor expansion.

 Six fixed cardinality/power budgets plus seventeen sequential Lagrange
 weights. Expected information propagates a candidate state only; the exact
 product-law window checker, never this surrogate, authorizes admission.
 """
 H=len(valid);paths=[]
 for a in range(1,len(o.catalogs[0])):
  path=tuple(a for _ in range(H))
  if all(valid[t][a] for t in range(H)):paths.append(path)
 for lam in LAMBDAS:
  P=o.initial.copy();path=[]
  for t,g in enumerate(o.catalogs):
   pred=P+o.process[t]*EYE;J=np.linalg.inv(pred);base=np.linalg.slogdet(J)[1]
   options=[]
   for j,a in enumerate(g):
    if not valid[t][j]:continue
    pp=law[list(reg.ids[t][j])]
    I=np.einsum('m,mij->ij',1-pp,a.H[:,:,None]*a.H[:,None,:]/a.variance[:,None,None]) if len(pp) else np.zeros((2,2))
    score=np.linalg.slogdet(J+I)[1]-base-float(lam)*a.cost
    options.append((score,-a.cost,-j,j,I))
   if not options:break
   z=max(options,key=lambda z:z[:3]);path.append(z[3]);P=posterior(pred,z[4])
  if len(path)==H:paths.append(tuple(path))
 return list(dict.fromkeys(paths))

def original(f,law,valid,seed,extra=()):
 initial,_=initialize(f,law,valid,extra);c=coordinate(f,law,valid,initial)
 out=[]
 for s in SEEDS:
  g=genetic(f,law,valid,initial,seed+s).path
  paths=[p for p in (c,g) if p is not None]
  out.append(min(paths,key=lambda p:(cost(f.o,p),p)) if paths else None)
 return out

def augment(f,law,valid,base,extra=()):
 candidates=fim_candidates(f.o,f.reg,law,valid)
 candidates=[p for p in candidates if f.check(p,law)<=.1-2e-11]
 candidates += [p for p in (*base,*extra) if p is not None and f.check(p,law)<=.1-2e-11]
 best=min(candidates,key=lambda p:(cost(f.o,p),p)) if candidates else None
 best=coordinate(f,law,valid,best)
 out=[]
 for p in base:
  pool=[q for q in (p,best) if q is not None]
  out.append(min(pool,key=lambda q:(cost(f.o,q),q)) if pool else None)
 return out,len(candidates)

def one(job):
 mode,seed,sc,reps=job;start=time.perf_counter()
 cfg,inst,route,o,gamma,reg,truth,cap,k,u,_,valid=problem(seed,*SCENARIOS[sc])
 law=np.minimum(u,cap);eligible=bool(route.flight_feasible and all(np.any(v) for v in valid));f=GeometryFactors(o,reg)
 rows=[];audit={}
 common=dict(cohort=mode,seed=seed,scenario=sc,anchors=cfg.anchors,H=len(route.events),registered=len(k),gate_max=SCENARIOS[sc][2],counts=';'.join(map(str,k)),eligible=int(eligible),repetitions=reps)
 def record(method,run,path,pmap,probes,**extra):
  r=dict(common,method=method,search_seed=run,admitted=0,path='',probes=probes,calibration_s=probes*cfg.pilot_s,**extra)
  if path is not None:
   path=tuple(path)
   if path not in audit:
    check=PlanCheck(o,reg,path,6);audit[path]=(check,simulate(o,path,cfg,seed,reps))
   check,stats=audit[path];c=check.evaluate(pmap)
   if abs(c-f.check(path,pmap))>3e-11 or c>.1-2e-11:raise ArithmeticError('Original KF check failed')
   r.update(admitted=1,path=';'.join(map(str,path)),certificate=c,true_certificate=check.evaluate(truth),objective=cost(o,path),ranging_us=1e6*np.mean([g[a].airtime_s for g,a in zip(o.catalogs,path)]),**stats)
  rows.append(r)
 if not eligible:
  for m in ('hard-augmented','calibrated-augmented','screen-augmented'):
   for s in SEEDS:record(m,s,None,cap,0,status='flight_or_data_ineligible')
  return rows
 hard0=original(f,cap,valid,seed)
 cal0=original(f,law,valid,seed,hard0)
 hard,n_h=augment(f,cap,valid,hard0)
 calibrated,n_c=augment(f,law,valid,cal0,hard)
 for j,s in enumerate(SEEDS):
  for m,p,pmap,probes in [('hard-augmented',hard[j],cap,0),('calibrated-augmented',calibrated[j],law,len(k)*PROBES),('screen-augmented',hard[j] if hard[j] is not None else calibrated[j],cap if hard[j] is not None else law,0 if hard[j] is not None else len(k)*PROBES)]:
   record(m,s,p,pmap,probes,hard_candidates=n_h,refined_candidates=n_c)
  for p0,p1 in [(hard0[j],hard[j]),(cal0[j],calibrated[j])]:
   if p0 is not None and (p1 is None or cost(o,p1)>cost(o,p0)+1e-10):raise ArithmeticError('Portfolio lost original candidate')
 for r in rows:r['layout_wall_s']=time.perf_counter()-start;r['unique_paths_audited']=len(audit)
 return rows

def main():
 """Generate the independent final 200-layout admission cohort from source code."""
 ap=argparse.ArgumentParser(description='Latest independent 200-layout UAV-IoT admission experiment')
 ap.add_argument('--workers',type=int,default=4)
 ap.add_argument('--repetitions',type=int,default=2048)
 ap.add_argument('--limit',type=int,default=200,help='Number of geometries (1-200); less than 200 is a smoke test, not published evidence')
 a=ap.parse_args()
 if not (1 <= a.limit <= 200):ap.error('--limit must be between 1 and 200')
 if a.workers<1 or a.repetitions<1:ap.error('workers and repetitions must be positive')
 (ROOT/'results').mkdir(exist_ok=True)
 (ROOT/'validation').mkdir(exist_ok=True)
 mode='fresh200'
 jobs=[(mode,8000001+j,2,a.repetitions) for j in range(a.limit)]
 stem='final_admission_200' if a.limit==200 and a.repetitions==2048 else 'pilot_admission'
 path=ROOT/'results'/(stem+'_protocol.json')
 source_names=['run_latest.py','candidate_planning.py','scheduler_comparison_final.py','evidence_certificates.py']
 meta=dict(status='declared_before_execution',published_configuration=bool(a.limit==200 and a.repetitions==2048),jobs=jobs,
  search_seeds=SEEDS,lambda_grid=list(map(float,LAMBDAS)),new_independent_layouts=a.limit,
  subset_is_pilot=a.limit<200,registry='restricted K=6H, seven actions, 512 probes/stream, delta=.01',
  selection='cheapest passing original or FIM-family candidate, then coordinate refinement',
  source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_names})
 path.write_text(json.dumps(meta,indent=2))
 rows=[];started=time.perf_counter()
 with ProcessPoolExecutor(a.workers) as pool:
  for j,rr in enumerate(pool.map(one,jobs)):
   rows.extend(rr)
   write(ROOT/'results'/(stem+'_records.csv'),rows)
   print(j+1,'/',len(jobs),'seed',rr[0]['seed'],'elapsed',round(time.perf_counter()-started,1),flush=True)
 meta.update(status='completed',rows=len(rows),wall_s=time.perf_counter()-started)
 path.write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
