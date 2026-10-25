"""Executed reviewer revision 5. Historical runs are never overwritten.

All new radio comparisons use a shared, policy-independent feasible route.
The separately retained 500-layout study still optimizes the route catalog.
"""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,copy,csv,hashlib,json,time,sys
import numpy as np
from scipy.stats import binom
from secure_drop_research import generate_instance,feasible_routes,inv2
from revision_experiments import research_config,table_for,ci99,wilson99
from final_revision_experiments import dictionary,save_csv,paired_rows
from reviewer_extensions import MissionDesigner
from robust_certificates import (Action,Allocation,CertificateRadio,BurstRadio,
    make_tables,plan_certified,simulate,log_radial_bounds,NOMINAL_T,ROBUST_T)

SOURCES=['secure_drop_research.py','wireless_extensions.py','risk_pattern_extensions.py',
         'reviewer_extensions.py','revision_experiments.py','final_revision_experiments.py',
         'robust_certificates.py','robust_experiments.py']
METRICS=['violation','pilot_fraction','pilot_energy_mj','goodput_bits','radio_cost',
         'radial_union_bound','max_conditional_union','budget_exceed_mission','nees',
         'mean_active_links','mean_attempts','mean_repetitions','decision_ms','planning_ms','radio_gap']

def index_range(n,args):return np.array_split(np.arange(n),args.parts)[args.part]
def destination(out,name,args):return out/(name+(f'_part{args.part}' if args.parts>1 else '')+'.csv')
def route_for(inst,cfg):
    routes=feasible_routes(inst,cfg)
    return min(routes,key=lambda r:r.flight_j/cfg.energy_scale_j+cfg.time_weight*r.completion_s/cfg.time_scale_s) if routes else None
def row(inst,cfg,method,allocation,metrics=None,clock=0.,**fields):
    r=dict(seed=inst.seed,method=method,admitted=int(allocation is not None),alpha=cfg.mission_risk,
           radius=cfg.radius_m,anchors=cfg.anchors,area=cfg.area_m,slot_us=cfg.pilot_s*1e6,
           planning_ms=clock,flight_kj=allocation.route.flight_j/1000 if allocation else np.nan,
           objective=allocation.objective if allocation else np.nan,horizon=len(allocation.route.events) if allocation else np.nan)
    for k in METRICS:r.setdefault(k,np.nan)
    if metrics:r.update(metrics)
    if allocation and metrics and np.isfinite(allocation.lower_radio_cost):
        r['radio_lower_bound']=allocation.lower_radio_cost
        # Offline cost only: an offline lower bound is not an online-policy LB.
        if not fields.get('causal',False):r['radio_gap']=(metrics['radio_cost']-allocation.lower_radio_cost)/max(metrics['radio_cost'],1)
    r.update(fields);return r

def summarize(records,keys=('case','method','state')):
    groups={}
    for r in records:groups.setdefault(tuple(r.get(k,'') for k in keys),[]).append(r)
    output=[]
    for values,data in groups.items():
        good=[r for r in data if r['admitted']];r=dict(zip(keys,values));r.update(instances=len(data),admitted=len(good))
        r['admission_low99'],r['admission_high99']=wilson99(len(good),len(data))
        for k in METRICS:r[k],r[k+'_ci99']=ci99([x.get(k,np.nan) for x in good])
        r['max_executed_conditional']=max((x.get('max_conditional_union',0.) for x in good),default=np.nan)
        output.append(r)
    return output

def write_records(out,name,records,args):
    save_csv(destination(out,name+'_records',args),records)
    save_csv(destination(out,name+'_summary',args),summarize(records))

def holm(values):
    values=np.asarray(values);order=np.argsort(values);result=np.empty(len(values));running=0.
    for j,index in enumerate(order):running=max(running,(len(values)-j)*values[index]);result[index]=min(1.,running)
    return result

def legacy_suite(args,out):
    cfg=research_config();code=dictionary(cfg);records=[]
    with Path('results/v4_main_records.csv').open() as f:
        old={int(x['seed']):x for x in csv.DictReader(f) if x['method']=='Robust-SPEB'}
    for count in index_range(args.audit_layouts,args):
        inst=generate_instance(930001+int(count),cfg);tables,_=table_for(inst,cfg,code)
        design=MissionDesigner(feasible_routes(inst,cfg),tables,cfg).plan('SPEB-joint')
        assert bool(design)==bool(int(old[inst.seed]['admitted']))
        allocation=None;metrics=None
        if design:
            assert abs(design.plan.objective-float(old[inst.seed]['objective']))<1e-8
            allocation=Allocation(design.cfg,design.plan.route,tuple(Action(i) for i in design.plan.schedules),
                 design.tables,[CertificateRadio(t) for t in design.tables],'Legacy-SPEB',design.plan.objective)
            metrics=simulate(allocation,inst,args.audit_reps)
            metrics['excess_p']=float(binom.sf(metrics['failures']-1,args.audit_reps,.1))
        r=row(inst,cfg,'Original SPEB',allocation,metrics,case='original500',state='bounded')
        r['legacy_sufficient_bound']=float(old[inst.seed]['radial_union_bound']) if design else np.nan
        records.append(r)
        if (count+1)%25==0:
            print('legacy risk',count+1,flush=True);write_records(out,'v5_legacy',records,args)
    write_records(out,'v5_legacy',records,args)

def price_suite(args,out):
    cfg=research_config();code=dictionary(cfg);records=[]
    for count in index_range(args.instances,args):
        inst=generate_instance(960001+int(count),cfg);tables=make_tables(inst,cfg,code);route=route_for(inst,cfg)
        radios=[CertificateRadio(t) for t in tables];plans={}
        for method,mode,causal in [('SPEB','SPEB',False),('GRE','GRE',False),
                ('Certified SPEB','Certified-SPEB',False),('Causal GRE','GRE',True),
                ('Causal certified SPEB','Certified-SPEB',True)]:
            clock=time.perf_counter()
            if causal:plan=plans['GRE' if mode=='GRE' else 'Certified SPEB']
            else:plan=plan_certified(inst,cfg,tables,mode,radios,routes=[route]) if route else None
            plans[method]=plan
            met=simulate(plan,inst,args.price_reps,causal=causal) if plan else None
            records.append(row(inst,cfg,method,plan,met,1000*(time.perf_counter()-clock),case='matched-certificate',state='bounded',causal=causal))
        if (count+1)%25==0:print('certificate price',count+1,flush=True);write_records(out,'v5_price',records,args)
    write_records(out,'v5_price',records,args)

def markov_suite(args,out):
    records=[]
    for count in index_range(args.instances,args):
        seed=950001+int(count)
        for slot in (100,50):
            cfg=research_config(pilot_s=slot*1e-6,pilot_control_s=.4*slot*1e-6)
            inst=generate_instance(seed,cfg);tables=make_tables(inst,cfg,dictionary(cfg));route=route_for(inst,cfg)
            specifications=[('Independent','independent'),('Markov','nominal')] if slot==100 else [
                ('Independent','independent'),('Markov','nominal'),('Robust-Markov','robust')]
            for mode,law in specifications:
                clock=time.perf_counter();radios=[BurstRadio(t,law,(1,) if slot==100 else (1,2,3)) for t in tables]
                plan=plan_certified(inst,cfg,tables,mode,radios,routes=[route]) if route else None
                elapsed=1000*(time.perf_counter()-clock)
                for initial in ('stationary','1') if slot==50 else ('stationary',):
                    met=simulate(plan,inst,args.replications,loss_mode='markov',initial=initial) if plan else None
                    records.append(row(inst,cfg,mode,plan,met,elapsed,case=f'slot{slot}',state=initial))
                if slot==50 and law=='robust':
                    met=simulate(plan,inst,args.replications,loss_mode='markov',initial='1',
                        true_transition=ROBUST_T,true_bad=.85,true_good_multiplier=1.5) if plan else None
                    records.append(row(inst,cfg,mode,plan,met,elapsed,case='slot50-boundary',state='1'))
        if (count+1)%20==0:print('Markov design',count+1,flush=True);write_records(out,'v5_markov',records,args)
    write_records(out,'v5_markov',records,args)

def causal_suite(args,out):
    records=[];cfg=research_config(pilot_s=50e-6,pilot_control_s=20e-6);code=dictionary(cfg)
    for count in index_range(args.causal_layouts,args):
        inst=generate_instance(950001+int(count),cfg);tables=make_tables(inst,cfg,code);route=route_for(inst,cfg)
        radios=[BurstRadio(t,'nominal') for t in tables]
        plan=plan_certified(inst,cfg,tables,'Markov',radios,routes=[route]) if route else None
        for method,causal,belief in [('Static Markov',False,False),('Causal worst-state',True,False),('Causal belief',True,True)]:
            met=simulate(plan,inst,args.causal_reps,causal=causal,belief_aware=belief,loss_mode='markov') if plan else None
            records.append(row(inst,cfg,method,plan,met,case='causal-markov',state='stationary',causal=causal))
        if (count+1)%10==0:print('causal Markov',count+1,flush=True);write_records(out,'v5_causal',records,args)
    write_records(out,'v5_causal',records,args)

def risk_suite(args,out):
    records=[]
    settings=[(a,2.) for a in (.10,.05,.01,.001,.00001)]+[(.05,r) for r in (1.,1.5,3.)]
    for alpha,radius in settings:
        cfg=research_config(mission_risk=alpha,radius_m=radius);code=dictionary(cfg)
        for count in index_range(args.instances,args):
            inst=generate_instance(970001+int(count),cfg);tables=make_tables(inst,cfg,code);route=route_for(inst,cfg)
            radios=[CertificateRadio(t) for t in tables]
            for method in ('SPEB','GRE','Certified-SPEB'):
                clock=time.perf_counter();p=plan_certified(inst,cfg,tables,method,radios,routes=[route]) if route else None
                met=simulate(p,inst,args.replications) if p else None
                records.append(row(inst,cfg,method,p,met,1000*(time.perf_counter()-clock),case=f'alpha{alpha:g}-R{radius:g}',state='bounded'))
        print('risk grid',alpha,radius,flush=True);write_records(out,'v5_risk',records,args)

def nested_instance(seed,cfg):
    master=generate_instance(seed,replace(cfg,anchors=8));order=np.array([0,2,4,6,1,5,3,7])[:cfg.anchors]
    master.anchors=master.anchors[order];master.gains=master.gains[:,order]
    return master

def density_suite(args,out):
    records=[]
    for anchors,area in [(4,1200),(6,1200),(8,1200),(6,900),(6,600)]:
        cfg=research_config(anchors=anchors,area_m=area);code=dictionary(cfg)
        for count in index_range(args.instances,args):
            inst=nested_instance(975001+int(count),cfg);tables=make_tables(inst,cfg,code);route=route_for(inst,cfg)
            radios=[CertificateRadio(t) for t in tables];base=None
            for method in ('GRE','Certified-SPEB','Causal GRE'):
                clock=time.perf_counter();p=base if method=='Causal GRE' else plan_certified(inst,cfg,tables,method,radios,routes=[route]) if route else None
                if method=='GRE':base=p
                met=simulate(p,inst,args.replications,causal=method=='Causal GRE') if p else None
                records.append(row(inst,cfg,method,p,met,1000*(time.perf_counter()-clock),case=f'M{anchors}-A{area}',state='bounded',causal=method=='Causal GRE'))
        print('density',anchors,area,flush=True);write_records(out,'v5_density',records,args)

def stress_suite(args,out):
    records=[];cfg=research_config();code=dictionary(cfg)
    for count in index_range(args.instances,args):
        inst=generate_instance(980001+int(count),cfg);route=route_for(inst,cfg)
        for label,nonlinear,bias,sd,margin in [('Affine',False,0,0,0),('Nonlinear ranges',True,0,0,0),
            ('Nonlinear plus 1m',True,1.,0,0),('CSI 2dB',False,0,2.,0),('CSI 2dB margin3',False,0,2.,3.)]:
            designed=copy.deepcopy(inst);designed.gains*=10**(-margin/10)
            tables=make_tables(designed,cfg,code);p=plan_certified(designed,cfg,tables,'GRE',routes=[route]) if route else None
            met=simulate(p,inst,args.replications,causal=True,nonlinear=nonlinear,bias_m=bias,csi_sd=sd,csi_margin=margin) if p else None
            records.append(row(inst,cfg,label,p,met,case='model-stress',state='bounded',causal=True))
        if (count+1)%25==0:print('model stress',count+1,flush=True);write_records(out,'v5_stress',records,args)
    write_records(out,'v5_stress',records,args)

def bias_suite(args,out):
    records=[];cfg=research_config();code=dictionary(cfg)
    for count in index_range(args.bias_layouts,args):
        inst=generate_instance(990001+int(count),cfg);tables=make_tables(inst,cfg,code);route=route_for(inst,cfg)
        radios=[BurstRadio(t,'independent',(1,)) for t in tables]
        for bound in (0.,.05,.1,.25,1.):
            p=plan_certified(inst,cfg,tables,'Independent',radios,bias_bound=bound,repetitions=(1,),routes=[route]) if route else None
            met=simulate(p,inst,args.replications,loss_mode='independent',bias_m=bound,bias_mode='adversarial') if p else None
            records.append(row(inst,cfg,f'Bound {bound:g}',p,met,case=f'bias{bound:g}',state='matched'))
            if bound==0.:
                met=simulate(p,inst,args.replications,loss_mode='independent',bias_m=.25,bias_mode='adversarial') if p else None
                records.append(row(inst,cfg,'Unprotected .25',p,met,case='bias.25-unprotected',state='mismatch'))
        if (count+1)%20==0:print('bounded bias',count+1,flush=True);write_records(out,'v5_bias',records,args)
    write_records(out,'v5_bias',records,args)

def merge(args,out):
    for suite in ('legacy','price','markov','causal','risk','density','stress','bias'):
        paths=sorted(out.glob(f'v5_{suite}_records_part*.csv'))
        if not paths:continue
        records=[]
        for path in paths:
            with path.open() as f:
                for r in csv.DictReader(f):
                    row={}
                    for k,v in r.items():
                        try:row[k]=float(v)
                        except ValueError:row[k]=v
                        if k in ('admitted','seed','failures','samples') and isinstance(row[k],float) and np.isfinite(row[k]):row[k]=int(row[k])
                    records.append(row)
        if suite=='legacy':
            good=[r for r in records if r['admitted']];adjusted=holm([r['excess_p'] for r in good])
            for r,p in zip(good,adjusted):r['holm_excess_p']=p;r['confirmed_excess']=int(p<=.01 and r['violation']>.1)
        seen=set()
        for r in records:
            key=(r['seed'],r.get('case',''),r['method'],r.get('state',''))
            if key in seen:raise RuntimeError('Duplicate part records')
            seen.add(key)
        save_csv(out/f'v5_{suite}_records.csv',records);save_csv(out/f'v5_{suite}_summary.csv',summarize(records))
        if suite=='price':
            pairs=[('GRE','Certified SPEB'),('Causal GRE','Causal certified SPEB'),('Certified SPEB','SPEB'),('Causal GRE','GRE')]
            save_csv(out/'v5_price_paired.csv',paired_rows(records,pairs))
        print('merged',suite,len(records),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['all','merge','legacy','price','markov','risk','density','stress','bias','causal'],default='all')
    parser.add_argument('--instances',type=int,default=200);parser.add_argument('--replications',type=int,default=512)
    parser.add_argument('--audit-layouts',type=int,default=500);parser.add_argument('--audit-reps',type=int,default=16384)
    parser.add_argument('--price-reps',type=int,default=2048);parser.add_argument('--causal-layouts',type=int,default=40)
    parser.add_argument('--causal-reps',type=int,default=128);parser.add_argument('--bias-layouts',type=int,default=100)
    parser.add_argument('--part',type=int,default=0);parser.add_argument('--parts',type=int,default=1)
    parser.add_argument('--output',type=Path,default=Path('results'));args=parser.parse_args()
    if not 0<=args.part<args.parts:parser.error('Invalid part index')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.suite=='merge':merge(args,args.output);return
    sources={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in SOURCES}
    suites=['legacy','price','markov','risk','density','stress','bias','causal'] if args.suite=='all' else [args.suite]
    for name in suites:
        start=time.perf_counter();globals()[name+'_suite'](args,args.output)
        for path,digest in sources.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Numerical source changed during experiment')
        suffix=f'_part{args.part}' if args.parts>1 else ''
        meta=dict(suite=name,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  sources=sources,runtime_s=time.perf_counter()-start,python=sys.version.split()[0],numpy=np.__version__,
                  route_rule='Shared minimum propulsion/time route, independent of radio policies',
                  protocol='First successful range per anchor; every attempted repeat charged; noise-independent losses')
        (args.output/f'v5_{name}_metadata{suffix}.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.parts==1:
        # Generate the same final inference as merge without fabricating parts.
        if 'legacy' in suites:
            with (args.output/'v5_legacy_records.csv').open() as f:data=list(csv.DictReader(f))
            good=[r for r in data if int(r['admitted'])];p=holm([float(r['excess_p']) for r in good])
            for r,value in zip(good,p):r['holm_excess_p']=value;r['confirmed_excess']=int(value<=.01 and float(r['violation'])>.1)
            save_csv(args.output/'v5_legacy_records.csv',data)
        from assemble_robust_results import main as assemble
        assemble(args.output)

if __name__=='__main__':main()
