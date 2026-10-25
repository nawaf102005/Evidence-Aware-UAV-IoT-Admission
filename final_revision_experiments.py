"""Third revision: common-admission comparisons, weighted risk, coverage and size.

Each suite has a separate record/metadata file; --reports only reads records.
Default main study: 500 layouts x 128 realizations. Other suites: 200 layouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import time
from dataclasses import asdict,replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Ellipse
import numpy as np
import scipy
from scipy.stats import binomtest

from secure_drop_research import Config,RadioTable,generate_instance,feasible_routes,save_csv,inv2
from wireless_extensions import (ExtendedRadio,extended_plan,adaptive_plan,evaluate,
    gaussian_envelope_cap,radial_tail_bounds,generate_umi_instance,packet_payload_budget)
from revision_experiments import research_config,table_for,ci99,wilson99,read_csv
from risk_pattern_extensions import (PatternRadio,pattern_plan,evaluate_pattern,split_plan,
    choose_gre_terms,generate_aerial_instance,scalable_eopt,markov_pattern_probabilities,
    loss_patterns,pattern_probabilities)


LABELS={'SDW-power':'SDW power','SPEB-joint':'Robust SPEB','ERGO-joint':'E-opt / isotropic',
    'GRE-joint':'GRE','Adaptive-GRE':'Causal GRE','GRE-shortest':'GRE shortest',
    'PHY-GRE':'PHY-GRE (split)','Pattern-GRE':'Pattern-GRE','Causal-Pattern-GRE':'Causal Pattern-GRE'}
METHODS=('SDW-power','SPEB-joint','ERGO-joint','GRE-joint','Adaptive-GRE','GRE-shortest')


def save_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def dictionary(cfg):
    return np.array(list(itertools.product(range(len(cfg.powers_w)),repeat=cfg.anchors)),dtype=int)


def output_record(seed,method,plan,metrics,clock):
    row=dict(seed=seed,method=method,admitted=int(plan is not None),planning_ms=clock)
    if metrics: row.update(metrics)
    for key in ('violation','pilot_fraction','goodput_bits','radial_union_bound','nees',
                'conditional_union_bound','max_conditional_union','decision_ms_mean'):
        row.setdefault(key,np.nan)
    row.update(flight_kj=plan.route.flight_j/1000 if plan else np.nan,
        objective=plan.objective if plan else np.nan)
    return row


def summary(rows):
    result=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        all_rows=[r for r in rows if r['method']==method]; accepted=[r for r in all_rows if r['admitted']]
        entry=dict(method=method,instances=len(all_rows),admitted=len(accepted))
        entry['admission_low99'],entry['admission_high99']=wilson99(len(accepted),len(all_rows))
        for key in ('violation','pilot_fraction','goodput_bits','radial_union_bound','nees',
                    'conditional_union_bound','max_conditional_union','flight_kj','objective',
                    'planning_ms','decision_ms_mean'):
            entry[key],entry[key+'_ci99']=ci99([r.get(key,np.nan) for r in accepted])
        entry['bound_above_target']=sum(r['radial_union_bound']>.1+1e-9 for r in accepted)
        result.append(entry)
    return result


def paired_rows(rows,pairs):
    result=[]
    for first,second in pairs:
        a={r['seed']:r for r in rows if r['method']==first}
        b={r['seed']:r for r in rows if r['method']==second}
        common=[s for s in a if a[s]['admitted'] and b[s]['admitted']]
        gains=sum(a[s]['admitted'] and not b[s]['admitted'] for s in a)
        losses=sum(b[s]['admitted'] and not a[s]['admitted'] for s in a)
        row=dict(first=first,second=second,common=len(common),gains=gains,losses=losses,
            mcnemar_p=binomtest(gains,gains+losses,.5).pvalue if gains+losses else 1.)
        for key in ('violation','pilot_fraction','goodput_bits','flight_kj','objective'):
            row[key+'_difference'],row[key+'_ci99']=ci99([a[s][key]-b[s][key] for s in common])
        result.append(row)
    ordered=sorted(result,key=lambda r:r['mcnemar_p']); previous=0.
    for i,row in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*row['mcnemar_p']))
        row['holm_p']=previous
    return result


def main_suite(args,output):
    cfg=research_config(); code=dictionary(cfg); rows=[]
    for trial in range(args.main_instances):
        instance=generate_instance(args.seed+trial,cfg); tables,extra=table_for(instance,cfg,code)
        routes=feasible_routes(instance,cfg); plans={}
        for method in METHODS:
            clock=time.perf_counter()
            if method=='Adaptive-GRE':
                plan=adaptive_plan(plans['GRE-joint'],method,extra,tables,cfg)
            elif method=='GRE-shortest':
                shortest=min(routes,key=lambda r:r.lengths.sum()) if routes else None
                plan=extended_plan('GRE-joint',[shortest] if shortest else [],tables,extra,cfg)
            else: plan=extended_plan(method,routes,tables,extra,cfg)
            plans[method]=plan; milliseconds=1000*(time.perf_counter()-clock)
            metrics=evaluate(instance,plan,tables,extra,cfg,args.replications) if plan else None
            rows.append(output_record(instance.seed,method,plan,metrics,milliseconds))
        if (trial+1)%50==0: print(f'main {trial+1}/{args.main_instances}',flush=True)
    save_csv(output/'final_main_records.csv',rows)
    save_csv(output/'final_main_summary.csv',summary(rows))
    save_csv(output/'final_main_paired.csv',paired_rows(rows,[('GRE-joint','ERGO-joint'),
        ('Adaptive-GRE','GRE-joint'),('Adaptive-GRE','SPEB-joint'),('GRE-joint','GRE-shortest')]))


def patterns_suite(args,output):
    cfg=research_config(); code=dictionary(cfg); rows=[]
    for trial in range(args.instances):
        instance=generate_instance(args.seed+trial,cfg); tables,extra=table_for(instance,cfg,code)
        routes=feasible_routes(instance,cfg)
        clock=time.perf_counter(); weighted=[PatternRadio(t) for t in tables]
        plan=pattern_plan(routes,tables,weighted,cfg)
        milliseconds=1000*(time.perf_counter()-clock)
        for method,causal in [('Pattern-GRE',False),('Causal-Pattern-GRE',True)]:
            metrics=evaluate_pattern(instance,plan,tables,weighted,cfg,args.replications,causal) if plan else None
            rows.append(output_record(instance.seed,method,plan,metrics,milliseconds))
        phy_cfg=replace(cfg,mission_risk=.05)
        phy_tables,phy_extra=table_for(instance,phy_cfg,code)
        clock=time.perf_counter(); phy_plan=split_plan(routes,phy_tables,phy_extra,phy_cfg)
        phy_ms=1000*(time.perf_counter()-clock)
        metrics=evaluate(instance,phy_plan,phy_tables,phy_extra,phy_cfg,args.replications,loss_mode='independent') if phy_plan else None
        rows.append(output_record(instance.seed,'PHY-GRE',phy_plan,metrics,phy_ms))
        if (trial+1)%25==0: print(f'patterns {trial+1}/{args.instances}',flush=True)
    save_csv(output/'pattern_records.csv',rows);save_csv(output/'pattern_summary.csv',summary(rows))
    save_csv(output/'pattern_paired.csv',paired_rows(rows,[('Pattern-GRE','PHY-GRE'),
        ('Causal-Pattern-GRE','Pattern-GRE'),('Causal-Pattern-GRE','PHY-GRE')]))


def propagation_suite(args,output):
    rows=[]; diagnostics=[]
    cases=[('UMi-1200',research_config(altitude_m=10.),generate_umi_instance),
           ('UMi-600',research_config(altitude_m=10.,area_m=600.),generate_umi_instance),
           ('UMa-AV-1200',research_config(altitude_m=35.),generate_aerial_instance),
           ('UMa-AV-600',research_config(altitude_m=35.,area_m=600.),generate_aerial_instance)]
    for case,cfg,generator in cases:
        code=dictionary(cfg)
        for trial in range(args.instances):
            try:
                instance=generator(550001+trial,cfg)
            except ValueError as error:
                if 'applicability' not in str(error) and 'd2D=10' not in str(error): raise
                for method in ('ERGO-joint','GRE-joint','Adaptive-GRE','SPEB-joint'):
                    row=output_record(550001+trial,method,None,None,0.)
                    row.update(case=case,domain_valid=0);rows.append(row)
                diagnostics.append(dict(seed=550001+trial,case=case,node=-1,domain_valid=0))
                continue
            tables,extra=table_for(instance,cfg,code)
            routes=feasible_routes(instance,cfg); main_plan=None
            for method in ('ERGO-joint','GRE-joint','Adaptive-GRE','SPEB-joint'):
                clock=time.perf_counter()
                plan=adaptive_plan(main_plan,'Adaptive-GRE',extra,tables,cfg) if method=='Adaptive-GRE' else extended_plan(method,routes,tables,extra,cfg)
                if method=='GRE-joint': main_plan=plan
                milliseconds=1000*(time.perf_counter()-clock)
                metrics=evaluate(instance,plan,tables,extra,cfg,args.replications) if plan else None
                row=output_record(instance.seed,method,plan,metrics,milliseconds); row.update(case=case,domain_valid=1); rows.append(row)
            # Local all-peak information ceiling; separate information
            # insufficiency from route/battery failure and expected-data limits.
            for node in range(1,cfg.stops+1):
                peak=int(np.flatnonzero(np.all(code==len(cfg.powers_w)-1,axis=1))[0])
                cap=gaussian_envelope_cap(cfg)
                incoming=float(np.linalg.norm(instance.nodes[node]-instance.nodes[0]))
                prior=(cap+cfg.drift_m2_per_m*incoming)*np.eye(2)
                risks=[float(radial_tail_bounds(inv2(inv2(prior)+S[peak]),cfg.radius_m,256)[1]) for S in extra[node].surviving]
                gamma_peak=tables[node].sinr[:,-1]
                diagnostics.append(dict(seed=instance.seed,case=case,node=node,domain_valid=1,route_available=int(bool(routes)),
                    peak_worst_risk=max(risks),peak_information_failure=int(max(risks)>cfg.mission_risk/cfg.epochs_max),
                    peak_payload_failure=int(tables[node].bits[peak]<cfg.data_min_bits),
                    median_peak_sinr_db=float(np.median(10*np.log10(gamma_peak))),
                    median_peak_std_m=float(np.median(np.sqrt(tables[node].variance[:,-1])))))
            if (trial+1)%50==0: print(f'{case} {trial+1}/{args.instances}',flush=True)
    save_csv(output/'propagation_records.csv',rows);save_csv(output/'propagation_diagnostics.csv',diagnostics)
    totals=[]
    for case,_,_ in cases:
        for row in summary([r for r in rows if r['case']==case]):row['case']=case;totals.append(row)
    save_csv(output/'propagation_summary.csv',totals)
    # Controlled UMi channel interventions retain layout, route and jammer.
    ablations=[]; cfg=research_config(altitude_m=10.); code=dictionary(cfg)
    from wireless_extensions import umi_pathloss
    for trial in range(args.instances):
        instance=generate_umi_instance(550001+trial,cfg)
        d=np.linalg.norm(instance.nodes[:,None]-instance.anchors[None],axis=2)
        los_probability=np.minimum(18/d,1)*(1-np.exp(-d/36))+np.exp(-d/36)
        los=np.random.default_rng(instance.seed+40_000_000).random(d.shape)<los_probability
        variants={'standard':instance,'no-shadow':replace(instance,gains=10**(-umi_pathloss(d,los)/10)),
            'all-LOS-no-shadow':replace(instance,gains=10**(-umi_pathloss(d,True)/10))}
        for variant,item in variants.items():
            tables,extra=table_for(item,cfg,code); plan=extended_plan('GRE-joint',feasible_routes(item,cfg),tables,extra,cfg)
            ablations.append(dict(seed=item.seed,variant=variant,admitted=int(plan is not None)))
        floor_cfg=replace(cfg,range_floor_m=.15)
        tables,extra=table_for(instance,floor_cfg,code)
        plan=extended_plan('GRE-joint',feasible_routes(instance,floor_cfg),tables,extra,floor_cfg)
        ablations.append(dict(seed=instance.seed,variant='floor-0.15',admitted=int(plan is not None)))
    save_csv(output/'umi_counterfactual_records.csv',ablations)


def sensitivity_suite(args,output):
    rows=[]; quadrature=[]; split_rows=[]
    settings=[('r',x,research_config(erasure_budget=x)) for x in (0,1,2)]
    settings += [('control_us',x,research_config(pilot_control_s=x*1e-6)) for x in (20,40,60)]
    for parameter,value,cfg in settings:
        code=dictionary(cfg)
        for trial in range(args.instances):
            instance=generate_instance(660001+trial,cfg); tables,extra=table_for(instance,cfg,code)
            plan=extended_plan('GRE-joint',feasible_routes(instance,cfg),tables,extra,cfg)
            row=dict(seed=instance.seed,parameter=parameter,value=value,admitted=int(plan is not None),
                pilot_fraction=plan.mean_pilot_fraction if plan else np.nan)
            rows.append(row)
        print(f'sensitivity {parameter}={value} complete',flush=True)
    cfg=research_config(); code=dictionary(cfg)
    for trial in range(args.instances):
        instance=generate_instance(660001+trial,cfg); table=RadioTable(instance,1,cfg,code); table.bits=packet_payload_budget(table)
        extra=ExtendedRadio(table); incoming=np.linalg.norm(instance.nodes[1]-instance.nodes[0])
        prior=(gaussian_envelope_cap(cfg)+cfg.drift_m2_per_m*incoming)*np.eye(2)
        for terms in (64,128,256,512):
            clock=time.perf_counter(); index=choose_gre_terms(extra,prior,terms)
            if index is not None:
                brackets=[radial_tail_bounds(inv2(inv2(prior)+S[index]),cfg.radius_m,terms) for S in extra.surviving]
                gap=max(float(up-lo) for lo,up in brackets)
                upper=max(float(up) for lo,up in brackets)
            else:gap=upper=np.nan
            quadrature.append(dict(seed=instance.seed,terms=terms,feasible=int(index is not None),
                pilot_fraction=table.count[index]*cfg.pilot_s/cfg.frame_s if index is not None else np.nan,
                bracket_gap=gap,risk_upper=upper,milliseconds=1000*(time.perf_counter()-clock)))
        routes=feasible_routes(instance,cfg)
        for local_risk in (.02,.04,.05,.06,.08):
            split_cfg=replace(cfg,mission_risk=local_risk)
            tables,radios=table_for(instance,split_cfg,code)
            plan=split_plan(routes,tables,radios,split_cfg,packet_risk=.1-local_risk)
            split_rows.append(dict(seed=instance.seed,local_risk=local_risk,packet_risk=.1-local_risk,
                admitted=int(plan is not None),pilot_fraction=plan.mean_pilot_fraction if plan else np.nan,
                objective=plan.objective if plan else np.nan))
        if (trial+1)%50==0:print(f'quadrature / risk split {trial+1}/{args.instances}',flush=True)
    save_csv(output/'final_sensitivity_records.csv',rows)
    save_csv(output/'quadrature_records.csv',quadrature);save_csv(output/'risk_split_records.csv',split_rows)
    # Risk-split choice maximizes admission, then minimizes mean cost, using
    # geometry/channel inputs alone; no error-outcome tuning is used.
    optimized=[]
    for seed in dict.fromkeys(r['seed'] for r in split_rows):
        candidates=[r for r in split_rows if r['seed']==seed and r['admitted']]
        row=min(candidates,key=lambda r:r['objective']).copy() if candidates else dict(seed=seed,admitted=0,local_risk=np.nan,pilot_fraction=np.nan,objective=np.nan)
        optimized.append(row)
    save_csv(output/'risk_split_selected.csv',optimized)
    # Exact Markov pattern law: compare count tails under identical marginals.
    markov=[]; masks=loss_patterns(6); T=np.array([[.97,.03],[.2,.8]]); stationary=np.array([.2/.23,.03/.23])
    for initial in (np.array([1.,0.]),np.array([0.,1.]),stationary):
        law=markov_pattern_probabilities(np.full(6,.02),np.full(6,.8),T,initial)
        marginal=law@masks; independent=pattern_probabilities(marginal,masks)
        for r in (0,1,2):
            markov.append(dict(initial_good=float(initial[0]),budget=r,
                markov_tail=float(law[masks.sum(axis=1)>r].sum()),
                independent_tail=float(independent[masks.sum(axis=1)>r].sum())))
    save_csv(output/'markov_pattern_tails.csv',markov)


def scalability_suite(args,output):
    rows=[]
    for M in (4,6,8,10,20,50):
        cfg=research_config(anchors=M)
        for trial in range(args.solver_instances):
            instance=generate_instance(770001+trial,cfg); length=np.linalg.norm(instance.nodes[1]-instance.nodes[0])
            result=scalable_eopt(instance,1,length,cfg); result.pop('levels')
            result.update(seed=instance.seed,anchors=M)
            if M<=8:
                code=dictionary(cfg); table=RadioTable(instance,1,cfg,code); table.bits=packet_payload_budget(table)
                clock=time.perf_counter(); index=table.select(cfg.covariance_cap+cfg.drift_m2_per_m*length,1)
                result['exact_cost']=table.cost[index] if index is not None else np.nan
                result['exact_selection_ms']=1000*(time.perf_counter()-clock)
                if index is not None and result['lower_bound']>result['exact_cost']+1e-6:
                    raise RuntimeError('Invalid relaxation lower bound.')
            else: result['exact_cost']=result['exact_selection_ms']=np.nan
            rows.append(result)
        print(f'scale M={M} complete',flush=True)
    save_csv(output/'scalable_solver_records.csv',rows)


def reports(output):
    figures=output.parent/'figures';figures.mkdir(exist_ok=True)
    main=read_csv(output/'final_main_summary.csv');packet=read_csv(output/'pattern_summary.csv')
    prop=read_csv(output/'propagation_summary.csv');paired=read_csv(output/'final_main_paired.csv')
    def fmt(row,key,scale=1):
        mean=row[key]*scale;ci=row.get(key+'_ci99',np.nan)*scale
        return f'${mean:.3f}\\pm{ci:.3f}$' if np.isfinite(ci) and row['admitted']>=10 else (f'{mean:.3f}' if row['admitted']>=10 else '--')
    table=[r'\begin{table*}[t]',r'\centering\small',
        r'\caption{Default 40/60-$\mu$s protocol. Panel A: bounded one-erasure tests, 500 layouts $\times$ 128 realizations. Panel B: independent PHY losses, 200 paired layouts $\times$ 128 realizations. Layout means have 99\% intervals; admission counts include all layouts. Panels use different loss guarantees.}',
        r'\label{tab:main}',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Policy & Admitted & Violation & Pilot fraction & Payload (kbit) & Mean radial bound \\',r'\midrule',
        r'\multicolumn{6}{l}{A. Bounded erasures; causal GRE retains the admitted GRE route} \\']
    for r in main:
        table.append(f"{LABELS[r['method']]} & {int(r['admitted'])}/{int(r['instances'])} & {fmt(r,'violation')} & {fmt(r,'pilot_fraction')} & {fmt(r,'goodput_bits',.001)} & {r['radial_union_bound']:.3f} "+chr(92)*2)
    table += [r'\midrule',r'\multicolumn{6}{l}{B. Independent packet losses; weighted risk includes every loss pattern} \\']
    for r in packet:
        table.append(f"{LABELS[r['method']]} & {int(r['admitted'])}/{int(r['instances'])} & {fmt(r,'violation')} & {fmt(r,'pilot_fraction')} & {fmt(r,'goodput_bits',.001)} & {r['radial_union_bound']:.3f} "+chr(92)*2)
    table += [r'\bottomrule\end{tabular}',r'\end{table*}']
    (output/'final_main_table.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{table*}[t]',r'\centering\small',
        r'\caption{Propagation subsets with the default packet/airtime protocol; 200 layouts and 128 realizations in each setting. UMi uses terminal/anchor heights 20/10 m; UMa-AV uses 60/25 m. Metrics with fewer than 10 admitted layouts are suppressed. Area side lengths are explicit; no deployment validation is inferred.}',
        r'\label{tab:propagation}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Scenario & Policy & Admitted & Violation & Pilot fraction & Payload (kbit) \\',r'\midrule']
    for case in dict.fromkeys(r['case'] for r in prop):
        for r in [r for r in prop if r['case']==case and r['method'] in ('GRE-joint','Adaptive-GRE','SPEB-joint')]:
            table.append(f"{case} & {LABELS[r['method']]} & {int(r['admitted'])}/{int(r['instances'])} & {fmt(r,'violation')} & {fmt(r,'pilot_fraction')} & {fmt(r,'goodput_bits',.001)} "+chr(92)*2)
    table += [r'\bottomrule\end{tabular}',r'\end{table*}'];(output/'final_propagation_table.tex').write_text('\n'.join(table)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8.8,2.65),layout='constrained')
    pairs=[('GRE-joint','ERGO-joint'),('Adaptive-GRE','GRE-joint'),('Adaptive-GRE','SPEB-joint'),('GRE-joint','GRE-shortest')]
    names=['GRE − E-opt','Causal − GRE','Causal − SPEB','GRE − shortest']
    for ax,key,title,scale in zip(axes,('pilot_fraction','goodput_bits'),('Paired pilot difference','Paired payload difference (kbit)'),(1,.001)):
        selected=[next(r for r in paired if (r['first'],r['second'])==p) for p in pairs]
        ax.errorbar(range(4),[r[key+'_difference']*scale for r in selected],
            yerr=[r[key+'_ci99']*scale for r in selected],fmt='o',capsize=4,color='#226b82')
        ax.axhline(0,color='#666666',lw=.8);ax.set_xticks(range(4),names,rotation=24,ha='right',fontsize=8)
        ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
    fig.savefig(figures/'final_paired.pdf',bbox_inches='tight');fig.savefig(figures/'final_paired.png',dpi=160,bbox_inches='tight');plt.close(fig)
    q=read_csv(output/'quadrature_records.csv'); splits=read_csv(output/'risk_split_records.csv')
    sens=read_csv(output/'final_sensitivity_records.csv')
    fig,axes=plt.subplots(1,3,figsize=(9.1,2.7),layout='constrained')
    for terms in (64,128,256,512):
        values=[r['bracket_gap'] for r in q if r['terms']==terms and r['feasible']]
        axes[0].plot(terms,np.mean(values),'o',color='#226b82')
    axes[0].set_xlabel('Quadrature terms N');axes[0].set_ylabel('Mean certified bracket width');axes[0].set_yscale('log');axes[0].grid(alpha=.2)
    for ax,param,label in ((axes[1],'r','Erasure budget r'),(axes[2],'control_us','Control duration (µs)')):
        selected=[r for r in sens if r['parameter']==param]
        x=sorted(set(r['value'] for r in selected));y=[];lo=[];hi=[]
        for v in x:
            group=[r for r in selected if r['value']==v];p=sum(r['admitted'] for r in group)/len(group);low,high=wilson99(sum(r['admitted'] for r in group),len(group));y.append(p);lo.append(max(0,p-low));hi.append(max(0,high-p))
        ax.errorbar(x,y,yerr=[lo,hi],fmt='o-',capsize=3,color='#226b82');ax.set_ylim(-.05,1.05);ax.set_xlabel(label);ax.set_ylabel('GRE admission fraction');ax.grid(alpha=.2)
    fig.savefig(figures/'final_sensitivity.pdf',bbox_inches='tight');fig.savefig(figures/'final_sensitivity.png',dpi=160,bbox_inches='tight');plt.close(fig)
    system_figures(figures)


def system_figures(figures):
    fig,axes=plt.subplots(1,3,figsize=(10.,3.0),layout='constrained')
    ax=axes[0]; anchors=np.array([[.1,.15],[.16,.82],[.58,.96],[.91,.8],[.92,.12],[.48,.03]])
    route=np.array([[.1,.07],[.30,.30],[.70,.38],[.57,.65],[.48,.49],[.2,.65],[.1,.07]])
    ax.plot(route[:,0],route[:,1],'-',color='#777777',lw=1.2);ax.scatter(route[1:4,0],route[1:4,1],s=25,color='#c68a32',label='Service')
    ax.scatter(anchors[:,0],anchors[:,1],marker='^',s=35,color='#226b82',label='Anchor')
    ax.scatter(*route[0],marker='s',s=30,color='#444444');ax.text(.06,.01,'Depot',fontsize=7)
    ax.scatter(.48,.49,marker='s',s=30,color='#41966b');ax.text(.42,.45,'Charger',fontsize=7)
    uav=np.array([.57,.65]);ax.scatter(*uav,marker='X',s=65,color='#b44142',zorder=5)
    for i,a in enumerate(anchors):ax.plot([uav[0],a[0]],[uav[1],a[1]],'--' if i==1 else ':',color='#b44142' if i==1 else '#226b82',lw=.8)
    ax.annotate('',xy=anchors[2],xytext=uav,arrowprops=dict(arrowstyle='->',color='#41966b',lw=1.6))
    ax.text(.59,.62,'UAV',fontsize=8);ax.text(.18,.76,'erased',fontsize=7,color='#b44142');ax.set_xlim(0,1.05);ax.set_ylim(-.03,1.05);ax.set_aspect('equal');ax.axis('off');ax.set_title('(a) Service geometry (top view)',fontsize=9)
    ax=axes[1]; colors=['#226b82','#75a7b5']
    for i in range(3):
        start=i*.1
        ax.add_patch(Rectangle((start,.55),.04,.2,color=colors[0]));ax.add_patch(Rectangle((start+.04,.55),.06,.2,color=colors[1]))
        ax.text(start+.05,.81,f'Link {i+1}',ha='center',fontsize=7)
    ax.add_patch(Rectangle((.3,.55),.04,.2,color='#777777'));ax.text(.32,.49,'Frame\noverhead',ha='center',va='top',fontsize=7)
    for i in range(4):
        start=.34+i*.13
        ax.add_patch(Rectangle((start,.55),.12,.2,color='#41966b'));ax.add_patch(Rectangle((start+.12,.55),.01,.2,color='#c68a32'))
    ax.text(.65,.82,'Data + ACKs',ha='center',fontsize=8)
    ax.text(.01,.24,'Each link: 40 µs control + 60 µs ranging\nFrame: 1 ms; overhead: 40 µs\nRate dictionary and ACK time determine packet count',fontsize=7,va='top')
    ax.set_xlim(0,1.);ax.set_ylim(0,1.);ax.set_yticks([]);ax.set_xticks([0,.3,.34,1.],[0,300,340,1000]);ax.set_xlabel('Time (µs)',fontsize=8);ax.set_title('(b) Shared frame (illustrative)',fontsize=9)
    ax=axes[2]
    for x,angle in ((0,10),(1.2,60),(2.4,25)):
        ax.add_patch(Ellipse((x,0),1.,1.,fill=False,edgecolor='#226b82',ls='--',lw=1.))
        ax.add_patch(Ellipse((x,0),.72,.28,angle=angle,facecolor='#75a7b5',edgecolor='#226b82',alpha=.6))
    ax.text(1.2,.71,'P⁺ stays inside the design cap',ha='center',fontsize=8)
    ax.text(1.2,-.78,'No covariance reset in the filter\nCausal allocation retains P⁺\nGRE cap differs from the isotropic E-opt cap',ha='center',fontsize=7)
    ax.set_xlim(-.7,3.);ax.set_ylim(-1.2,1.);ax.set_aspect('equal');ax.set_xticks([0,1.2,2.4],['Epoch 1','Epoch 2','Epoch 3'],fontsize=7);ax.set_yticks([]);ax.set_title('(c) Covariance induction',fontsize=9)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.savefig(figures/'system_geometry_frame.pdf',bbox_inches='tight');fig.savefig(figures/'system_geometry_frame.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=('main','patterns','propagation','sensitivity','scalability','reports','all'),default='all')
    p.add_argument('--output',type=Path,default=Path('results'));p.add_argument('--main-instances',type=int,default=500)
    p.add_argument('--instances',type=int,default=200);p.add_argument('--replications',type=int,default=128)
    p.add_argument('--solver-instances',type=int,default=20);p.add_argument('--seed',type=int,default=930001)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    suites=dict(main=main_suite,patterns=patterns_suite,propagation=propagation_suite,sensitivity=sensitivity_suite,scalability=scalability_suite)
    if args.suite=='reports':reports(args.output);return
    chosen=list(suites) if args.suite=='all' else [args.suite]
    for name in chosen:
        start=time.perf_counter();suites[name](args,args.output)
        metadata=dict(suite=name,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            runtime_s=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            config=asdict(research_config()),sources={file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
                for file in ('secure_drop_research.py','wireless_extensions.py','risk_pattern_extensions.py','revision_experiments.py','final_revision_experiments.py')})
        (args.output/f'final_{name}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.suite=='all':reports(args.output)


if __name__=='__main__':main()
