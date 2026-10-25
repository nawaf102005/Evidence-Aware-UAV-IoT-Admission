"""Revision 4: route-specific risk budgets, loss-law tests, and pruned verification.

Legacy numerical modules remain unchanged. A certificate is checked independently
of how a candidate codeword is obtained; E-opt feasibility is not GRE feasibility.
"""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import asdict,dataclass

import numpy as np

from secure_drop_research import Config,RadioTable,inv2
from wireless_extensions import (ExtendedRadio,extended_plan,adaptive_plan,
    gaussian_envelope_cap,bler_normal_approx,radial_tail_bounds,data_schedule,
    poisson_binomial_over_budget)
from risk_pattern_extensions import (PatternRadio,pattern_plan,split_plan,
    pattern_probabilities,markov_pattern_probabilities,loss_patterns)


@dataclass(frozen=True)
class HorizonConfig(Config):
    horizon: int=0
    @property
    def epochs_max(self):
        return self.horizon if self.horizon else self.stops+2


@dataclass
class Design:
    plan: object
    cfg: Config
    tables: list
    radios: list
    local_risk: float=.1
    packet_risk: float=0.


def scoped_radios(base_tables,cfg,horizon,mission_risk=None,patterns=False):
    fields=asdict(cfg);fields.pop('horizon',None)
    if mission_risk is not None:fields['mission_risk']=mission_risk
    scoped=HorizonConfig(**fields,horizon=horizon)
    tables=[]
    for base in base_tables:
        table=copy.copy(base);table.cfg=scoped;tables.append(table)
    radios=[PatternRadio(table) if patterns else ExtendedRadio(table) for table in tables]
    return scoped,tables,radios


class MissionDesigner:
    def __init__(self,routes,tables,cfg):
        self.routes,self.tables,self.cfg=routes,tables,cfg
        self.cache={}
    def family(self,horizon,local_risk=.1,patterns=False):
        key=(horizon,local_risk,patterns)
        if key not in self.cache:
            self.cache[key]=scoped_radios(self.tables,self.cfg,horizon,local_risk,patterns)
        return self.cache[key]
    def plan(self,method,local_risk=.05,route_horizon=True):
        best=None
        horizons=sorted({len(r.events) for r in self.routes}) if route_horizon else [self.cfg.epochs_max]
        for horizon in horizons:
            routes=[r for r in self.routes if len(r.events)==horizon] if route_horizon else self.routes
            is_pattern=method=='Pattern-GRE'
            local=local_risk if method=='PHY-GRE' else .1
            cfg,tables,radios=self.family(horizon,local,is_pattern)
            if method=='Pattern-GRE':plan=pattern_plan(routes,tables,radios,cfg)
            elif method=='PHY-GRE':plan=split_plan(routes,tables,radios,cfg,.1-local)
            elif method=='GRE-shortest':
                # Select the globally shortest catalog route before its radio
                # design; do not choose one shortest route per horizon.
                shortest=min(self.routes,key=lambda r:r.lengths.sum()) if self.routes else None
                if shortest is None or (route_horizon and len(shortest.events)!=horizon):continue
                plan=extended_plan('GRE-joint',[shortest],tables,radios,cfg)
            else:plan=extended_plan(method,routes,tables,radios,cfg)
            if plan is not None and (best is None or plan.objective<best.plan.objective):
                best=Design(plan,cfg,tables,radios,local,.1-local if method=='PHY-GRE' else 0.)
        return best
    def causal(self,base,pattern=False):
        if base is None:return None
        if pattern:
            # Actual covariance is evaluated per received-link history later.
            return Design(base.plan,base.cfg,base.tables,base.radios)
        plan=adaptive_plan(base.plan,'Adaptive-GRE',base.radios,base.tables,base.cfg)
        return Design(plan,base.cfg,base.tables,base.radios)
    def optimized_split(self,grid=(.02,.04,.05,.06,.08)):
        candidates=[self.plan('PHY-GRE',local_risk=alpha) for alpha in grid]
        feasible=[x for x in candidates if x is not None]
        return min(feasible,key=lambda x:x.plan.objective) if feasible else None


def pattern_law(p,mode,burst=.03,initial_state=0):
    """Joint law over active links, with no Gaussian-noise conditioning."""
    masks=loss_patterns(len(p))
    if mode=='doubled':return masks,pattern_probabilities(np.minimum(2*p,1),masks)
    probabilities=pattern_probabilities(p,masks)
    if mode=='burst':
        probabilities*=1-burst;probabilities[-1]+=burst
    elif mode=='markov':
        initial=np.eye(2)[initial_state]
        probabilities=markov_pattern_probabilities(p,.8+.2*p,np.array([[.97,.03],[.2,.8]]),initial,masks)
    elif mode!='independent':raise ValueError(mode)
    return masks,probabilities


def evaluate_design(instance,design,method,samples=128,loss_mode='worst',burst=.03):
    """Paired affine-KF trajectories including true loss-law stress.

    Assumed and true conditional pattern tails are recorded separately. A
    misspecified packet law does not cause a covariance-reset shortcut.
    """
    plan,cfg,tables,extra=design.plan,design.cfg,design.tables,design.radios
    pattern=method in ('Pattern-GRE','Causal-Pattern-GRE')
    causal=method in ('Causal-GRE','Causal-Pattern-GRE')
    gaussian=method in ('GRE','Causal-GRE','GRE-shortest','PHY-GRE','Optimized-PHY-GRE')
    epsilon=cfg.mission_risk/cfg.epochs_max
    rng=np.random.default_rng(instance.seed+90_000_000)
    errors=cfg.initial_sd_m*rng.normal(size=(samples,2))
    process=rng.normal(size=(cfg.stops+2,samples,2));noise=rng.normal(size=(cfg.stops+2,samples,cfg.anchors))
    lrng=np.random.default_rng(instance.seed+91_000_000)
    uniforms=lrng.random((cfg.stops+2,samples,cfg.anchors));bursts=lrng.random((cfg.stops+2,samples))
    max_packets=int(np.ceil(cfg.frame_s/(320/cfg.bandwidth_hz+10e-6)))
    data_uniforms=lrng.random((cfg.stops+2,samples,max_packets))
    mrng=np.random.default_rng(instance.seed+93_000_000)
    transitions=mrng.random((cfg.stops+2,samples,cfg.anchors))
    states=(mrng.random(samples)<(.03/.23)).astype(int)
    P=np.tile(cfg.initial_sd_m**2*np.eye(2),(samples,1,1))
    failed=np.zeros(samples,bool);over=np.zeros(samples,bool);unresolved=np.zeros(samples,bool)
    radial=np.zeros(samples);assumed=np.zeros(samples);actual=np.zeros(samples)
    pilots=payload=energy=cost=nees=0.;decision_ms=[];over_epochs=0
    for node,length,offline in zip(plan.route.events,plan.route.lengths,plan.schedules):
        Q=cfg.drift_m2_per_m*length;errors+=np.sqrt(Q)*process[node];P+=Q*np.eye(2)
        indices=np.full(samples,offline,int)
        if causal:
            rounded=np.round(P.reshape(samples,4),12)
            for values in np.unique(rounded,axis=0):
                members=np.all(rounded==values,axis=1);clock=time.perf_counter()
                selected=extra[node].choose_pattern(values.reshape(2,2)) if pattern else extra[node].choose(values.reshape(2,2),'Gaussian')
                decision_ms.append(1000*(time.perf_counter()-clock))
                if selected is None:
                    unresolved[members]=True
                else:indices[members]=selected
        table=tables[node];levels=table.codebook[indices];active=levels>0
        gamma=table.sinr[np.arange(cfg.anchors)[None,:],levels]
        model_p=.02+.98*bler_normal_approx(gamma,round(cfg.pilot_control_s*cfg.bandwidth_hz),48)
        if pattern:
            for index in np.unique(indices):
                members=indices==index;anchors=np.flatnonzero(table.codebook[index])
                masks=loss_patterns(len(anchors));p=extra[node].probabilities[index,anchors]
                parts=extra[node].parts[index,anchors]
                info=np.einsum('em,mij->eij',~masks,parts)
                post=inv2(inv2(P[members])[:,None]+info)
                _,upper=radial_tail_bounds(post,cfg.radius_m,256)
                assumed_probability=pattern_probabilities(p,masks)
                nominal=upper@assumed_probability
                if np.any(nominal>epsilon+1e-9):raise RuntimeError('Assumed-law conditional certificate failed.')
                assumed[members]+=nominal
                if loss_mode=='markov':
                    which=np.flatnonzero(members)
                    for state in (0,1):
                        selected=states[which]==state
                        if selected.any():actual[which[selected]]+=upper[selected]@pattern_law(p,'markov',initial_state=state)[1]
                else:
                    actual[members]+=upper@pattern_law(p,loss_mode,burst)[1]
        losses=np.zeros_like(active)
        if loss_mode=='worst':
            for index in np.unique(indices):
                selected=indices==index
                if causal:prior=P[np.flatnonzero(selected)[0]]
                else:prior=(gaussian_envelope_cap(cfg) if gaussian else cfg.covariance_cap)+Q;prior=prior*np.eye(2)
                erased=extra[node].erased_for(int(index),prior,gaussian)
                losses[np.ix_(np.flatnonzero(selected),erased)]=True
        elif loss_mode=='markov':
            # Hidden state persists across actual scheduled slots and epochs.
            # No transition is consumed for an inactive candidate anchor.
            for m in range(cfg.anchors):
                scheduled=active[:,m]
                probability=np.where(states==0,model_p[:,m],.8+.2*model_p[:,m])
                losses[:,m]=scheduled & (uniforms[node,:,m]<probability)
                thresholds=np.where(states==0,.03,.2)
                change=scheduled & (transitions[node,:,m]<thresholds)
                states[change]=1-states[change]
        else:
            p=np.minimum(2*model_p,1) if loss_mode=='doubled' else model_p
            losses=(uniforms[node]<p)&active
            if loss_mode=='burst':losses|=(bursts[node,:,None]<burst)&active
        losses &=active;lost=losses.sum(axis=1);over|=lost>cfg.erasure_budget
        over_epochs+=int((lost>cfg.erasure_budget).sum())
        received=active&~losses
        variance=table.variance[np.arange(cfg.anchors)[None,:],levels]
        for mask in np.unique(received,axis=0):
            members=np.all(received==mask,axis=1);anchors=np.flatnonzero(mask)
            if not len(anchors):continue
            H=table.H[anchors];R=variance[members][:,anchors]
            info=np.einsum('ni,ij,ik->njk',1/R,H,H);post=inv2(inv2(P[members])+info)
            K=np.einsum('nij,jk,nk->nik',post,H.T,1/R);A=np.eye(2)-np.einsum('nij,jk->nik',K,H)
            errors[members]=np.einsum('nij,nj->ni',A,errors[members])-np.einsum('nij,nj->ni',K,noise[node][members][:,anchors]*np.sqrt(R))
            P[members]=post
        failed|=(errors**2).sum(axis=1)>cfg.radius_m**2
        nees+=float(np.einsum('ni,nij,nj->n',errors,inv2(P),errors).mean())
        radial+=radial_tail_bounds(P,cfg.radius_m,256)[1]
        pilots+=float(table.count[indices].mean()*cfg.pilot_s/cfg.frame_s)
        energy+=float(table.energy[indices].mean());cost+=float(table.cost[indices].mean())
        if 1<=node<=cfg.stops:
            data_gamma=cfg.data_power_w*instance.gains[node].max()/(cfg.noise_w+instance.interference[node])
            symbols,packets,_=data_schedule(table.count[indices],data_gamma,cfg)
            success=1-bler_normal_approx(data_gamma,symbols,608)
            payload+=float((((np.arange(max_packets)[None,:]<packets[:,None])&(data_uniforms[node]<success[:,None])).sum(axis=1)*512).mean())
    epochs=len(plan.route.events);failed|=unresolved
    return dict(violation=float(failed.mean()),pilot_fraction=pilots/epochs,goodput_bits=payload,
        pilot_energy_mj=1000*energy,radio_cost=cost,nees=nees/epochs,
        radial_union_bound=float(np.minimum(radial,1).mean()),
        assumed_conditional_union=float(assumed.mean()) if pattern else np.nan,
        max_assumed_conditional_union=float(assumed.max()) if pattern else np.nan,
        true_conditional_union=float(actual.mean()) if pattern else np.nan,
        budget_exceed_mission=float(over.mean()),budget_exceed_epoch=over_epochs/(epochs*samples),
        unresolved=float(unresolved.mean()),decision_ms_mean=float(np.mean(decision_ms)) if decision_ms else 0.)


def pruned_pattern_bounds(prior,parts,probabilities,radius,cutoff,terms=256):
    """Retain <= cutoff losses, assign the exact residual mass risk one.

    Only active links are passed. No full 2^M allocation is constructed.
    The omitted-mass term is the stable Poisson-binomial tail.
    """
    p=np.asarray(probabilities);parts=np.asarray(parts);M=len(p)
    if cutoff<0 or not 0<=radius or np.any((p<0)|(p>1)):raise ValueError('Invalid pattern cutoff/law.')
    masks=[]
    for count in range(min(cutoff,M)+1):
        for erased in itertools.combinations(range(M),count):
            mask=np.zeros(M,bool);mask[list(erased)]=True;masks.append(mask)
    masks=np.asarray(masks);weights=pattern_probabilities(p,masks)
    info=np.einsum('em,mij->eij',~masks,parts)
    covariance=inv2(inv2(prior)+info);lower,upper=radial_tail_bounds(covariance,radius,terms)
    residual=float(poisson_binomial_over_budget(p,min(cutoff,M)))
    return dict(lower=float(weights@lower),upper=min(1.,float(weights@upper)+residual),
        residual_mass=residual,patterns=len(masks))


def codeword_atoms(instance,node,levels,cfg):
    powers=np.asarray(cfg.powers_w)[levels];active=np.flatnonzero(powers)
    delta=instance.nodes[node]-instance.anchors
    slant=np.sqrt((delta**2).sum(axis=1)+cfg.altitude_m**2);H=delta/slant[:,None]
    gamma=powers[active]*instance.gains[node,active]/(cfg.noise_w+instance.interference[node])
    beta=cfg.bandwidth_hz/np.sqrt(12)
    variance=cfg.range_floor_m**2+cfg.ranging_efficiency*299792458.**2/(8*np.pi**2*beta**2*gamma*cfg.bandwidth_hz*(cfg.pilot_s-cfg.pilot_control_s))
    atoms=H[active,:,None]*H[active,None,:]/variance[:,None,None]
    p=.02+.98*bler_normal_approx(gamma,round(cfg.pilot_control_s*cfg.bandwidth_hz),48)
    return atoms,p
