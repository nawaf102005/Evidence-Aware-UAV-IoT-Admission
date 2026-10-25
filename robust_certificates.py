"""Revision 5: loss-aware radio design and matched certification comparisons.

Historical numerical sources stay unchanged. Repetitions retain the first
successfully decoded range per anchor: extra ranges are deliberately discarded.
This makes covariance and loss weighting exact for the declared affine model.
"""
from __future__ import annotations
import copy, itertools, time
from dataclasses import dataclass, replace
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import beta
from secure_drop_research import inv2, feasible_routes,RadioTable
from wireless_extensions import (ExtendedRadio, bler_normal_approx, data_schedule,
                                expected_payload, radial_tail_bounds)
from risk_pattern_extensions import loss_patterns, pattern_probabilities

NOMINAL_T=np.array([[.97,.03],[.20,.80]])
ROBUST_T=np.array([[.96,.04],[.15,.85]])

def projection_cap(radius,target):
    if radius<=0 or not 0<target<1 or target/2==0:raise ValueError('Invalid representable radius-risk budget.')
    return float(radius**2/ndtri(target/2)**2)

def log_radial_bounds(P,radius,terms=256):
    """Left/right endpoint sums in log space, without lower-bound cancellation."""
    P=np.asarray(P);r=np.asarray(radius)
    if terms<1 or np.any(r<0):raise ValueError('Invalid radius/quadrature.')
    tr=P[...,0,0]+P[...,1,1];disc=np.hypot(P[...,0,0]-P[...,1,1],2*P[...,0,1])
    eigen=np.stack(((tr-disc)/2,(tr+disc)/2),axis=-1)
    if np.any(eigen[...,0]<=0):raise ValueError('Covariance must be positive definite.')
    angle=np.arange(terms+1)*np.pi/(2*terms)
    variance=eigen[...,0,None]+(eigen[...,1]-eigen[...,0])[...,None]*np.cos(angle)**2
    logs=-r[...,None]**2/(2*variance)
    lower=logs[...,1]+np.log(np.exp(logs[...,1:]-logs[...,1,None]).mean(axis=-1))
    upper=logs[...,0]+np.log(np.exp(logs[...,:-1]-logs[...,0,None]).mean(axis=-1))
    return lower,upper

def radius_after_bias(P,bias_energy,radius):
    """Mahalanobis mean budget -> a rigorous centered-radius sufficient test."""
    largest=np.linalg.eigvalsh(P)[...,-1]
    return np.maximum(0.,radius-np.sqrt(largest*np.asarray(bias_energy)))

def markov_available_joint(good,bad,active,repetitions=1,transition=NOMINAL_T,return_all=False):
    """Exact availability/terminal-state law for both pure starting states.

    Return [candidate, loss-pattern, starting-state, terminal-state]. A DP
    tracks successful-anchor bits through round-robin repetitions, not all
    2^(q*n) individual packet outcomes. State transitions follow scheduled slots.
    """
    good=np.atleast_2d(good);bad=np.atleast_2d(bad);active=np.atleast_2d(active).astype(bool)
    T=np.asarray(transition)
    if good.shape!=bad.shape or good.shape!=active.shape or repetitions<1:
        raise ValueError('Invalid availability dimensions.')
    if (np.any(good<0) or np.any(bad>1) or np.any(good>1) or np.any(bad<0)
            or T.shape!=(2,2) or np.any(T<0) or not np.allclose(T.sum(1),1)):
        raise ValueError('Invalid Markov law.')
    count,M=good.shape;bits=1<<np.arange(M);states=1<<M
    joint=np.zeros((count,states,2,2))
    initial=np.sum((~active)*bits,axis=1);idx=np.arange(count)
    joint[idx,initial,0,0]=1.;joint[idx,initial,1,1]=1.
    masks=loss_patterns(M);availability=(~masks).astype(int)@bits;history={}
    for repetition in range(repetitions):
        for m in range(M):
            zero=np.flatnonzero((np.arange(states)&bits[m])==0);one=zero|bits[m]
            probability=np.stack((good[:,m],bad[:,m]),axis=-1)[:,None,None,:]
            update=joint.copy()
            update[:,zero]=joint[:,zero]*probability
            update[:,one]=joint[:,one]+joint[:,zero]*(1-probability)
            d0=update[...,0].copy();d1=update[...,1].copy()
            update[...,0]=d0*T[0,0]+d1*T[1,0]
            update[...,1]=d0*T[0,1]+d1*T[1,1]
            joint=np.where(active[:,m,None,None,None],update,joint)
        if return_all:history[repetition+1]=joint[:,availability]
    # Inactive anchors start with their bit set, matching zero erasure mass.
    return history if return_all else joint[:,availability]

def make_tables(instance,cfg,codebook):
    tables=[RadioTable(instance,n,cfg,codebook) for n in range(cfg.stops+2)]
    for node,table in enumerate(tables):
        gamma=cfg.data_power_w*instance.gains[node].max()/(cfg.noise_w+instance.interference[node])
        table.bits=expected_payload(table.count,gamma,cfg)
        table.cost=table.count*(cfg.pilot_s/1e-4)+cfg.pilot_energy_price_per_j*table.energy
        table.sorted_indices=np.argsort(table.cost,kind='stable')
    return tables

def initial_mixture_joint(joint,belief):
    return np.einsum('pej,e->pj',joint,np.asarray(belief))

def update_belief(good,bad,active,repetitions,received,belief,transition=NOMINAL_T):
    """Filter hidden channel state using all observed packet indicators."""
    belief=np.asarray(belief,dtype=float).copy()
    for q in range(repetitions):
        for m in np.flatnonzero(active):
            p=np.array([good[m],bad[m]])
            belief*=1-p if received[q,m] else p
            total=belief.sum()
            if total<=0:raise ValueError('Zero-probability observation.')
            belief=(belief/total)@transition
    return belief

@dataclass(frozen=True)
class Action:
    index:int
    repetitions:int=1

@dataclass
class Allocation:
    cfg:object
    route:object
    actions:tuple
    tables:list
    radios:list
    mode:str
    objective:float
    bias_bound:float=0.
    lower_radio_cost:float=float('nan')

class CertificateRadio(ExtendedRadio):
    def __init__(self,table):
        super().__init__(table)
        self.trace_thresholds={}
    def choose_cert(self,prior,target,mode='GRE',bias_energy=0.):
        key=(mode,target,bias_energy,*prior.ravel())
        if key in self.cache:return self.cache[key]
        cfg,table=self.table.cfg,self.table
        candidates=table.sorted_indices[table.bits[table.sorted_indices]>=cfg.data_min_bits]
        precision=inv2(prior);frontier=None
        best_post=np.stack([inv2(precision+S[-1]) for S in self.surviving])
        if mode=='SPEB':
            nu=cfg.radius_m**2/(-2*np.log(target))
            impossible=np.trace(best_post,axis1=-2,axis2=-1).max()>2*nu*(1-1e-10)
        else:
            rr=radius_after_bias(best_post,bias_energy,cfg.radius_m) if bias_energy else cfg.radius_m
            impossible=log_radial_bounds(best_post,rr,32)[0].max()>np.log(target)
        if impossible:self.cache[key]=None;return None
        if mode in ('SPEB','Certified-SPEB'):
            worst=np.maximum.reduce([np.trace(inv2(precision+S),axis1=-2,axis2=-1) for S in self.surviving])
            if mode=='SPEB':
                nu=cfg.radius_m**2/(-2*np.log(target))
                indices=candidates[worst[candidates]<=2*nu*(1-1e-10)]
                answer=Action(int(indices[0])) if len(indices) else None
                self.cache[key]=answer;return answer
            # Exactly the candidates selected by some scalar SPEB threshold,
            # with the common cost/tie ordering. Each is checked by GRE.
            values=worst[candidates];previous=np.r_[np.inf,np.minimum.accumulate(values)[:-1]]
            candidates=candidates[values<previous]
            frontier=worst
        answer=None
        for begin in range(0,len(candidates),64):
            chunk=candidates[begin:begin+64]
            posterior=np.stack([inv2(precision+S[chunk]) for S in self.surviving],axis=1)
            radii=radius_after_bias(posterior,bias_energy,cfg.radius_m) if bias_energy else cfg.radius_m
            unresolved=np.arange(len(chunk))
            for terms in (32,128,512,2048):
                rr=np.asarray(radii)[unresolved] if np.ndim(radii) else radii
                lower,upper=log_radial_bounds(posterior[unresolved],rr,terms)
                lo=lower.max(axis=-1);up=upper.max(axis=-1)
                passing=up<=np.log(target)+np.log1p(-1e-10)
                # A cheaper ambiguous codeword must be refined before taking
                # a more expensive passing one, to preserve restricted exactness.
                if np.any(passing):
                    first=np.flatnonzero(passing)[0]
                    if terms==2048 or np.all(lo[:first]>np.log(target)):
                        index=int(chunk[unresolved[first]]);answer=Action(index)
                        if frontier is not None:self.trace_thresholds[key]=float(frontier[index])
                        break
                unresolved=unresolved[lo<=np.log(target)]
                if not len(unresolved):break
            if answer is not None:break
        self.cache[key]=answer;return answer

class BurstRadio(CertificateRadio):
    def __init__(self,table,law='nominal',repetitions=(1,2,3)):
        super().__init__(table)
        self.law=law;self.repetitions=repetitions;self.patterns=loss_patterns(table.cfg.anchors)
        self.weights={};self.costs={};self.energies={};self.counts={};self.bits={}
        gamma=table.sinr[np.arange(table.cfg.anchors)[None,:],table.codebook]
        p=.02+.98*bler_normal_approx(gamma,round(table.cfg.bandwidth_hz*table.cfg.pilot_control_s),48)
        active=table.codebook>0;self.good=p
        self.bad=.8+.2*p;self.transition=NOMINAL_T
        if law=='robust':
            p=np.minimum(1.5*p,1.);self.bad=.85+.15*p;self.transition=ROBUST_T
        self.emission_good=p;self.active=active
        for q in repetitions:
            if law=='independent':
                weights=pattern_probabilities(np.where(active,p**q,0),self.patterns)
                self.weights[q]=weights[...,None]
            else:
                self.weights[q]=np.full((len(table.codebook),len(self.patterns),2),np.nan)
            self.costs[q]=q*table.cost;self.energies[q]=q*table.energy;self.counts[q]=q*table.count
            # Same coded-payload dictionary, with every repeated slot charged.
            gamma_d=table.rate/table.cfg.bandwidth_hz
            gamma_d=np.expm1(gamma_d*np.log(2))
            self.bits[q]=expected_payload(self.counts[q],gamma_d,table.cfg)
        self.active_keys=active@(1<<np.arange(table.cfg.anchors))
        peaks=np.flatnonzero(np.all((table.codebook==0)|(table.codebook==len(table.cfg.powers_w)-1),axis=1))
        self.peaks=peaks;self.peak_for={int(self.active_keys[i]):int(i) for i in peaks}
        self.ordered_actions=sorted((float(self.costs[q][i]),q,int(i)) for q in repetitions
                                   for i in table.sorted_indices if self.bits[q][i]>=table.cfg.data_min_bits)
        self.lower_bounds={}
    def weights_for(self,q,indices):
        indices=np.atleast_1d(indices).astype(int)
        missing=np.unique(indices[np.isnan(self.weights[q][indices,0,0])])
        if len(missing):
            self.weights[q][missing]=markov_available_joint(self.emission_good[missing],self.bad[missing],self.active[missing],q,self.transition).sum(axis=-1)
        return self.weights[q][indices]
    def conditional_weights(self,q,index,belief=None):
        w=self.weights_for(q,[index])[0]
        return w if belief is None or self.law=='independent' else (w@belief)[...,None]
    def risk_bounds(self,prior,action,target=None,belief=None,bias_energy=0.,terms=256):
        lower,upper=self.risk_bounds_many(np.asarray(prior)[None],action,
            None if belief is None else np.asarray(belief)[None],np.asarray([bias_energy]),terms)
        return float(lower[0]),float(upper[0])
    def risk_bounds_many(self,priors,action,beliefs=None,bias_energy=None,terms=256):
        index,q=action.index,action.repetitions
        weights=self.weights_for(q,[index])[0]
        enabled=weights.sum(axis=-1)>0;patterns=self.patterns[enabled]
        info=np.einsum('em,mij->eij',~patterns,self.parts[index])
        post=inv2(inv2(priors)[:,None]+info)
        radius=radius_after_bias(post,np.asarray(bias_energy)[:,None],self.table.cfg.radius_m) if bias_energy is not None and np.any(bias_energy) else self.table.cfg.radius_m
        lower,upper=log_radial_bounds(post,radius,terms)
        w=weights[enabled][None] if beliefs is None or self.law=='independent' else np.einsum('ei,ni->ne',weights[enabled],beliefs)[...,None]
        with np.errstate(divide='ignore'):lw=np.log(w)
        return np.exp(logsumexp(lower[...,None]+lw,axis=1)).max(axis=-1),np.exp(logsumexp(upper[...,None]+lw,axis=1)).max(axis=-1)
    def choose_burst(self,prior,target,belief=None,bias_energy=0.):
        key=('burst',target,bias_energy,None if belief is None else tuple(belief),*prior.ravel())
        if key in self.cache:return self.cache[key]
        cfg,table=self.table.cfg,self.table
        possible={}
        for q in self.repetitions:
            peaks=self.peaks[self.bits[q][self.peaks]>=cfg.data_min_bits]
            weights=self.weights_for(q,peaks)
            if belief is not None and self.law!='independent':weights=(weights@belief)[...,None]
            enabled=np.any(weights.sum(axis=-1)>0,axis=0)
            info=np.einsum('em,cmij->ceij',~self.patterns[enabled],self.parts[peaks])
            post=inv2(inv2(prior)+info)
            radii=radius_after_bias(post,bias_energy,cfg.radius_m) if bias_energy else cfg.radius_m
            lower,_=log_radial_bounds(post,radii,32)
            with np.errstate(divide='ignore'):lw=np.log(weights[:,enabled])
            lower=logsumexp(lower[...,None]+lw,axis=1).max(axis=-1)
            possible[q]=set(self.active_keys[peaks[lower<=np.log(target)]].tolist())
        # Power monotonicity holds for a FIXED active set/slot order: the
        # peak-power word has smaller noise and no larger loss emissions.
        # No monotonicity across different slot schedules is assumed.
        all_actions=[a for a in self.ordered_actions if int(self.active_keys[a[2]]) in possible[a[1]]]
        answer=None
        # Group equal-q consecutive blocks without changing global cost order.
        position=0
        while position<len(all_actions):
            q=all_actions[position][1];end=position+1
            while end<min(position+24,len(all_actions)) and all_actions[end][1]==q:end+=1
            chunk=np.array([x[2] for x in all_actions[position:end]])
            weights=self.weights_for(q,chunk)
            if belief is not None and self.law!='independent':weights=(weights@belief)[...,None]
            enabled=np.any(weights.sum(axis=-1)>0,axis=0)
            info=np.einsum('em,cmij->ceij',~self.patterns[enabled],self.parts[chunk])
            post=inv2(inv2(prior)+info)
            radii=radius_after_bias(post,bias_energy,cfg.radius_m) if bias_energy else cfg.radius_m
            with np.errstate(divide='ignore'):lw=np.log(weights[:,enabled])
            pending=np.arange(len(chunk))
            for terms in (16,64,256,1024):
                rr=np.asarray(radii)[pending] if np.ndim(radii) else radii
                lower,upper=log_radial_bounds(post[pending],rr,terms)
                lo=logsumexp(lower[...,None]+lw[pending],axis=1).max(axis=-1)
                up=logsumexp(upper[...,None]+lw[pending],axis=1).max(axis=-1)
                good=up<=np.log(target)+np.log1p(-1e-10)
                if np.any(good):
                    first=np.flatnonzero(good)[0]
                    if terms==1024 or np.all(lo[:first]>np.log(target)):
                        answer=Action(int(chunk[pending[first]]),q);break
                pending=pending[lo<=np.log(target)]
                if not len(pending):break
            if answer is not None:break
            position=end
        self.cache[key]=answer;return answer

    def candidate_bounds(self,prior,q,indices,belief=None,bias_energy=0.,terms=256):
        weights=self.weights_for(q,indices)
        if belief is not None and self.law!='independent':weights=(weights@belief)[...,None]
        enabled=np.any(weights.sum(axis=-1)>0,axis=0)
        info=np.einsum('em,cmij->ceij',~self.patterns[enabled],self.parts[indices])
        post=inv2(inv2(prior)+info)
        radius=radius_after_bias(post,bias_energy,self.table.cfg.radius_m) if bias_energy else self.table.cfg.radius_m
        lower,upper=log_radial_bounds(post,radius,terms)
        with np.errstate(divide='ignore'):lw=np.log(weights[:,enabled])
        return np.exp(logsumexp(lower[...,None]+lw,axis=1)).max(axis=-1),np.exp(logsumexp(upper[...,None]+lw,axis=1)).max(axis=-1)

    def choose_greedy(self,prior,target,belief=None,bias_energy=0.):
        """Peak active-set screening, then checked local power/repetition moves.

        Optimistic lower brackets plus minimum positive power give a valid
        cost lower bound. Failed local search is never called infeasibility.
        """
        key=('greedy',target,bias_energy,None if belief is None else tuple(belief),*prior.ravel())
        if key in self.cache:
            self.last_lower_bound=self.lower_bounds[key]
            return self.cache[key]
        table,cfg=self.table,self.table.cfg;best=None;lower_cost=np.inf
        for q in self.repetitions:
            peaks=self.peaks[self.bits[q][self.peaks]>=cfg.data_min_bits]
            if not len(peaks):continue
            lower,upper=self.candidate_bounds(prior,q,peaks,belief,bias_energy,64)
            ambiguous=(lower<=target)&(upper>target*(1-1e-10))
            if np.any(ambiguous):
                l,u=self.candidate_bounds(prior,q,peaks[ambiguous],belief,bias_energy,1024)
                lower[ambiguous]=l;upper[ambiguous]=u
            possible=peaks[lower<=target]
            if len(possible):
                counts=q*table.count[possible]
                bound=counts*(cfg.pilot_s/1e-4)+cfg.pilot_energy_price_per_j*counts*cfg.pilot_s*min(x for x in cfg.powers_w if x>0)
                lower_cost=min(lower_cost,float(bound.min()))
            for index in peaks[upper<=target*(1-1e-10)]:
                action=Action(int(index),q)
                if best is None or self.costs[q][index]<self.costs[best.repetitions][best.index]:best=action
        if best is not None:
            encoding=len(cfg.powers_w)**np.arange(cfg.anchors-1,-1,-1)
            while True:
                neighbors=[];levels=table.codebook[best.index]
                for m in np.flatnonzero(levels):
                    new=levels.copy();new[m]-=1;index=int(new@encoding)
                    assert np.array_equal(table.codebook[index],new)
                    neighbors.append(Action(index,best.repetitions))
                neighbors += [Action(best.index,q) for q in self.repetitions if q<best.repetitions]
                neighbors=sorted((x for x in neighbors if self.bits[x.repetitions][x.index]>=cfg.data_min_bits),key=lambda x:self.costs[x.repetitions][x.index])
                improvement=None
                for action in neighbors:
                    if self.risk_bounds(prior,action,belief=belief,bias_energy=bias_energy,terms=1024)[1]<=target*(1-1e-10):
                        improvement=action;break
                if improvement is None:break
                best=improvement
        self.cache[key]=best;self.lower_bounds[key]=lower_cost;self.last_lower_bound=lower_cost
        return best

def make_allocation(cfg,route,actions,tables,radios,mode,bias_bound=0.):
    cost=0.
    for node,a in zip(route.events,actions):
        cost+=a.repetitions*tables[node].cost[a.index]
    objective=route.flight_j/cfg.energy_scale_j+cfg.time_weight*route.completion_s/cfg.time_scale_s+cfg.radio_weight*cost
    return Allocation(cfg,route,tuple(actions),tables,radios,mode,float(objective),bias_bound)

def plan_certified(instance,cfg,tables,mode='GRE',radios=None,bias_bound=0.,repetitions=(1,2,3),routes=None,greedy=True):
    radios=radios or [CertificateRadio(t) if mode in ('GRE','SPEB','Certified-SPEB') else
                     BurstRadio(t,{'Independent':'independent','Markov':'nominal','Robust-Markov':'robust'}[mode],repetitions) for t in tables]
    best=None
    for route in (feasible_routes(instance,cfg) if routes is None else routes):
        target=cfg.mission_risk/len(route.events);cap=projection_cap(cfg.radius_m,target)
        nu=cfg.radius_m**2/(-2*np.log(target));actions=[];cumulative=0.;lower_cost=0.
        for epoch,(node,length) in enumerate(zip(route.events,route.lengths),1):
            cumulative+=length
            if mode in ('GRE','SPEB','Certified-SPEB'):
                base=cfg.initial_sd_m**2 if epoch==1 else nu if mode=='SPEB' else cap
                prior=(base+cfg.drift_m2_per_m*length)*np.eye(2)
                action=radios[node].choose_cert(prior,target,mode)
            else:
                prior=(cfg.initial_sd_m**2+cfg.drift_m2_per_m*cumulative)*np.eye(2)
                energy=epoch*cfg.anchors*(bias_bound/cfg.range_floor_m)**2
                choose=radios[node].choose_greedy if greedy else radios[node].choose_burst
                action=choose(prior,target,bias_energy=energy)
                if greedy:lower_cost+=radios[node].last_lower_bound
            if action is None:break
            actions.append(action)
        if len(actions)==len(route.events):
            candidate=make_allocation(cfg,route,actions,tables,radios,mode,bias_bound)
            if mode not in ('GRE','SPEB','Certified-SPEB') and greedy:candidate.lower_radio_cost=lower_cost
            if best is None or candidate.objective<best.objective:best=candidate
    return best

def binomial_interval(successes,samples,confidence=.99):
    a=(1-confidence)/2
    lo=0. if successes==0 else beta.ppf(a,successes,samples-successes+1)
    hi=1. if successes==samples else beta.ppf(1-a,successes+1,samples-successes)
    return float(lo),float(hi)

def simulate(allocation,instance,samples=256,causal=False,belief_aware=False,
             loss_mode='bounded',initial='stationary',bias_m=0.,bias_mode='constant',
             nonlinear=False,csi_sd=0.,csi_margin=0.,true_transition=NOMINAL_T,
             true_bad=.8,true_good_multiplier=1.):
    """Estimator-error experiment; hidden states are never revealed to policy.

    Bias can depend on indicator history, never Gaussian residuals. The
    nonlinear and CSI-mismatch runs are empirical stresses without GRE claims.
    """
    a=allocation;cfg=a.cfg;Horizon=len(a.route.events);target=cfg.mission_risk/Horizon
    rng=np.random.default_rng(instance.seed+90_000_000)
    errors=cfg.initial_sd_m*rng.normal(size=(samples,2));mean=np.zeros_like(errors)
    process=rng.normal(size=(cfg.stops+2,samples,2))
    normals=rng.normal(size=(cfg.stops+2,3,samples,cfg.anchors))
    prng=np.random.default_rng(instance.seed+91_000_000)
    packet_u=prng.random((cfg.stops+2,3,samples,cfg.anchors))
    max_packets=int(np.ceil(cfg.frame_s/(320/cfg.bandwidth_hz+10e-6)))
    data_u=prng.random((cfg.stops+2,samples,max_packets))
    mrng=np.random.default_rng(instance.seed+93_000_000)
    transition_u=mrng.random((cfg.stops+2,3,samples,cfg.anchors))
    stationary=true_transition[0,1]/(true_transition[0,1]+true_transition[1,0])
    states=(mrng.random(samples)<stationary).astype(int) if initial=='stationary' else np.full(samples,int(initial))
    beliefs=np.tile([.20/.23,.03/.23],(samples,1))
    map_rng=np.random.default_rng(instance.seed+92_000_000)
    map_db=map_rng.normal(0,csi_sd,(cfg.stops+2,cfg.anchors))-csi_margin
    P=np.tile(cfg.initial_sd_m**2*np.eye(2),(samples,1,1))
    failed=np.zeros(samples,bool);unresolved=np.zeros(samples,bool);over=np.zeros(samples,bool)
    eta=np.zeros(samples);conditional=np.zeros(samples);radial=np.zeros(samples);radial_lower=np.zeros(samples);max_epoch_lower=np.zeros(samples)
    pilots=energy=cost=payload=nees=active_count=attempts=repetition_count=0.;decision=[]
    for epoch,(node,length,offline) in enumerate(zip(a.route.events,a.route.lengths,a.actions),1):
        Q=cfg.drift_m2_per_m*length;errors+=np.sqrt(Q)*process[node];P+=Q*np.eye(2)
        table=a.tables[node];radio=a.radios[node]
        selected=[offline for _ in range(samples)]
        if causal:
            keys=np.c_[P.reshape(samples,4),eta,beliefs] if belief_aware else np.c_[P.reshape(samples,4),eta]
            for key in np.unique(keys,axis=0):
                members=np.all(keys==key,axis=1);prior=key[:4].reshape(2,2);start=time.perf_counter()
                if isinstance(radio,BurstRadio):
                    budget=float(key[4])+cfg.anchors*(a.bias_bound/cfg.range_floor_m)**2
                    act=radio.choose_greedy(prior,target,belief=key[-2:] if belief_aware else None,bias_energy=budget)
                else:act=radio.choose_cert(prior,target,a.mode)
                decision.append(1000*(time.perf_counter()-start))
                if act is None:
                    if isinstance(radio,BurstRadio):
                        valid=radio.risk_bounds(prior,offline,belief=key[-2:] if belief_aware else None,bias_energy=budget,terms=1024)[1]<=target*(1+1e-8)
                    else:
                        posts=np.stack([inv2(inv2(prior)+S[offline.index]) for S in radio.surviving])
                        valid=log_radial_bounds(posts,cfg.radius_m,2048)[1].max()<=np.log(target)+1e-8
                    if not valid:unresolved[members]=True
                    act=offline
                for i in np.flatnonzero(members):selected[i]=act
        levels=np.array([table.codebook[act.index] for act in selected]);qvalues=np.array([act.repetitions for act in selected])
        active=levels>0;received_attempt=np.zeros((3,samples,cfg.anchors),bool)
        gamma=table.sinr[np.arange(cfg.anchors)[None,:],levels]
        p_design=.02+.98*bler_normal_approx(gamma,round(cfg.pilot_control_s*cfg.bandwidth_hz),48)
        p_true=.02+.98*bler_normal_approx(gamma*10**(-map_db[node]/10),round(cfg.pilot_control_s*cfg.bandwidth_hz),48)
        p_true=np.minimum(true_good_multiplier*p_true,1.)
        # Check the actual prior separately from cache/group selection.
        if isinstance(radio,BurstRadio):
            for act in set(selected):
                ids=np.array([i for i,x in enumerate(selected) if x==act])
                budget=eta[ids]+cfg.anchors*(a.bias_bound/cfg.range_floor_m)**2
                bound=radio.risk_bounds_many(P[ids],act,beliefs[ids] if belief_aware else None,budget,256)[1]
                ambiguous=bound>target*(1-1e-10)
                if np.any(ambiguous):
                    bound[ambiguous]=radio.risk_bounds_many(P[ids[ambiguous]],act,beliefs[ids[ambiguous]] if belief_aware else None,budget[ambiguous],1024)[1]
                if np.any(bound>target*(1+1e-8)):raise RuntimeError('Executed conditional certificate failed.')
                conditional[ids]+=bound
        if loss_mode=='bounded':
            for act in set(selected):
                ids=np.array([i for i,x in enumerate(selected) if x==act])
                values,inverse=np.unique(P[ids].reshape(len(ids),4),axis=0,return_inverse=True)
                for j,value in enumerate(values):
                    if a.mode=='Legacy-SPEB':
                        prior=(cfg.covariance_cap+Q)*np.eye(2)
                        erased=radio.erased_for(act.index,prior,False)
                    else:
                        posts=np.stack([inv2(inv2(value.reshape(2,2))+S[act.index]) for S in radio.surviving])
                        tails=np.exp(log_radial_bounds(posts,cfg.radius_m,512)[1])
                        erased=radio.erased[int(np.argmax(tails))]
                    members=ids[inverse==j];received_attempt[0,members]=active[members]
                    received_attempt[0][np.ix_(members,list(erased))]=False
        else:
            for q in range(3):
                for m in range(cfg.anchors):
                    scheduled=active[:,m]&(qvalues>q)
                    probability=np.where(states==0,p_true[:,m],true_bad+(1-true_bad)*p_true[:,m]) if loss_mode=='markov' else p_true[:,m]
                    received_attempt[q,:,m]=scheduled&(packet_u[node,q,:,m]>=probability)
                    if loss_mode=='markov':
                        change=scheduled&(transition_u[node,q,:,m]<np.where(states==0,true_transition[0,1],true_transition[1,0]))
                        states[change]=1-states[change]
            if belief_aware:
                for i,act in enumerate(selected):
                    beliefs[i]=update_belief(p_design[i],.8+.2*p_design[i],active[i],act.repetitions,
                                            received_attempt[:,i],beliefs[i])
        received=received_attempt.any(axis=0);first=received_attempt.argmax(axis=0)
        over|=((active&~received).sum(axis=1)>cfg.erasure_budget)
        variance=table.variance[np.arange(cfg.anchors)[None,:],levels]
        Rtrue=cfg.range_floor_m**2+(variance-cfg.range_floor_m**2)*10**(map_db[node][None,:]/10)
        for mask in np.unique(received,axis=0):
            members=np.all(received==mask,axis=1);anchors=np.flatnonzero(mask)
            if not len(anchors):continue
            ids=np.flatnonzero(members);h=table.H[anchors];R=variance[members][:,anchors]
            info=np.einsum('ni,ij,ik->njk',1/R,h,h);post=inv2(inv2(P[members])+info)
            K=np.einsum('nij,jk,nk->nik',post,h.T,1/R);A=np.eye(2)-np.einsum('nij,jk->nik',K,h)
            noise=np.empty((len(ids),len(anchors)))
            for j,m in enumerate(anchors):noise[:,j]=normals[node,first[ids,m],ids,m]*np.sqrt(Rtrue[ids,m])
            bias=np.full_like(noise,bias_m)
            if bias_mode=='adversarial' and bias_m:
                signs=np.array(list(itertools.product((-1.,1.),repeat=len(anchors))))
                centers=np.einsum('nij,nj->ni',A,mean[members])
                possibilities=centers[:,None,:]-bias_m*np.einsum('nij,sj->nsi',K,signs)
                chosen=np.argmax((possibilities**2).sum(axis=-1),axis=-1)
                bias=bias_m*signs[chosen]
            if nonlinear:
                delta=instance.nodes[node]-instance.anchors[anchors]
                true_delta=delta[None]+errors[members,None,:]
                residual=np.sqrt((true_delta**2).sum(axis=-1)+cfg.altitude_m**2)-np.sqrt((delta**2).sum(axis=-1)+cfg.altitude_m**2)
                errors[members]-=np.einsum('nij,nj->ni',K,residual+noise+bias)
            else:errors[members]=np.einsum('nij,nj->ni',A,errors[members])-np.einsum('nij,nj->ni',K,noise+bias)
            mean[members]=np.einsum('nij,nj->ni',A,mean[members])-np.einsum('nij,nj->ni',K,bias)
            P[members]=post;eta[members]+=a.bias_bound**2*(1/R).sum(axis=-1)
        if a.bias_bound and bias_m<=a.bias_bound and not nonlinear and csi_sd==csi_margin==0:
            mean_bound=np.linalg.eigvalsh(P)[:,-1]*eta
            if np.any((mean**2).sum(axis=1)>mean_bound+1e-8):raise RuntimeError('Bias-energy envelope failed.')
        failed|=(errors**2).sum(axis=1)>cfg.radius_m**2
        nees+=float(np.einsum('ni,nij,nj->n',errors,inv2(P),errors).mean())
        unique,inverse=np.unique(P.reshape(samples,4),axis=0,return_inverse=True)
        low,up=log_radial_bounds(unique.reshape(-1,2,2),cfg.radius_m,256 if isinstance(radio,BurstRadio) else 2048)
        low=np.exp(low)[inverse];up=np.exp(up)[inverse]
        radial+=up;radial_lower+=low;max_epoch_lower=np.maximum(max_epoch_lower,low)
        counts=active.sum(axis=1)*qvalues;pilots+=counts.mean()*cfg.pilot_s/cfg.frame_s
        active_count+=active.sum(axis=1).mean();attempts+=counts.mean();repetition_count+=qvalues.mean()
        energy+=float(np.mean([act.repetitions*table.energy[act.index] for act in selected]))
        cost+=float(np.mean([act.repetitions*table.cost[act.index] for act in selected]))
        if 1<=node<=cfg.stops:
            association=int(np.argmax(instance.gains[node]));gamma_d=np.expm1(table.rate/cfg.bandwidth_hz*np.log(2))
            symbols,packets,_=data_schedule(counts,gamma_d,cfg)
            success=1-bler_normal_approx(gamma_d*10**(-map_db[node,association]/10),symbols,608)
            payload+=float((((np.arange(max_packets)[None,:]<packets[:,None])&(data_u[node]<success[:,None])).sum(axis=1)*512).mean())
    failed|=unresolved;k=int(failed.sum());lo,hi=binomial_interval(k,samples)
    return dict(violation=k/samples,failures=k,samples=samples,risk_low99=lo,risk_high99=hi,
        pilot_fraction=pilots/Horizon,pilot_energy_mj=1000*energy,radio_cost=cost,goodput_bits=payload,
        nees=nees/Horizon,radial_union_bound=float(np.minimum(radial,1).mean()),
        radial_marginal_sum_lower=float(radial_lower.mean()),max_epoch_tail_lower=float(max_epoch_lower.mean()),
        max_conditional_union=float(conditional.max()),mean_conditional_union=float(conditional.mean()),
        budget_exceed_mission=float(over.mean()),unresolved=float(unresolved.mean()),
        mean_active_links=active_count/Horizon,mean_attempts=attempts/Horizon,mean_repetitions=repetition_count/Horizon,
        decision_ms=float(np.mean(decision)) if decision else 0.)
