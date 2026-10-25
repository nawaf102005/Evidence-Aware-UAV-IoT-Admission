"""Closed stochastic radius certificates and polynomial-size E-opt heuristics.

Pattern-GRE uses a no-observation prediction envelope. Unlike a bounded-loss
GRE envelope, it remains valid after *every* pattern, including all links lost.
No conditional covariance cap is inferred from an averaged risk constraint.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import linprog

from secure_drop_research import Config, RadioTable, eigmin2, inv2, generate_instance
from wireless_extensions import (ExtendedRadio, radial_tail_bounds, bler_normal_approx,
    plan_from_schedules, expected_payload, erasure_sets)


def loss_patterns(anchors):
    return np.array(list(itertools.product((False, True), repeat=anchors)), dtype=bool)


def pattern_probabilities(probabilities, patterns=None):
    p = np.asarray(probabilities, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Loss probabilities must be in [0,1].")
    patterns = loss_patterns(p.shape[-1]) if patterns is None else np.asarray(patterns)
    return np.prod(np.where(patterns, p[..., None, :], 1-p[..., None, :]), axis=-1)


def markov_pattern_probabilities(good_loss, bad_loss, transition, initial, patterns=None):
    """Two-state Gilbert--Elliott law over successive ranging slots.

    A row-stochastic transition acts *after* each slot. Conditional slot errors
    are independent given the hidden states. Return full joint pattern law;
    temporal persistence within a frame is therefore retained exactly.
    """
    good, bad = np.asarray(good_loss), np.asarray(bad_loss)
    transition, initial = np.asarray(transition), np.asarray(initial)
    if good.shape != bad.shape or transition.shape != (2,2) or initial.shape != (2,):
        raise ValueError("Invalid Gilbert--Elliott dimensions.")
    if (np.any((good < 0)|(good > 1)) or np.any((bad < 0)|(bad > 1))
            or np.any(transition < 0) or np.any(initial < 0)
            or not np.allclose(transition.sum(axis=1), 1) or not np.isclose(initial.sum(), 1)):
        raise ValueError("Invalid Gilbert--Elliott probability law.")
    patterns = loss_patterns(len(good)) if patterns is None else patterns
    state = np.broadcast_to(initial, (len(patterns),2)).copy()
    for m in range(len(good)):
        emission = np.array([good[m],bad[m]])
        state *= np.where(patterns[:,m,None], emission, 1-emission)
        state = state @ transition
    return state.sum(axis=1)


def weighted_radial_bounds(prior, parts, probabilities, radius, terms=256):
    """Exact finite loss-pattern sum, bracketed only in its angular integral."""
    probabilities = np.asarray(probabilities)
    patterns = loss_patterns(len(parts))
    weights = pattern_probabilities(probabilities, patterns)
    matrices = inv2(inv2(prior) + np.einsum("em,mij->eij", ~patterns, parts))
    lower, upper = radial_tail_bounds(matrices, radius, terms)
    return float(weights @ lower), float(weights @ upper)


class PatternRadio(ExtendedRadio):
    def __init__(self, table, packet_floor=.02, terms=256):
        super().__init__(table)
        self.patterns = loss_patterns(table.cfg.anchors)
        self.terms = terms
        gamma = table.sinr[np.arange(table.cfg.anchors)[None,:],table.codebook]
        symbols = max(1,round(table.cfg.bandwidth_hz*table.cfg.pilot_control_s))
        self.probabilities = np.where(table.codebook>0,
            packet_floor+(1-packet_floor)*bler_normal_approx(gamma,symbols,48),0.)
        self.pattern_weights = pattern_probabilities(self.probabilities,self.patterns)
        # The dictionary is small; storing 4^M x 2^M x 2 x 2 is deliberately
        # avoided. Surviving information is generated only for candidate blocks.

    def choose_pattern(self, prior, target=None):
        cfg, table = self.table.cfg,self.table
        target = cfg.mission_risk/cfg.epochs_max if target is None else target
        key = ("Pattern",float(target),self.terms,*np.round(prior.ravel(),12))
        if key in self.cache:
            return self.cache[key]
        candidates=table.sorted_indices[table.bits[table.sorted_indices]>=cfg.data_min_bits]
        for begin in range(0,len(candidates),24):
            chunk=candidates[begin:begin+24]
            # Collapse impossible patterns for this block; inactive anchors
            # have loss probability zero. All nonzero-probability events remain.
            enabled=np.any(self.pattern_weights[chunk]>0,axis=0)
            patterns=self.patterns[enabled]
            probability=self.pattern_weights[chunk][:,enabled]
            info=np.einsum("em,cmij->ceij",~patterns,self.parts[chunk])
            post=inv2(inv2(prior)+info)
            lower,_=radial_tail_bounds(post,cfg.radius_m,16)
            possible=np.flatnonzero((probability*lower).sum(axis=1)<=target)
            if not len(possible):
                continue
            _,upper=radial_tail_bounds(post[possible],cfg.radius_m,self.terms)
            feasible=np.flatnonzero((probability[possible]*upper).sum(axis=1)<=target*(1-1e-10))
            if len(feasible):
                index=int(chunk[possible[feasible[0]]]); self.cache[key]=index
                return index
        self.cache[key]=None
        return None

    def risk_bound(self,index,prior,terms=None):
        return weighted_radial_bounds(prior,self.parts[index],self.probabilities[index],
            self.table.cfg.radius_m,self.terms if terms is None else terms)


def pattern_plan(routes,tables,extra,cfg,method="Pattern-GRE"):
    """Restricted exact planner under the history-valid no-observation envelope."""
    best=None
    for route in routes:
        cumulative=0.; schedules=[]
        for node,length in zip(route.events,route.lengths):
            cumulative+=length
            prior=np.eye(2)*(cfg.initial_sd_m**2+cfg.drift_m2_per_m*cumulative)
            index=extra[node].choose_pattern(prior)
            if index is None:
                break
            schedules.append(index)
        if len(schedules)==len(route.events):
            plan=plan_from_schedules(method,route,schedules,tables,cfg)
            if best is None or plan.objective<best.objective:
                best=plan
    return best


def choose_gre_terms(extra,prior,terms=256,packet_risk=None):
    """Cost-ordered GRE search with an explicit quadrature/risk-split setting."""
    table,cfg=extra.table,extra.table.cfg
    key=("GRE-N",terms,packet_risk,*np.round(prior.ravel(),12))
    if key in extra.cache: return extra.cache[key]
    valid=table.bits>=cfg.data_min_bits
    if packet_risk is not None:
        valid &= extra.packet_budget_tail<=packet_risk/cfg.epochs_max
    indices=table.sorted_indices[valid[table.sorted_indices]]
    epsilon=cfg.mission_risk/cfg.epochs_max
    for start in range(0,len(indices),96):
        chunk=indices[start:start+96]
        posteriors=[inv2(inv2(prior)+info[chunk]) for info in extra.surviving]
        lower=np.zeros(len(chunk))
        for P in posteriors: lower=np.maximum(lower,radial_tail_bounds(P,cfg.radius_m,16)[0])
        possible=np.flatnonzero(lower<=epsilon)
        if not len(possible): continue
        upper=np.zeros(len(possible))
        for P in posteriors: upper=np.maximum(upper,radial_tail_bounds(P[possible],cfg.radius_m,terms)[1])
        feasible=np.flatnonzero(upper<=epsilon*(1-1e-10))
        if len(feasible):
            index=int(chunk[possible[feasible[0]]]); extra.cache[key]=index; return index
    extra.cache[key]=None
    return None


def split_plan(routes,tables,extra,cfg,packet_risk=.05):
    from wireless_extensions import gaussian_envelope_cap
    best=None; cap=gaussian_envelope_cap(cfg)
    if cfg.initial_sd_m**2>cap: return None
    for route in routes:
        schedules=[]
        for node,length in zip(route.events,route.lengths):
            prior=(cap+cfg.drift_m2_per_m*length)*np.eye(2)
            index=choose_gre_terms(extra[node],prior,256,packet_risk)
            if index is None: break
            schedules.append(index)
        if len(schedules)==len(route.events):
            plan=plan_from_schedules('PHY-GRE',route,schedules,tables,cfg)
            if best is None or plan.objective<best.objective: best=plan
    return best


def evaluate_pattern(instance,plan,tables,extra,cfg,samples=128,causal=False,packet_floor=.02):
    """Affine KF, with history-wise weighted-risk reallocation and safe fallback.

    This simulation evaluates independent losses; Markov pattern computation
    is separately provided and tested. Online schedules depend on covariances
    and received-link indicators, never on Gaussian observation residuals.
    """
    rng=np.random.default_rng(instance.seed+90_000_000)
    errors=cfg.initial_sd_m*rng.normal(size=(samples,2))
    process=rng.normal(size=(cfg.stops+2,samples,2))
    noise=rng.normal(size=(cfg.stops+2,samples,cfg.anchors))
    prng=np.random.default_rng(instance.seed+91_000_000)
    uniforms=prng.random((cfg.stops+2,samples,cfg.anchors))
    prng.random((cfg.stops+2,samples))  # shared burst draws consumed by reference
    max_packets=int(np.ceil(cfg.frame_s/(320/cfg.bandwidth_hz+10e-6)))
    data_uniforms=prng.random((cfg.stops+2,samples,max_packets))
    P=np.tile(np.eye(2)*cfg.initial_sd_m**2,(samples,1,1))
    failures=np.zeros(samples,bool); risk=np.zeros(samples); cert=np.zeros(samples)
    pilots=payload=nees=0.; unresolved=0; decision_ms=[]
    from wireless_extensions import data_schedule
    for node,length,offline in zip(plan.route.events,plan.route.lengths,plan.schedules):
        Q=cfg.drift_m2_per_m*length
        errors+=np.sqrt(Q)*process[node]; P+=Q*np.eye(2)
        indices=np.full(samples,offline,dtype=int)
        # At most 128 covariance states are considered in this finite test.
        # Cache includes the exact (12-digit) prior, and feasibility below is
        # rechecked at the unrounded prior before applying any decision.
        if causal:
            rounded=np.round(P.reshape(samples,4),12)
            for values in np.unique(rounded,axis=0):
                members=np.all(rounded==values,axis=1)
                clock=time.perf_counter()
                chosen=extra[node].choose_pattern(values.reshape(2,2))
                decision_ms.append(1000*(time.perf_counter()-clock))
                if chosen is None:
                    chosen=offline
                indices[members]=chosen
        for index in np.unique(indices):
            members=indices==index
            # Different histories can share a codeword. Evaluate the current
            # bound per history rather than substituting one sample's prior.
            patterns=extra[node].patterns
            weights=extra[node].pattern_weights[index]
            enable=weights>0
            S=np.einsum("em,mij->eij",~patterns[enable],extra[node].parts[index])
            posterior=inv2(inv2(P[members])[:,None]+S)
            _,upper=radial_tail_bounds(posterior,cfg.radius_m,256)
            conditional=upper@weights[enable]
            if np.any(conditional>cfg.mission_risk/cfg.epochs_max+1e-9):
                raise RuntimeError("History-wise stochastic certificate failed.")
            cert[members]+=conditional
        table=tables[node]; levels=table.codebook[indices]
        active=levels>0
        gamma=table.sinr[np.arange(cfg.anchors)[None,:],levels]
        probability=packet_floor+(1-packet_floor)*bler_normal_approx(gamma,round(cfg.pilot_control_s*cfg.bandwidth_hz),48)
        received=active & (uniforms[node]>=probability)
        variance=table.variance[np.arange(cfg.anchors)[None,:],levels]
        for mask in np.unique(received,axis=0):
            members=np.all(received==mask,axis=1); anchors=np.flatnonzero(mask)
            if not len(anchors):
                continue
            H=table.H[anchors]; R=variance[members][:,anchors]
            S=np.einsum("ni,ij,ik->njk",1/R,H,H)
            post=inv2(inv2(P[members])+S)
            K=np.einsum("nij,jk,nk->nik",post,H.T,1/R)
            A=np.eye(2)-np.einsum("nij,jk->nik",K,H)
            errors[members]=np.einsum("nij,nj->ni",A,errors[members])-np.einsum("nij,nj->ni",K,noise[node][members][:,anchors]*np.sqrt(R))
            P[members]=post
        failures|=(errors**2).sum(axis=1)>cfg.radius_m**2
        _,upper=radial_tail_bounds(P,cfg.radius_m,256); risk+=upper
        nees+=float(np.einsum("ni,nij,nj->n",errors,inv2(P),errors).mean())
        counts=active.sum(axis=1); pilots+=counts.mean()*cfg.pilot_s/cfg.frame_s
        if 1<=node<=cfg.stops:
            gamma_data=cfg.data_power_w*instance.gains[node].max()/(cfg.noise_w+instance.interference[node])
            symbols,packets,_=data_schedule(counts,gamma_data,cfg)
            success=1-bler_normal_approx(gamma_data,symbols,608)
            payload+=float((((np.arange(max_packets)[None,:]<packets[:,None]) & (data_uniforms[node]<success[:,None])).sum(axis=1)*512).mean())
    return dict(violation=float(failures.mean()),pilot_fraction=float(pilots/len(plan.route.events)),
        goodput_bits=payload,nees=nees/len(plan.route.events),radial_union_bound=float(np.minimum(risk,1).mean()),
        conditional_union_bound=float(cert.mean()),max_conditional_union=float(cert.max()),
        decision_ms_mean=float(np.mean(decision_ms)) if decision_ms else 0.,unresolved=unresolved)


def uma_aerial_pathloss(distance,los,h_ut=60.,h_bs=25.,fc_ghz=3.5):
    """TR 36.777 Annex B UMa-AV path-loss subset (22.5 < hUT <= 100)."""
    d=np.asarray(distance,dtype=float)
    if np.any((d<10)|(d>4000)) or not 22.5<h_ut<=100 or h_bs!=25.:
        raise ValueError("Aerial subset requires d2D=10..4000, hUT=22.5..100, hBS=25.")
    if not .5<fc_ghz<100:
        raise ValueError("Invalid carrier frequency.")
    d3=np.sqrt(d*d+(h_ut-h_bs)**2)
    pl_los=28+22*np.log10(d3)+20*np.log10(fc_ghz)
    pl_nlos=-17.5+(46-7*np.log10(h_ut))*np.log10(d3)+20*np.log10(40*np.pi*fc_ghz/3)
    return np.where(los,pl_los,pl_nlos)


def generate_aerial_instance(seed,cfg):
    if cfg.altitude_m!=35.:
        raise ValueError("UMa-AV subset: UAV/anchor heights 60/25 m; relative height 35 m.")
    instance=generate_instance(seed,cfg); rng=np.random.default_rng(seed+40_000_000)
    d=np.linalg.norm(instance.nodes[:,None]-instance.anchors[None],axis=2)
    h=60.; d1=max(460*np.log10(h)-700,18.); p1=4300*np.log10(h)-3800
    probability=np.where(d<=d1,1.,d1/d+np.exp(-d/p1)*(1-d1/d))
    los=rng.random(d.shape)<probability
    # Standard LOS/NLOS shadow standard deviations (Table B-3). Independent
    # service-node draws, no clustered fading or spatial-consistency claim.
    shadow=rng.normal(size=d.shape)*np.where(los,4.64*np.exp(-.0066*h),6.)
    pathloss=uma_aerial_pathloss(d,los)
    return replace(instance,gains=10**(-(pathloss+shadow)/10))


def scalable_eopt(instance,node,length,cfg,lp_rounds=6):
    """Forward greedy + pruning, with a continuous outer-approximation lower bound.

    Polynomial in M for a fixed erasure budget. Gap is relative to the exact
    DISCRETE E-opt local reserve, never to route/GRE/stochastic optimality.
    A heuristic failure is unresolved, not proof of infeasibility.
    """
    start=time.perf_counter(); M=cfg.anchors; L=len(cfg.powers_w)-1
    powers=np.asarray(cfg.powers_w[1:]); delta=instance.nodes[node]-instance.anchors
    d=np.sqrt((delta**2).sum(axis=1)+cfg.altitude_m**2); H=delta/d[:,None]
    gamma=instance.gains[node,:,None]*powers/(cfg.noise_w+instance.interference[node])
    beta=cfg.bandwidth_hz/np.sqrt(12)
    R=cfg.range_floor_m**2+cfg.ranging_efficiency*299792458.**2/(8*np.pi**2*beta**2*gamma*cfg.bandwidth_hz*(cfg.pilot_s-cfg.pilot_control_s))
    atoms=H[:,None,:,None]*H[:,None,None,:]/R[:,:,None,None]
    required=1/cfg.covariance_cap-1/(cfg.covariance_cap+cfg.drift_m2_per_m*length)
    sets=erasure_sets(M,cfg.erasure_budget)
    erased_masks=np.zeros((len(sets),M),bool)
    for j,s in enumerate(sets): erased_masks[j,list(s)]=True
    counts=np.arange(M+1); data_gamma=cfg.data_power_w*instance.gains[node].max()/(cfg.noise_w+instance.interference[node])
    allowed=counts[expected_payload(counts,data_gamma,cfg)>=cfg.data_min_bits]
    countmax=int(allowed[-1]) if len(allowed) else -1
    cost=np.tile(1+cfg.pilot_energy_price_per_j*cfg.pilot_s*powers,M)
    def matrices(levels):
        selected=np.zeros((M,2,2)); active=np.flatnonzero(levels)
        selected[active]=atoms[active,levels[active]-1]
        return np.einsum("em,mij->eij",~erased_masks,selected)
    def score(levels): return float(eigmin2(matrices(levels)).min())
    def value(levels): return float(sum(1+cfg.pilot_energy_price_per_j*cfg.pilot_s*cfg.powers_w[x] for x in levels if x))
    levels=np.zeros(M,int); current=0.
    for _ in range(max(0,countmax)):
        candidates=[]
        for anchor in np.flatnonzero(levels==0):
            proposal=levels.copy(); proposal[anchor]=L
            matrices_p=matrices(proposal)
            eigen=float(eigmin2(matrices_p).min())
            trace=float(np.trace(matrices_p,axis1=-2,axis2=-1).min())
            candidates.append((eigen,trace,-anchor,proposal))
        if not candidates: break
        candidate=max(candidates,key=lambda x:x[:3]); levels=candidate[3]; current=candidate[0]
        if current>=required-1e-9: break
    if current>=required-1e-9:
        for anchor in np.argsort(instance.gains[node]):
            for level in range(levels[anchor]):
                proposal=levels.copy(); proposal[anchor]=level
                if score(proposal)>=required-1e-9:
                    levels=proposal; break
        upper=value(levels); status="feasible"
    else: upper=np.inf; status="unresolved"; levels=None
    greedy_ms=1000*(time.perf_counter()-start)
    rows=[]; rhs=[]
    for anchor in range(M):
        row=np.zeros((M,L)); row[anchor]=1; rows.append(row.ravel()); rhs.append(1.)
    rows.append(np.ones(M*L)); rhs.append(float(countmax))
    # Integer rank/trace necessities strengthen the continuous relaxation.
    # They hold for every discrete feasible codeword despite not following
    # from the fractional cone alone. This avoids an artificial sub-anchor
    # lower bound in dense deployments.
    wmax=float((np.sum(H*H,axis=1)[:,None]/R).max())
    countmin=cfg.erasure_budget+max(2,int(np.ceil(2*required/wmax))) if required>0 else 0
    rows.append(-np.ones(M*L)); rhs.append(-float(countmin))
    def cut(erased,vector):
        row=np.einsum("i,mlij,j->ml",vector,atoms,vector); row[list(erased)]=0
        rows.append(-row.ravel()); rhs.append(-required)
    for erased in sets:
        for theta in np.arange(4)*np.pi/4: cut(erased,np.array([np.cos(theta),np.sin(theta)]))
    lower=0.; lp_status="not_run"; rounds=0
    for rounds in range(1,lp_rounds+1):
        result=linprog(cost,A_ub=np.asarray(rows),b_ub=np.asarray(rhs),bounds=(0,1),method="highs")
        if result.status==2:
            lp_status="infeasible"; status="infeasible"; break
        if not result.success: lp_status="unresolved"; break
        lower=max(lower,float(result.fun)); lp_status="bound"
        z=result.x.reshape(M,L)
        selected=np.einsum("ml,mlij->mij",z,atoms)
        S=np.einsum("em,mij->eij",~erased_masks,selected)
        eigen,vectors=np.linalg.eigh(S); violated=np.flatnonzero(eigen[:,0]<required-1e-8)
        if not len(violated): lp_status="conic_optimal"; break
        for e in violated: cut(sets[e],vectors[e,:,0])
    return dict(status=status,lp_status=lp_status,lower_bound=lower,upper_bound=upper,
        gap=max(0.,upper-lower)/max(upper,1.) if np.isfinite(upper) else np.nan,
        levels=levels.tolist() if levels is not None else None,greedy_ms=greedy_ms,
        milliseconds=1000*(time.perf_counter()-start),rounds=rounds)
