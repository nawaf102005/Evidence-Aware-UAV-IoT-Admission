"""Matched PPO-Lagrangian and Double-DQN candidate generators, R22.

These are new, inspectable NumPy implementations of established algorithms,
not reproductions of a different UAV trajectory or multi-agent system.
No test geometry, true packet probability or verification label enters training.
The environment uses a deterministic expected-information surrogate; its output
is a proposal, never a risk certificate. Final checking uses the original KF.
"""
from pathlib import Path
import json, time
import numpy as np
from memory_certificates import EYE, posterior, radial_bounds


class Network:
    def __init__(self, inputs, outputs, seed, hidden=48):
        r=np.random.default_rng(seed)
        self.p={'w1':r.normal(0,np.sqrt(2/inputs),(inputs,hidden)),
                'b1':np.zeros(hidden),'w2':r.normal(0,np.sqrt(2/hidden),(hidden,hidden)),
                'b2':np.zeros(hidden),'wo':r.normal(0,.01,(hidden,outputs)),
                'bo':np.zeros(outputs)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()};self.iter=0

    def forward(self,x,cache=False):
        x=np.atleast_2d(x);p=self.p
        h1=np.maximum(0,x@p['w1']+p['b1'])
        h2=np.maximum(0,h1@p['w2']+p['b2'])
        y=h2@p['wo']+p['bo']
        return (y,(x,h1,h2)) if cache else y

    def backward(self,dy,cache,rate=3e-4):
        x,h1,h2=cache;p=self.p
        g={'wo':h2.T@dy,'bo':dy.sum(0)}
        d2=(dy@p['wo'].T)*(h2>0)
        g.update(w2=h1.T@d2,b2=d2.sum(0))
        d1=(d2@p['w2'].T)*(h1>0)
        g.update(w1=x.T@d1,b1=d1.sum(0))
        norm=np.sqrt(sum(float(np.sum(v*v)) for v in g.values()))
        scale=min(1.,.5/max(norm,1e-30));self.iter+=1
        for k in p:
            gg=g[k]*scale;self.m[k]=.9*self.m[k]+.1*gg
            self.v[k]=.999*self.v[k]+.001*gg*gg
            p[k]-=rate*(self.m[k]/(1-.9**self.iter))/(np.sqrt(self.v[k]/(1-.999**self.iter))+1e-8)

    def save(self,path,**metadata):
        np.savez_compressed(path,**self.p,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        z=np.load(path,allow_pickle=False)
        n=cls(z['w1'].shape[0],z['wo'].shape[1],0,z['w1'].shape[1])
        n.p={k:z[k].copy() for k in n.p};return n


def softmax(logits,valid):
    x=np.where(valid,logits,-1e9);x=x-x.max(axis=-1,keepdims=True)
    v=np.exp(x);return v/v.sum(axis=-1,keepdims=True)


class Surrogate:
    """Open-loop decision state; no realized reception is observed by a policy.

    E[information] is used only for training/proposing plans. It does not equal
    the expected Riccati covariance. That approximation is checked separately.
    """
    def __init__(self,oracle,valid,alpha=.1):
        self.o,self.valid,self.alpha=oracle,valid,alpha
        self.H=len(oracle.catalogs);self.A=len(oracle.catalogs[0]);self.reset()
        self.features=[];self.expected=[]
        for t,g in enumerate(oracle.catalogs):
            ff=[];ee=[]
            for a in g:
                atom=(a.H[:,:,None]*a.H[:,None,:]/a.variance[:,None,None]) if a.support else np.empty((0,2,2))
                info=np.sum((1-a.loss)[:,None,None]*atom,axis=0) if a.support else np.zeros((2,2))
                v=np.linalg.eigvalsh(info)
                ff.extend([len(a.support)/2,a.power_w/.32,
                           np.log1p(v[0])/5,np.log1p(v[1])/5,
                           float(np.mean(a.loss)) if a.support else 0.,a.cost/3])
                ee.append(info)
            self.features.append(np.array(ff));self.expected.append(ee)

    def reset(self):
        self.t=0;self.P=self.o.initial.copy();self.sumrisk=0.;return self

    def state(self):
        P=self.P+self.o.process[self.t]*EYE
        eig=np.linalg.eigvalsh(P)
        glob=[self.t/self.H,self.H/30,np.log1p(eig[0])/3,np.log1p(eig[1])/3,
              P[0,1]/max(float(np.trace(P)),1e-8),self.o.process[self.t]/2,
              min(self.sumrisk/self.alpha,10.)/10,1.]
        return np.r_[glob,self.features[self.t]]

    def step(self,a):
        if not self.valid[self.t][a]:raise ValueError('Queue-infeasible action.')
        g=self.o.catalogs[self.t]
        self.P=posterior(self.P+self.o.process[self.t]*EYE,self.expected[self.t][a])
        risk=float(radial_bounds(self.P,self.o.radius,32)[1]);self.sumrisk+=risk
        cost=g[a].cost/self.H;self.t+=1;done=self.t==self.H
        return cost,risk/self.alpha,done


def propose(network,oracle,valid,kind='ppo',rng=None):
    env=Surrogate(oracle,valid);path=[]
    for _ in range(env.H):
        y=network.forward(env.state())[0]
        if kind=='ppo':
            p=softmax(y[:env.A],valid[env.t])
            a=int(np.argmax(p)) if rng is None else int(rng.choice(env.A,p=p))
        else:a=int(np.argmax(np.where(valid[env.t],y,-1e9)))
        path.append(a);env.step(a)
    return tuple(path)


def train_ppo(environments,seed,episodes=2048,batch_episodes=32):
    """Clipped PPO, GAE(.95), gamma=1, dual update on episode risk.

    Shared two-layer actor/value network, 4 epochs/minibatch128, clip .2,
    entropy .01, Adam3e-4, norm .5. Final checkpoint is always retained.
    """
    rng=np.random.default_rng(seed);A=environments[0].A
    net=Network(len(environments[0].reset().state()),A+1,seed)
    dual=1.;log=[];steps=0;tic=time.perf_counter()
    for start in range(0,episodes,batch_episodes):
        xs=[];aa=[];old=[];adv=[];returns=[];masks=[];risks=[];costs=[]
        for episode in range(min(batch_episodes,episodes-start)):
            env=environments[int(rng.integers(len(environments)))];env.reset()
            trajectory=[];totalcost=0.
            for t in range(env.H):
                x=env.state();mask=env.valid[t];y=net.forward(x)[0]
                prob=softmax(y[:A],mask);a=int(rng.choice(A,p=prob))
                c,r,done=env.step(a);totalcost+=c
                trajectory.append((x,a,np.log(max(prob[a],1e-30)),y[A],-c-dual*r,mask))
            gae=0.;vals=[v[3] for v in trajectory]+[0.]
            aep=[]
            for t in range(len(trajectory)-1,-1,-1):
                gae=trajectory[t][4]+vals[t+1]-vals[t]+.95*gae;aep.append(gae)
            aep=aep[::-1]
            for v,z in zip(trajectory,aep):
                xs.append(v[0]);aa.append(v[1]);old.append(v[2]);adv.append(z)
                returns.append(z+v[3]);masks.append(v[5])
            risks.append(env.sumrisk/env.alpha);costs.append(totalcost);steps+=env.H
        xs=np.array(xs);aa=np.array(aa);old=np.array(old);adv=np.array(adv)
        returns=np.array(returns);masks=np.array(masks)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        for epoch in range(4):
            order=rng.permutation(len(xs))
            for s in range(0,len(xs),128):
                ix=order[s:s+128];y,cache=net.forward(xs[ix],True)
                probs=softmax(y[:,:A],masks[ix]);idx=np.arange(len(ix))
                ratio=np.exp(np.clip(np.log(np.maximum(probs[idx,aa[ix]],1e-30))-old[ix],-20,20))
                active=np.where(adv[ix]>=0,ratio<=1.2,ratio>=.8)
                coeff=-adv[ix]*ratio*active/len(ix)
                d=coeff[:,None]*(-probs);d[idx,aa[ix]]+=coeff
                lp=np.log(np.maximum(probs,1e-30))
                d+=.01*probs*(lp-(probs*lp).sum(1,keepdims=True))/len(ix)
                dy=np.c_[d,.5*(y[:,A]-returns[ix])/len(ix)]
                net.backward(dy,cache)
        dual=float(np.clip(dual+.05*(np.mean(risks)-1),0,50))
        log.append(dict(episodes=start+len(risks),steps=steps,dual=dual,
                        proxy_risk=float(np.mean(risks)),cost=float(np.mean(costs))))
    return net,log,dict(steps=steps,episodes=episodes,seconds=time.perf_counter()-tic,dual=dual)


def train_ddqn(environments,seed,episodes=2048):
    """Double DQN with replay and a target network; Lagrangian cost reward.

    Same state/actions/budget as PPO. Replay stores raw costs and risk, then
    relabels with the current multiplier. Adam3e-4, batch64, update/4 steps,
    hard target/500 steps, epsilon1->.05 over70% training, gamma1.
    """
    rng=np.random.default_rng(seed);A=environments[0].A;D=len(environments[0].reset().state())
    net=Network(D,A,seed);target=Network(D,A,seed)
    target.p={k:v.copy() for k,v in net.p.items()}
    capacity=50000;X=np.zeros((capacity,D));Y=X.copy();act=np.zeros(capacity,int)
    cc=np.zeros(capacity);rr=cc.copy();done=cc.copy();mask=np.ones((capacity,A),bool)
    dual=1.;steps=0;log=[];tic=time.perf_counter()
    for episode in range(episodes):
        env=environments[int(rng.integers(len(environments)))];env.reset();totalcost=0.
        for t in range(env.H):
            x=env.state();v=env.valid[t];eps=max(.05,1.-.95*episode/(.7*episodes))
            a=int(rng.choice(np.flatnonzero(v))) if rng.random()<eps else int(np.argmax(np.where(v,net.forward(x)[0],-1e9)))
            c,r,d=env.step(a);totalcost+=c
            nxt=np.zeros(D) if d else env.state();nm=np.ones(A,bool) if d else env.valid[env.t]
            z=steps%capacity;X[z]=x;Y[z]=nxt;act[z]=a;cc[z]=c;rr[z]=r;done[z]=d;mask[z]=nm
            steps+=1
            if steps>=512 and steps%4==0:
                ix=rng.integers(min(steps,capacity),size=64)
                best=np.argmax(np.where(mask[ix],net.forward(Y[ix]),-1e9),axis=1)
                future=target.forward(Y[ix])[np.arange(64),best]
                truth=-cc[ix]-dual*rr[ix]+(1-done[ix])*future
                q,cache=net.forward(X[ix],True);dy=np.zeros_like(q)
                dy[np.arange(64),act[ix]]=np.clip(q[np.arange(64),act[ix]]-truth,-1,1)/64
                net.backward(dy,cache)
            if steps%500==0:target.p={k:v.copy() for k,v in net.p.items()}
        dual=float(np.clip(dual+.003*(env.sumrisk/env.alpha-1),0,50))
        if (episode+1)%32==0:
            log.append(dict(episodes=episode+1,steps=steps,dual=dual,
                            proxy_risk=env.sumrisk/env.alpha,cost=totalcost))
    return net,log,dict(steps=steps,episodes=episodes,seconds=time.perf_counter()-tic,dual=dual)
