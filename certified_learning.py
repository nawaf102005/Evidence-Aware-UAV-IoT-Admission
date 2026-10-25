"""Final R22 learning protocol: exact w2 training risk and certified input law.

The initial expected-information pilot is kept in full. It revealed a model
surrogate gap, so this final protocol uses the same exact additive w2 factors
available to the DP comparator. It has a NEW held-out test cohort (6400001+).
All training runs, including failures, and the final checkpoint are reported.
"""
import numpy as np
from learning_schedulers import Network,softmax,train_ppo,train_ddqn
from learning_schedulers import Surrogate as Base


class Surrogate(Base):
    def __init__(self,oracle,valid,alpha=.1):
        super().__init__(oracle,valid,alpha)
        self.factors=oracle.factors(2)

    def reset(self):
        super().reset();self.previous=0;return self

    def local_risks(self):
        f=self.factors[self.t]
        return f if f.ndim==1 else f[self.previous]

    def state(self):
        # Previous action and the actual one-step w2 factor make the risk
        # state Markov. No true p or test Monte Carlo label is provided.
        previous=np.eye(self.A)[self.previous]
        risks=np.minimum(10,self.local_risks()/(self.alpha/self.H))/10
        return np.r_[super().state(),previous,risks]

    def step(self,a):
        risk=float(self.local_risks()[a]);before=self.sumrisk
        c,_,done=super().step(a)
        self.sumrisk=before+risk;self.previous=a
        return c,risk/self.alpha,done


def propose(network,oracle,valid,kind='ppo',rng=None):
    env=Surrogate(oracle,valid);path=[]
    for t in range(env.H):
        y=network.forward(env.state())[0]
        if kind=='ppo':
            p=softmax(y[:env.A],valid[t])
            a=int(np.argmax(p)) if rng is None else int(rng.choice(env.A,p=p))
        else:a=int(np.argmax(np.where(valid[t],y,-1e9)))
        path.append(a);env.step(a)
    return tuple(path)
