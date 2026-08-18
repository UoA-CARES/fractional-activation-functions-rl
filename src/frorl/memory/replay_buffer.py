from __future__ import annotations
import numpy as np


class ReplayBuffer:
    def __init__(self,capacity:int=1_000_000,seed:int|None=None):
        self.capacity=int(capacity); self.storage=[]; self.pos=0
        self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.storage)
    def add(self,state,action,reward,next_state,done):
        item=(np.asarray(state,dtype=np.float32),np.asarray(action,dtype=np.float32),float(reward),np.asarray(next_state,dtype=np.float32),float(done))
        if len(self.storage)<self.capacity: self.storage.append(item)
        else: self.storage[self.pos]=item
        self.pos=(self.pos+1)%self.capacity
    def sample(self,batch_size:int):
        if len(self.storage)<batch_size: raise ValueError(f"Need {batch_size} samples; buffer has {len(self.storage)}")
        idx=self.rng.integers(0,len(self.storage),size=batch_size)
        batch=[self.storage[i] for i in idx]
        s,a,r,ns,d=zip(*batch)
        return (np.stack(s),np.stack(a),np.asarray(r,dtype=np.float32)[:,None],np.stack(ns),np.asarray(d,dtype=np.float32)[:,None])
