"""Small Gym-like adapter for dm_control.suite (optional dependency)."""
from __future__ import annotations
import numpy as np

class DMCEnv:
    def __init__(self,domain:str,task:str,seed:int=0):
        try: from dm_control import suite
        except ImportError as e: raise ImportError("Install with: pip install -e '.[dmc]'") from e
        self.env=suite.load(domain,task,task_kwargs={"random":seed}); self._action_spec=self.env.action_spec()
        self.action_num=int(np.prod(self._action_spec.shape)); self.max_action=float(np.max(np.abs([self._action_spec.minimum,self._action_spec.maximum])))
        self._obs_keys=None
    def _flat(self,obs):
        if self._obs_keys is None: self._obs_keys=tuple(obs.keys())
        return np.concatenate([np.asarray(obs[k]).ravel() for k in self._obs_keys]).astype(np.float32)
    def reset(self):
        ts=self.env.reset(); return self._flat(ts.observation),{}
    def step(self,action):
        ts=self.env.step(action); obs=self._flat(ts.observation); reward=float(ts.reward or 0.0); done=bool(ts.last())
        return obs,reward,done,False,{}
    def sample_action(self):
        return np.random.uniform(self._action_spec.minimum,self._action_spec.maximum).astype(np.float32)
    def close(self): pass
