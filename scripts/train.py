#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from frorl.algorithms import TD3,SAC
from frorl.experiments import activation_layout
from frorl.memory import ReplayBuffer
from frorl.models.networks import DeterministicPolicy,TanhGaussianPolicy,TwinQNetwork
from frorl.training import train
from frorl.utils import get_device,set_seed


def gym_env(name,seed):
    try: import gymnasium as gym
    except ImportError as e: raise SystemExit("Gymnasium is required. Install: pip install -e '.[gym]'") from e
    env=gym.make(name); env.reset(seed=seed); env.action_space.seed(seed); return env

def dmc_env(label,seed):
    from frorl.environments import DMCEnv
    domain,task=label.split('-',1); return DMCEnv(domain,task,seed)

def dims(env):
    if hasattr(env,'observation_space'):
        state_dim=int(np.prod(env.observation_space.shape)); action_dim=int(np.prod(env.action_space.shape)); max_action=float(np.max(np.abs(env.action_space.high)))
    else:
        o,_=env.reset(); state_dim=o.size; action_dim=env.action_num; max_action=env.max_action
    return state_dim,action_dim,max_action

def main():
    p=argparse.ArgumentParser(); p.add_argument('--algo',choices=['TD3','SAC'],required=True); p.add_argument('--env',required=True)
    p.add_argument('--activation',default='relu'); p.add_argument('--alpha',type=float,default=None); p.add_argument('--layers',type=int,choices=[1,2],default=2)
    p.add_argument('--placement',default='all-both'); p.add_argument('--seed',type=int,default=0); p.add_argument('--steps',type=int,default=1_000_000)
    p.add_argument('--start-steps',type=int,default=1000); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--buffer-size',type=int,default=1_000_000)
    p.add_argument('--eval-every',type=int,default=10_000); p.add_argument('--eval-episodes',type=int,default=10); p.add_argument('--output',default='results')
    a=p.parse_args(); set_seed(a.seed); device=get_device(); placement='both' if a.layers==1 else a.placement
    is_dmc=a.env.lower() in {'cartpole-swingup','cheetah-run','finger-spin','walker-walk'}
    env=(dmc_env if is_dmc else gym_env)(a.env,a.seed); ev=(dmc_env if is_dmc else gym_env)(a.env,a.seed+10_000)
    state_dim,action_dim,max_action=dims(env); hidden=[256]*a.layers; actor_acts,critic_acts=activation_layout(a.activation,a.alpha,a.layers,placement)
    critic=TwinQNetwork(state_dim,action_dim,hidden,critic_acts)
    if a.algo=='TD3':
        actor=DeterministicPolicy(state_dim,action_dim,max_action,hidden,actor_acts); agent=TD3(actor,critic,3e-4,3e-4,.99,.005,2,device)
    else:
        actor=TanhGaussianPolicy(state_dim,action_dim,max_action,hidden,actor_acts); agent=SAC(actor,critic,3e-4,3e-4,3e-4,.99,.005,1.0,1,1,device)
    alpha_label='base' if a.alpha is None else f'{a.alpha:g}'; out=Path(a.output)/a.algo/a.env/a.activation/f'a{alpha_label}'/f'{a.layers}layer'/(placement)/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    train(env,ev,agent,ReplayBuffer(a.buffer_size,a.seed),a.steps,a.start_steps,a.batch_size,a.eval_every,a.eval_episodes,out)
    env.close(); ev.close()
if __name__=='__main__': main()
