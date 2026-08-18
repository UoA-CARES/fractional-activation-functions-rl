import numpy as np, torch
from frorl.algorithms import TD3,SAC
from frorl.memory import ReplayBuffer
from frorl.models.activations import activation_factory
from frorl.models.networks import DeterministicPolicy,TanhGaussianPolicy,TwinQNetwork

def memory():
    m=ReplayBuffer(1000,0)
    for _ in range(300): m.add(np.random.randn(3),np.random.uniform(-1,1,2),np.random.randn(),np.random.randn(3),False)
    return m

def test_td3_step():
    acts=[activation_factory('frelu',.2)]; a=DeterministicPolicy(3,2,1,[32],acts); c=TwinQNetwork(3,2,[32],acts); agent=TD3(a,c,3e-4,3e-4,.99,.005,2,torch.device('cpu')); info=agent.train_policy(memory(),64); assert 'critic_loss' in info

def test_sac_step():
    acts=[activation_factory('fgelu',.2)]; a=TanhGaussianPolicy(3,2,1,[32],acts); c=TwinQNetwork(3,2,[32],acts); agent=SAC(a,c,3e-4,3e-4,3e-4,.99,.005,1,1,1,torch.device('cpu')); info=agent.train_policy(memory(),64); assert 'critic_loss' in info and 'alpha' in info
