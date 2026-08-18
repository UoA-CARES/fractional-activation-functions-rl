import torch
from frorl.models.activations import activation_factory
from frorl.models.networks import DeterministicPolicy,TanhGaussianPolicy,TwinQNetwork

def test_network_shapes():
    acts=[activation_factory('frelu',.2),activation_factory('relu')]
    actor=DeterministicPolicy(5,2,1.0,[256,256],acts); critic=TwinQNetwork(5,2,[256,256],acts)
    s=torch.randn(4,5); a=actor(s); q1,q2=critic(s,a); assert a.shape==(4,2); assert q1.shape==q2.shape==(4,1)

def test_sac_sample():
    acts=[activation_factory('fswish',.2)]; actor=TanhGaussianPolicy(5,2,1.0,[256],acts); a,lp=actor.sample(torch.randn(4,5)); assert a.shape==(4,2); assert lp.shape==(4,1)
