from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal


class MLP(nn.Module):
    """Feature MLP with an independent activation factory per hidden layer."""
    def __init__(self, input_dim: int, hidden_sizes: list[int], activations: list):
        super().__init__()
        if len(hidden_sizes) != len(activations):
            raise ValueError("hidden_sizes and activations must have the same length")
        layers=[]; prev=input_dim
        for width, factory in zip(hidden_sizes, activations):
            layers += [nn.Linear(prev,width), factory()]
            prev=width
        self.model=nn.Sequential(*layers); self.output_dim=prev
    def forward(self,x): return self.model(x)


class DeterministicPolicy(nn.Module):
    def __init__(self,state_dim,action_dim,max_action,hidden_sizes,activations):
        super().__init__()
        self.max_action=float(max_action); self.action_dim=action_dim; self.num_actions=action_dim
        self.backbone=MLP(state_dim,hidden_sizes,activations)
        self.output_layer=nn.Linear(self.backbone.output_dim,action_dim)
    def forward(self,state):
        return self.max_action*torch.tanh(self.output_layer(self.backbone(state)))


class TanhGaussianPolicy(nn.Module):
    LOG_STD_MIN=-20; LOG_STD_MAX=2
    def __init__(self,state_dim,action_dim,max_action,hidden_sizes,activations):
        super().__init__()
        self.max_action=float(max_action); self.action_dim=action_dim; self.num_actions=action_dim
        self.backbone=MLP(state_dim,hidden_sizes,activations)
        self.mean=nn.Linear(self.backbone.output_dim,action_dim)
        self.log_std=nn.Linear(self.backbone.output_dim,action_dim)
    def forward(self,state):
        x=self.backbone(state); mean=self.mean(x)
        log_std=torch.clamp(self.log_std(x),self.LOG_STD_MIN,self.LOG_STD_MAX)
        return mean,torch.exp(log_std)
    def sample(self,state):
        mean,std=self.forward(state); normal=Normal(mean,std)
        x_t=normal.rsample(); y_t=torch.tanh(x_t); action=y_t*self.max_action
        log_prob=normal.log_prob(x_t)-torch.log(1-y_t.pow(2)+1e-6)
        return action,log_prob.sum(dim=1,keepdim=True)


class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_sizes,activations):
        super().__init__(); self.backbone=MLP(state_dim+action_dim,hidden_sizes,activations)
        self.output_layer=nn.Linear(self.backbone.output_dim,1)
    def forward(self,state,action):
        return self.output_layer(self.backbone(torch.cat([state,action],dim=1)))


class TwinQNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_sizes,activations):
        super().__init__()
        self.q1=QNetwork(state_dim,action_dim,hidden_sizes,activations)
        self.q2=QNetwork(state_dim,action_dim,hidden_sizes,activations)
    def forward(self,state,action): return self.q1(state,action),self.q2(state,action)
