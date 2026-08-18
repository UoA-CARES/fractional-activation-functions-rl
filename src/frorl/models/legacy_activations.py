"""Compatibility implementations recovered from the earlier development code.

These classes are not used by the default experiment registry. They are retained
for traceability when comparing old checkpoints or reproducing intermediate runs.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class FractionalReLUPositive(nn.Module):
    """Original positive-only form; intentionally has no Gamma normalization."""
    def __init__(self, a=0.1, epsilon=1e-6, clip_value=1.0):
        super().__init__(); self.a=a; self.epsilon=epsilon; self.clip_value=clip_value
    def forward(self, x):
        x_clamped = torch.clamp(x, min=1e-6)
        positive_part = torch.pow(x_clamped, 1-self.a)
        return torch.where(x > 0, positive_part, torch.zeros_like(x))


class FPReLU2(nn.Module):
    """Original FPReLU2 with unbounded softplus negative parameter."""
    def __init__(self, a=0.2, parameter_init=1.5):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))
        self.p_raw = nn.Parameter(torch.tensor(float(parameter_init)))
        self.g = math.gamma(2-float(a))
    def p(self): return F.softplus(self.p_raw)
    def forward(self, x):
        pos=(x>0).float()*x.clamp_min(1e-8).pow(1-self.a)/self.g
        neg=(x<0).float()*(-self.p()/self.g)*(-x).clamp_min(1e-8).pow(1-self.a)
        return pos+neg


class FLReLU2(nn.Module):
    """Original learnable-small-slope FLReLU2."""
    def __init__(self, a=0.1, k_init=0.1):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))
        self.k=nn.Parameter(torch.tensor(float(k_init)))
        self.g=math.gamma(2-float(a))
    def forward(self,x):
        pos=(x>0).float()*x.clamp_min(1e-8).pow(1-self.a)/self.g
        neg=(x<0).float()*(-torch.clamp(self.k,0.01,0.3)/self.g)*(-x).clamp_min(1e-8).pow(1-self.a)
        return pos+neg


class FractionalSwish(nn.Module):
    def __init__(self,a=0.1): super().__init__(); self.a=a
    def forward(self,x):
        alpha=torch.clamp(torch.tensor(self.a,device=x.device,dtype=x.dtype),0.,1.)
        sig=torch.sigmoid(x); swish=x*sig
        return swish+alpha*sig*(1-swish)


class FractionalSwishBeta(nn.Module):
    def __init__(self,a=0.1,beta_init=1.0):
        super().__init__(); self.a=a; self.beta_raw=nn.Parameter(torch.tensor(float(beta_init)))
    def forward(self,x):
        alpha=torch.clamp(torch.tensor(self.a,device=x.device,dtype=x.dtype),0.,1.)
        beta=torch.clamp(F.softplus(self.beta_raw),0.1,10.0)
        sig=torch.sigmoid(beta*x); swish=x*sig
        return swish+alpha*sig*(1-swish)


class FractionalGELU(nn.Module):
    def __init__(self,a=0.1,h=0.5,n_iter=3):
        super().__init__(); self.a=a; self.h=h; self.n_iter=n_iter
        self.register_buffer("gamma_consts",torch.tensor([math.gamma(i+1) for i in range(n_iter)],dtype=torch.float32))
    def _gelu_tanh_part(self,z):
        return z*(1+torch.tanh(0.7978845608028654*(z+0.044715*z**3)))
    def forward(self,x):
        alpha=torch.clamp(torch.tensor(self.a,device=x.device,dtype=x.dtype),0.,2.)
        h=torch.tensor(self.h,device=x.device,dtype=x.dtype)
        gamma_a_plus_one=torch.exp(torch.lgamma(1+alpha)); gc=self.gamma_consts.to(x)
        out=torch.zeros_like(x)
        for i in range(self.n_iter):
            z=x-i*h; denom=gc[i]*torch.exp(torch.lgamma(1-i+alpha)); coeff=gamma_a_plus_one/(denom+1e-8)
            out=out+(-1. if i%2 else 1.)*coeff*self._gelu_tanh_part(z)
        return out/((h**alpha)*2.)
