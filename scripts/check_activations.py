#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from frorl.models.activations import ACTIVATIONS,activation_factory
x=torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True)
for name in ACTIVATIONS:
    alpha=.2 if name.startswith('f') else None
    m=activation_factory(name,alpha)(); y=m(x); y.sum().backward(retain_graph=True)
    print(f'{name:8s}',y.detach().tolist(),'finite=',bool(torch.isfinite(y).all()))
