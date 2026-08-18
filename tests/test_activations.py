import torch
import pytest
from frorl.models.activations import ACTIVATIONS,activation_factory,FReLU,FLReLU,FPReLU

@pytest.mark.parametrize('name',ACTIVATIONS)
def test_activation_finite_and_shape(name):
    alpha=.3 if name.startswith('f') else None; m=activation_factory(name,alpha)(); x=torch.linspace(-3,3,101,requires_grad=True); y=m(x)
    assert y.shape==x.shape; assert torch.isfinite(y).all(); y.sum().backward(); assert x.grad is not None; assert torch.isfinite(x.grad).all()

@pytest.mark.parametrize('cls',[FReLU,FLReLU,FPReLU])
def test_fractional_rectifier_zero(cls):
    y=cls(.2)(torch.tensor([0.])); assert y.item()==0.0

def test_alpha_zero_frelu_equals_relu():
    x=torch.tensor([-2.,0.,.25,2.]); assert torch.allclose(FReLU(0.0)(x),torch.relu(x),atol=1e-7)
