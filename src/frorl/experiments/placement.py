from __future__ import annotations
from frorl.models.activations import activation_factory

PLACEMENTS=("both","all-actor","all-critic","all-both","first-actor","first-both")

def activation_layout(name:str,alpha:float|None,layers:int,placement:str):
    if layers not in (1,2): raise ValueError("layers must be 1 or 2")
    if layers==1:
        if placement not in ("both","all-both"): raise ValueError("one-layer experiments use placement='both'")
        f=activation_factory(name,alpha); return [f],[f]
    if placement not in PLACEMENTS[1:]: raise ValueError(f"two-layer placement must be one of {PLACEMENTS[1:]}")
    chosen=activation_factory(name,alpha); relu=activation_factory("relu")
    if placement=="all-actor": return [chosen,chosen],[relu,relu]
    if placement=="all-critic": return [relu,relu],[chosen,chosen]
    if placement=="all-both": return [chosen,chosen],[chosen,chosen]
    if placement=="first-actor": return [chosen,relu],[relu,relu]
    if placement=="first-both": return [chosen,relu],[chosen,relu]
    raise AssertionError(placement)
