"""AUC utilities matching the normalized learning-curve metric used by the study."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def normalized_auc(steps, rewards, budget: float, anchor_at_zero: bool = True) -> float:
    steps=np.asarray(steps,dtype=float); rewards=np.asarray(rewards,dtype=float)
    if steps.size != rewards.size or steps.size == 0: raise ValueError('steps and rewards must be non-empty and equal length')
    order=np.argsort(steps); steps,rewards=steps[order],rewards[order]
    if anchor_at_zero and steps[0] > 0:
        steps=np.concatenate([[0.],steps]); rewards=np.concatenate([[rewards[0]],rewards])
    mask=steps <= budget; steps,rewards=steps[mask],rewards[mask]
    if steps.size==0: raise ValueError('no observations inside budget')
    if steps[-1] < budget:
        steps=np.concatenate([steps,[budget]]); rewards=np.concatenate([rewards,[rewards[-1]]])
    keep=np.concatenate([[True],np.diff(steps)>0]); steps,rewards=steps[keep],rewards[keep]
    if steps.size<2: raise ValueError('need at least two distinct step values')
    return float(np.trapezoid(rewards,steps))/float(budget)


def read_eval(path: Path):
    import csv
    s=[]; r=[]
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            s.append(float(row['total_steps'])); r.append(float(row.get('mean_reward',row.get('episode_reward'))))
    return s,r


def main():
    p=argparse.ArgumentParser();p.add_argument('eval_csv',type=Path);p.add_argument('--budget',type=float,default=1_000_000);a=p.parse_args()
    s,r=read_eval(a.eval_csv); print(f'{normalized_auc(s,r,a.budget):.10g}')
if __name__=='__main__': main()
