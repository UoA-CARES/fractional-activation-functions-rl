#!/usr/bin/env python
"""Generate or execute the complete activation comparison matrix."""
import argparse,subprocess,sys
BASE=['relu','lrelu','prelu','gelu','swish']
FRAC=['frelu','flrelu','fprelu','fgelu','fswish']
ALPHAS=[.1,.2,.3,.4,.5]
PLACEMENTS=['all-actor','all-critic','all-both','first-actor','first-both']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--algo',choices=['TD3','SAC'],required=True);p.add_argument('--env',required=True);p.add_argument('--layers',type=int,choices=[1,2],required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--run',action='store_true');p.add_argument('--steps',type=int,default=1_000_000);a=p.parse_args()
    jobs=[]
    for activation in BASE+FRAC:
        alphas=ALPHAS if activation in FRAC else [None]
        placements=['both'] if a.layers==1 else PLACEMENTS
        for alpha in alphas:
            for place in placements:
                for seed in a.seeds:
                    cmd=[sys.executable,'scripts/train.py','--algo',a.algo,'--env',a.env,'--activation',activation,'--layers',str(a.layers),'--placement',place,'--seed',str(seed),'--steps',str(a.steps)]
                    if alpha is not None: cmd += ['--alpha',str(alpha)]
                    jobs.append(cmd)
    print(f'# {len(jobs)} jobs')
    for cmd in jobs:
        print(' '.join(cmd),flush=True)
        if a.run: subprocess.run(cmd,check=True)
if __name__=='__main__': main()
