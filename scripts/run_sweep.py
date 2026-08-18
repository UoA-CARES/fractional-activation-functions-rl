#!/usr/bin/env python
import argparse,subprocess,sys
FRACTIONAL={'frelu','flrelu','fprelu','fgelu','fswish'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--algo',choices=['TD3','SAC'],required=True); p.add_argument('--env',required=True); p.add_argument('--activation',required=True)
    p.add_argument('--layers',type=int,choices=[1,2],default=2); p.add_argument('--placement',default='all-both'); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]); p.add_argument('--alphas',type=float,nargs='+',default=[.1,.2,.3,.4,.5]); p.add_argument('--run',action='store_true'); p.add_argument('--steps',type=int,default=1_000_000)
    a=p.parse_args(); alphas=a.alphas if a.activation.lower() in FRACTIONAL else [None]
    for alpha in alphas:
        for seed in a.seeds:
            cmd=[sys.executable,'scripts/train.py','--algo',a.algo,'--env',a.env,'--activation',a.activation,'--layers',str(a.layers),'--placement',a.placement,'--seed',str(seed),'--steps',str(a.steps)]
            if alpha is not None: cmd += ['--alpha',str(alpha)]
            print(' '.join(cmd),flush=True)
            if a.run: subprocess.run(cmd,check=True)
if __name__=='__main__': main()
