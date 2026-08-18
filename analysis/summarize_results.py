"""Scan result folders and summarize normalized AUC per run."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from auc import normalized_auc,read_eval

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--budget',type=float,default=1_000_000);p.add_argument('--output',type=Path,default=Path('results/auc_summary.csv'));a=p.parse_args()
    rows=[]
    for eval_file in a.results.rglob('eval.csv'):
        cfg_file=eval_file.parent/'config.json'
        if not cfg_file.exists(): continue
        cfg=json.loads(cfg_file.read_text()); s,r=read_eval(eval_file)
        rows.append({**{k:cfg.get(k) for k in ('algo','env','activation','alpha','layers','placement','seed')},'auc':normalized_auc(s,r,a.budget),'path':str(eval_file.parent)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=['algo','env','activation','alpha','layers','placement','seed','auc','path']
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f'wrote {len(rows)} rows to {a.output}')
if __name__=='__main__': main()
