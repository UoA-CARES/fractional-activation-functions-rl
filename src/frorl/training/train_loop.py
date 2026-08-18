from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np


def evaluate(env,agent,episodes=10):
    returns=[]
    for _ in range(episodes):
        obs,_=env.reset(); done=False; total=0.0
        while not done:
            action=agent.select_action_from_policy(obs,evaluation=True); obs,r,term,trunc,_=env.step(action); total+=float(r); done=term or trunc
        returns.append(total)
    return float(np.mean(returns)),float(np.std(returns))


def train(env,eval_env,agent,memory,total_steps,start_steps,batch_size,eval_every,eval_episodes,out_dir,noise_scale=0.1):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    train_csv=out/'train.csv'; eval_csv=out/'eval.csv'
    with train_csv.open('w',newline='') as ft, eval_csv.open('w',newline='') as fe:
        tw=csv.writer(ft); ew=csv.writer(fe); tw.writerow(['total_steps','episode','episode_reward','episode_length']); ew.writerow(['total_steps','mean_reward','std_reward'])
        obs,_=env.reset(); ep_ret=0.0; ep_len=0; ep=0
        for step in range(1,total_steps+1):
            if step<=start_steps:
                action=env.action_space.sample() if hasattr(env,'action_space') else env.sample_action()
            else: action=agent.select_action_from_policy(obs,evaluation=False,noise_scale=noise_scale)
            nxt,reward,term,trunc,_=env.step(action); done=term or trunc
            memory.add(obs,action,reward,nxt,done); obs=nxt; ep_ret+=float(reward); ep_len+=1
            if step>start_steps and len(memory)>=batch_size: agent.train_policy(memory,batch_size)
            if done:
                tw.writerow([step,ep,ep_ret,ep_len]); ft.flush(); obs,_=env.reset(); ep_ret=0.; ep_len=0; ep+=1
            if step%eval_every==0:
                mean,std=evaluate(eval_env,agent,eval_episodes); ew.writerow([step,mean,std]); fe.flush(); print(f"step={step} eval_mean={mean:.3f} eval_std={std:.3f}")
    agent.save_models(str(out),'model')
