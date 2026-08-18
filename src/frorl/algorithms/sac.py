from __future__ import annotations
import copy, logging, os
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from frorl.utils.utils import evaluating, soft_update_params

class SAC:
    def __init__(self,actor_network,critic_network,actor_lr,critic_lr,alpha_lr,gamma,tau,reward_scale,policy_update_freq,target_update_freq,device):
        self.type="policy"; self.device=device; self.actor_net=actor_network.to(device); self.critic_net=critic_network.to(device)
        self.target_critic_net=copy.deepcopy(self.critic_net).to(device); self.target_critic_net.eval()
        self.gamma=gamma; self.tau=tau; self.reward_scale=reward_scale; self.learn_counter=0
        self.policy_update_freq=policy_update_freq; self.target_update_freq=target_update_freq
        self.target_entropy=-float(self.actor_net.mean.out_features)
        self.actor_net_optimiser=torch.optim.Adam(self.actor_net.parameters(),lr=actor_lr)
        self.critic_net_optimiser=torch.optim.Adam(self.critic_net.parameters(),lr=critic_lr)
        self.log_alpha=torch.tensor(np.log(1.0),device=device,requires_grad=True)
        self.log_alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=alpha_lr)
    @property
    def alpha(self): return self.log_alpha.exp()
    def select_action_from_policy(self,state,evaluation=False,noise_scale=0.0):
        del noise_scale; self.actor_net.eval()
        with torch.no_grad():
            st=torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
            if evaluation:
                mean,_=self.actor_net(st); action=torch.tanh(mean)*self.actor_net.max_action
            else: action,_=self.actor_net.sample(st)
            action=action.cpu().numpy().flatten()
        self.actor_net.train(); return action
    def _update_critic(self,s,a,r,ns,d):
        with torch.no_grad():
            with evaluating(self.actor_net): na,nlog=self.actor_net.sample(ns)
            tq1,tq2=self.target_critic_net(ns,na); tq=torch.minimum(tq1,tq2)-self.alpha*nlog
            target=r*self.reward_scale+self.gamma*(1.-d)*tq
        q1,q2=self.critic_net(s,a); l1=F.mse_loss(q1,target); l2=F.mse_loss(q2,target); loss=l1+l2
        self.critic_net_optimiser.zero_grad(); loss.backward(); self.critic_net_optimiser.step(); return l1.item(),l2.item(),loss.item()
    def _update_actor_alpha(self,s):
        pi,log_pi=self.actor_net.sample(s)
        with evaluating(self.critic_net): q1,q2=self.critic_net(s,pi)
        actor_loss=(self.alpha.detach()*log_pi-torch.minimum(q1,q2)).mean()
        self.actor_net_optimiser.zero_grad(); actor_loss.backward(); self.actor_net_optimiser.step()
        alpha_loss=-(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad(); alpha_loss.backward(); self.log_alpha_optimizer.step(); return actor_loss.item(),alpha_loss.item()
    def train_policy(self,memory,batch_size)->dict[str,Any]:
        self.learn_counter+=1; s,a,r,ns,d=memory.sample(batch_size)
        ts=lambda x: torch.as_tensor(x,dtype=torch.float32,device=self.device)
        s,a,r,ns,d=map(ts,(s,a,r,ns,d)); l1,l2,lt=self._update_critic(s,a,r,ns,d)
        info={"critic_loss_one":l1,"critic_loss_two":l2,"critic_loss":lt}
        if self.learn_counter%self.policy_update_freq==0:
            al,tl=self._update_actor_alpha(s); info.update(actor_loss=al,alpha_loss=tl,alpha=self.alpha.item())
        if self.learn_counter%self.target_update_freq==0: soft_update_params(self.critic_net,self.target_critic_net,self.tau)
        return info
    def save_models(self,filepath,filename):
        os.makedirs(filepath,exist_ok=True); torch.save(self.actor_net.state_dict(),f"{filepath}/{filename}_actor.pth"); torch.save(self.critic_net.state_dict(),f"{filepath}/{filename}_critic.pth"); logging.info("SAC models saved")
    def load_models(self,filepath,filename):
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pth",map_location=self.device)); self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pth",map_location=self.device))
