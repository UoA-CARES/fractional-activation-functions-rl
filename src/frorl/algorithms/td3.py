from __future__ import annotations
import copy, logging, os
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from frorl.utils.utils import evaluating, soft_update_params

class TD3:
    def __init__(self,actor_network,critic_network,actor_lr,critic_lr,gamma,tau,policy_update_freq,device):
        self.type="policy"; self.device=device
        self.actor_net=actor_network.to(device); self.critic_net=critic_network.to(device)
        self.target_actor_net=copy.deepcopy(self.actor_net).to(device); self.target_actor_net.eval()
        self.target_critic_net=copy.deepcopy(self.critic_net).to(device); self.target_critic_net.eval()
        self.gamma=gamma; self.tau=tau; self.noise_clip=0.5; self.policy_noise=0.2
        self.learn_counter=0; self.policy_update_freq=policy_update_freq; self.action_num=self.actor_net.num_actions
        self.actor_net_optimiser=torch.optim.Adam(self.actor_net.parameters(),lr=actor_lr)
        self.critic_net_optimiser=torch.optim.Adam(self.critic_net.parameters(),lr=critic_lr)
    def select_action_from_policy(self,state,evaluation=False,noise_scale=0.1):
        self.actor_net.eval()
        with torch.no_grad():
            action=self.actor_net(torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)).cpu().numpy().flatten()
            if not evaluation:
                action=action+np.random.normal(0,noise_scale,size=self.action_num); action=np.clip(action,-1.,1.)
        self.actor_net.train(); return action
    def _update_critic(self,states,actions,rewards,next_states,dones):
        with torch.no_grad():
            next_actions=self.target_actor_net(next_states)
            noise=torch.clamp(self.policy_noise*torch.randn_like(next_actions),-self.noise_clip,self.noise_clip)
            next_actions=torch.clamp(next_actions+noise,-1.,1.)
            tq1,tq2=self.target_critic_net(next_states,next_actions); target=torch.minimum(tq1,tq2)
            q_target=rewards+self.gamma*(1.-dones)*target
        q1,q2=self.critic_net(states,actions); l1=F.mse_loss(q1,q_target); l2=F.mse_loss(q2,q_target); loss=l1+l2
        self.critic_net_optimiser.zero_grad(); loss.backward(); self.critic_net_optimiser.step()
        return l1.item(),l2.item(),loss.item()
    def _update_actor(self,states):
        actions=self.actor_net(states)
        with evaluating(self.critic_net): q1,_=self.critic_net(states,actions)
        loss=-q1.mean(); self.actor_net_optimiser.zero_grad(); loss.backward(); self.actor_net_optimiser.step(); return loss.item()
    def train_policy(self,memory,batch_size)->dict[str,Any]:
        self.learn_counter+=1; s,a,r,ns,d=memory.sample(batch_size)
        ts=lambda x: torch.as_tensor(x,dtype=torch.float32,device=self.device)
        s,a,r,ns,d=map(ts,(s,a,r,ns,d)); l1,l2,lt=self._update_critic(s,a,r,ns,d)
        info={"critic_loss_one":l1,"critic_loss_two":l2,"critic_loss":lt}
        if self.learn_counter%self.policy_update_freq==0:
            info["actor_loss"]=self._update_actor(s)
            soft_update_params(self.critic_net,self.target_critic_net,self.tau); soft_update_params(self.actor_net,self.target_actor_net,self.tau)
        return info
    def save_models(self,filepath,filename):
        os.makedirs(filepath,exist_ok=True); torch.save(self.actor_net.state_dict(),f"{filepath}/{filename}_actor.pth"); torch.save(self.critic_net.state_dict(),f"{filepath}/{filename}_critic.pth"); logging.info("TD3 models saved")
    def load_models(self,filepath,filename):
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pth",map_location=self.device)); self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pth",map_location=self.device))
