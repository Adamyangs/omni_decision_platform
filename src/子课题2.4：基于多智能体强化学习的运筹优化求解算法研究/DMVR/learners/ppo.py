import torch as th
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

import time

class PPO:
    def __init__(self, agent, gamma, lmbda, device, 
                 lr, update_epochs, agent_num, steps, 
                 envs_num, max_grad_norm, high_envs,num_minibatches,
                 clip_vloss, clip_coef, norm_adv, vf_coef):
        self.agent = agent
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        self.lr = lr
        self.update_epochs = update_epochs
        self.agent_num = agent_num
        self.steps = steps
        self.envs_num = envs_num
        self.max_grad_norm = max_grad_norm
        self.high_envs = high_envs
        self.num_minibatches=num_minibatches
        self.clip_vloss = clip_vloss
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.vf_coef = vf_coef
        
        self.optimizer = th.optim.Adam(self.agent.parameters(), lr=lr)
        
        agent.backbone.decoder.init()
        
    def update(self, high_buffer, update, num_updates):
        th.cuda.empty_cache()
        
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * self.lr
        print("lrnow: ",lrnow)
        self.optimizer.param_groups[0]["lr"] = lrnow
        obs, actions, logprobs, high_rewards, values,tbd_node_idxs = high_buffer.sample(1)
        
        # requires_grad = False
        obs = obs[0]                                                              #[steps, obs_dict]  array
        actions = actions[0].clone().detach()                                     #[steps, num_envs, agent_num]
        logprobs = logprobs[0].clone().detach()                                   #[steps, num_envs, agent_num]
        high_rewards = th.tensor(high_rewards[0], dtype=th.float).to(self.device) #[steps, num_envs, agent_num]
        values = values[0].clone().detach()                                       #[steps, num_envs, agent_num]
        tbd_node_idxs = tbd_node_idxs[0].long().clone()                           #[steps, num_envs, 1]
        
        with th.no_grad():
            returns, advantages, values = self.gae( 
                                               high_rewards,
                                               values,)
        
        # flatten the batch
        # 按照时间，把同一个step下的所有obs,按照key拼接在一起
        # just for test
        # b_obs=dict()
        # for k in obs[0].keys():
        #     b_obs[k]= np.concatenate([obs_[k] for obs_ in obs[:-1]])
        
        # obs[0]['tbd_node_idx'] = obs[0]['tbd_node_idx'].cpu().numpy()
        b_obs = {
            k: np.concatenate([obs_[k] for obs_ in obs]) for k in obs[0].keys()
        }                                                       #[num_envs*steps, obs_dict] (1024,50,2), (1024,50,2)...

        # Edited  第一个时刻的所有的环境cat, 第二个时刻的所有的环境的cat
        b_logprobs = logprobs.reshape(-1, self.agent_num)       #[num_envs*step, agent_num] logprobs[0] logprobs[1] logprobs[2]
        b_actions = actions.reshape(-1, self.agent_num)         #[num_envs*step, agent_num]
        b_advantages = advantages.reshape(-1, self.agent_num)   #[num_envs*step, agent_num]
        b_returns = returns.reshape(-1, self.agent_num)         #[num_envs*step, agent_num]
        b_values = values.reshape(-1, self.agent_num)           #[num_envs*step, agent_num]
        b_tbd_node_idxs = tbd_node_idxs.reshape(-1, 1)          #[num_envs*step, 1]

        # Optimizing the policy and value network
        assert self.envs_num % self.num_minibatches == 0
        envsperbatch = self.envs_num // self.num_minibatches
        envinds = np.arange(self.envs_num)
        flatinds = np.arange(self.steps * self.envs_num).reshape(self.steps, self.envs_num)

        clipfracs = []
        import time
        # t1=time.time()
        for _ in range(self.update_epochs):
            np.random.shuffle(envinds)
            # make mini-batch by envs_num
            for start in range(0, self.envs_num, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]  # mini batch env id
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                r_inds = np.tile(np.arange(envsperbatch), self.steps)
                
                cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}         #[256,50,2]
                encoder_state = self.agent.backbone.encode(cur_obs)            #[256,51,128], [256,1,128]
                
                
                action, high_actions_u, new_log_probs, newvalue, entropy, _ = self.agent.get_action_and_value_cached(
                    next_obs_multi = {k: v[mb_inds] for k, v in b_obs.items()},
                    action = b_actions[mb_inds],
                    state = (embedding[r_inds, :] for embedding in encoder_state),
                    tbd_node_idx=b_tbd_node_idxs[mb_inds],
                ) 
                # //action, logits, new_log_probs, high_actions_u, newvalue, entropy = self.agent.get_action_and_value_cached(
                # //    next_obs_multi = {k: v[mb_inds] for k, v in b_obs.items()},
                # //    action = b_actions[mb_inds],
                # //    state = (embedding[r_inds, :] for embedding in encoder_state),
                # //    tbd_node_idx=b_tbd_node_idxs[mb_inds],
                # //)
                
                logratio = new_log_probs - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # 防止梯度爆炸
                assert  (ratio.min() > 1e-8) and (ratio.max() < 1e8)
                
                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                # pg_loss1 = -mb_advantages * ratio
                # pg_loss2 = -mb_advantages * th.clamp(
                #     ratio, 1 - self.clip_coef, 1 + self.clip_coef
                # )
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                
                # pg_loss = th.max(pg_loss1, pg_loss2).mean()
                #? pg_loss = -th.min(pg_loss1, pg_loss2).mean()
                pg_loss = th.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1, self.agent_num)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # entropy_loss = entropy.mean()
                # loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                entropy_loss = entropy.mean()
                loss = pg_loss + v_loss * self.vf_coef - 0.1*entropy_loss
                # loss = pg_loss + v_loss * self.vf_coef

                th.cuda.empty_cache()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                th.cuda.empty_cache()
        # t2=time.time()
        # x=t2-t1
        
    def gae(self, high_rewards, values,):
        # bootstrap value if not done
        
        # next_value = self.agent.get_value_cached(next_obs_multi, encoder_state)  # Env_num x Agent_num
        advantages = th.zeros_like(high_rewards).to('cuda')  # steps(51) x Env_num x Agent_num
        lastgaelam = th.zeros(self.envs_num, self.agent_num).to('cuda')  # Env_num x Agent_num
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                nextnonterminal =  0 # 1.0 - next_done  # next_done: Env_num
                nextvalues = 0  # next_value  # Env_num x Agent_num
                # nextvalues = values[t]
            else:
                nextnonterminal = 1.0   #  - dones[t + 1]
                nextvalues = values[t + 1]  # B x T
            delta = high_rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.lmbda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
            
        return returns, advantages, values