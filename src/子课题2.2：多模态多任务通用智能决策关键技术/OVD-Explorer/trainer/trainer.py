from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from utils.core import np_to_pytorch_batch
from torch.nn import functional as F

import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
N_Q = 20

def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    # print("(input, target, tau, weight, expanded_input, expanded_target", input, target, tau, weight, expanded_input, expanded_target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    # print("L, sign, rho", L, sign, rho,  rho.sum(dim=-1).mean())
    return rho.sum(dim=-1).mean()

def quantile_huber_loss(x, y, kappa=1):

    batch_size = x.shape[0] 
    num_quant = x.shape[1]

    #Get x and y to repeat here
    x = x.unsqueeze(2).repeat(1,1,num_quant)
    y = y.unsqueeze(2).repeat(1,1,num_quant).transpose(1,2)

    tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant).cuda() + 0.5 / num_quant
    tau_hat = tau_hat.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,num_quant)
    
    diff = y-x

    if kappa == 0:
        huber_loss = diff.abs()
    else:
        huber_loss = 0.5 * diff.abs().clamp(min=0.0, max=kappa).pow(2)
        huber_loss += kappa * (diff.abs() - diff.abs().clamp(min=0.0, max=kappa))

    quantile_loss = (tau_hat - (diff < 0).float()).abs() * huber_loss

    return quantile_loss.mean(2).mean(0).sum()
    
class SACTrainer(object):
    def __init__(
            self,
            policy_producer,
            q_producer,

            action_space=None,

            discount=0.99,
            train_num=1,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            use_aleatoric=False,
            redq=False,
            tau_type="iqn",
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.num_quantiles = N_Q
        self.tau_type = tau_type
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.use_aleatoric = use_aleatoric
        self.redq = redq
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = - \
                    np.prod(action_space.shape).item()

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount
        self.reward_scale = reward_scale

        """
        The class mutable state
        """
        self.policy = policy_producer()
        self.target_policy = policy_producer().cuda()
        if self.redq:
            self.qf10s = [q_producer() for _ in range(10)]
            self.target_qf10s = [q_producer() for _ in range(10)]
            self.qf10s_optimizer = [optimizer_class(
                self.qf10s[i].parameters(),
                lr=qf_lr,
            ) for i in range(10)]
            self.random_index = np.random.randint(0, 10, 2)
        else:
            self.qf1 = q_producer()
            self.qf2 = q_producer()
            self.target_qf1 = q_producer()
            self.target_qf2 = q_producer()
            self.qf1_optimizer = optimizer_class(
                self.qf1.parameters(),
                lr=qf_lr,
            )
            self.qf2_optimizer = optimizer_class(
                self.qf2.parameters(),
                lr=qf_lr,
            )

        if self.use_automatic_entropy_tuning:
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        # self.qf1_optimizer = optimizer_class(
        #     self.qf1.parameters(),
        #     lr=qf_lr,
        # )
        # self.qf2_optimizer = optimizer_class(
        #     self.qf2.parameters(),
        #     lr=qf_lr,
        # )
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, np_batch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)
    
    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            # if fp is None:
            #     fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        # print(obs)
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi +
                            self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            # print("alpha_loss.backward()")
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        if not self.redq:
            if not self.use_aleatoric:
                """
                QF Loss
                """
                q_new_actions = torch.min(
                    self.qf1(obs, new_obs_actions),
                    self.qf2(obs, new_obs_actions),
                )
                q1_pred = self.qf1(obs, actions)
                q2_pred = self.qf2(obs, actions)
                # Make sure policy accounts for squashing
                # functions like tanh correctly!
                new_next_actions, _, _, new_log_pi, *_ = self.policy(
                    next_obs, reparameterize=True, return_log_prob=True,
                )
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                ) - alpha * new_log_pi

                q_target = self.reward_scale * rewards + \
                    (1. - terminals) * self.discount * target_q_values
                qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
                qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

                """
                Update networks
                """
                self.qf1_optimizer.zero_grad()
                qf1_loss.backward()
                # print("qf1_loss.backward()")
                self.qf1_optimizer.step()

                self.qf2_optimizer.zero_grad()
                qf2_loss.backward()
                # print("qf2_loss.backward()")
                self.qf2_optimizer.step()
            else:
                """
                QF Loss
                """
                # Make sure policy accounts for squashing
                # functions like tanh correctly!
                tau_n, tau_hat_n, presum_tau_n = self.get_tau(obs, new_obs_actions, fp=None)
                q_new_actions = torch.min(
                    self.qf1(obs, new_obs_actions, tau_hat_n),
                    self.qf2(obs, new_obs_actions, tau_hat_n),
                )
                with torch.no_grad():
                    # print(next_obs)
                    new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
                        next_obs, reparameterize=True, return_log_prob=True,
                    )
                    next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, new_next_actions, fp=None)

                    target_q_values = torch.min(
                        self.target_qf1(next_obs, new_next_actions, next_tau_hat),
                        self.target_qf2(next_obs, new_next_actions, next_tau_hat),
                    ) - alpha * new_log_pi
                    # print("target_q_values, new_log_pi:", target_q_values, target_q_values.shape, new_log_pi[0], new_log_pi.shape, rewards[0], rewards.shape)
                    
                    q_target = self.reward_scale * rewards + \
                        (1. - terminals) * self.discount * target_q_values
                # N_QUANT = 20
                # QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]
                # QUANTS_TARGET = (np.linspace(0.0, 1.0, N_QUANT + 1)[:-1] + QUANTS)/2
                # qf1 loss
                # q_target = q_target.unsqueeze(1) # (m , 1, N_QUANT)
                tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=None)
                q1_pred = self.qf1(obs, actions, tau_hat)
                q2_pred = self.qf2(obs, actions, tau_hat)
                # print("q1_pred", q1_pred[0], q1_pred.shape)
                # print("q2_pred", q2_pred[0], q2_pred.shape)
                # print("q_target", q_target[0], q_target.shape)

                # quantile Huber loss
                qf1_loss = quantile_regression_loss(q1_pred, q_target.detach(), tau_hat, next_presum_tau)
                qf2_loss = quantile_regression_loss(q2_pred, q_target.detach(), tau_hat, next_presum_tau)
                # print("obs, actions, tau_hat, next_presum_tau", obs, actions, tau_hat, next_presum_tau)
                # u = q_target.detach() - q1_pred # (m, N_QUANT, N_QUANT)
                # tau = torch.FloatTensor(QUANTS_TARGET).view(1, -1).cuda() # (1, N_QUANT, 1)
                # # print("u", u, u.shape)
                # # print("tau", tau, tau.shape)
                # weight = torch.abs(tau - u.le(0.).float()) # (m, N_QUANT, N_QUANT)
                # # weight = torch.ones(q1_pred.shape).cuda()
                # print("weight", weight[0], weight.shape)
                
                # qf1_loss = nn.functional.smooth_l1_loss(q1_pred, q_target.detach(), reduction='none')
                # print("qf1_loss", qf1_loss[0], qf1_loss.shape)
                # qf1_loss = torch.mean(weight * qf1_loss, dim=1)
                # qf1_loss = qf1_loss.mean()
                # # print('1 qf1_loss',qf1_loss.shape, qf1_loss)
                
                # # calc importance weighted loss
                # # b_w, b_idxes = np.ones_like(rewards.cpu().numpy()), None
                # # b_w = torch.Tensor(b_w).cuda()
                # # # loos = b_w * qf1_loss
                # # print('2 b_w * qf1_loss',(b_w * qf1_loss).shape, b_w)
                # # qf1_loss = torch.mean(b_w * qf1_loss)

                # qf2_loss = nn.functional.smooth_l1_loss(q2_pred, q_target.detach(), reduction='none')
                # print("qf2_loss", qf2_loss[0], qf2_loss.shape)
                # qf2_loss = torch.mean(weight * qf2_loss, dim=1).mean()
                # # print('1 qf2_loss',qf2_loss.shape, qf2_loss)
                
                # calc importance weighted loss
                # b_w, b_idxes = np.ones_like(rewards.cpu().numpy()), None
                # b_w = torch.Tensor(b_w).cuda()
                # # loos = b_w * qf2_loss
                # # print('2 b_w * qf2_loss',(b_w * qf2_loss).shape, b_w)
                # qf2_loss = torch.mean(b_w * qf2_loss)

                # qf1_loss = self.qf_criterion(q1_pred, q_target.detach(), use_aleatoric)
                # qf2_loss = self.qf_criterion(q2_pred, q_target.detach(), use_aleatoric)
                # print("loss:", qf1_loss, qf2_loss)
                """
                Update networks
                """
                self.qf1_optimizer.zero_grad()
                qf1_loss.backward()
                # print("qf1_loss.backward()")
                self.qf1_optimizer.step()

                self.qf2_optimizer.zero_grad()
                qf2_loss.backward()
                # print("qf2_loss.backward()")
                self.qf2_optimizer.step()
        else:
            if not self.use_aleatoric:
                """
                QF Loss
                """
                q_new_actions = torch.mean(torch.stack(
                    [self.qf10s[i](obs, new_obs_actions) for i in range(10)]
                ))
                q10s_pred = [self.qf10s[i](obs, actions) for i in range(10)]
                # Make sure policy accounts for squashing
                # functions like tanh correctly!
                new_next_actions, _, _, new_log_pi, *_ = self.policy(
                    next_obs, reparameterize=True, return_log_prob=True,
                )
                self.random_index = np.random.randint(0, 10, 2)
                target_q_values = torch.min(torch.stack([
                    self.target_qf10s[i](next_obs, new_next_actions) for i in self.random_index
                ])
                ) - alpha * new_log_pi

                q_target = self.reward_scale * rewards + \
                    (1. - terminals) * self.discount * target_q_values
                qf10s_loss = [self.qf_criterion(q10s_pred[i], q_target.detach()) for i in range(10)]

                """
                Update networks
                """
                for i in range(10):
                    self.qf10s_optimizer[i].zero_grad()
                    qf10s_loss[i].backward(retain_graph=True)
                    self.qf10s_optimizer[i].step()
            else:
                """
                QF Loss
                """
                # Make sure policy accounts for squashing
                # functions like tanh correctly!
                tau_n, tau_hat_n, presum_tau_n = self.get_tau(obs, new_obs_actions, fp=None)
                q_new_actions = torch.mean(torch.stack(
                    [self.qf10s[i](obs, new_obs_actions, tau_hat_n) for i in range(10)]
                ))
                self.random_index = np.random.randint(0, 10, 2)
                with torch.no_grad():
                    new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
                        next_obs, reparameterize=True, return_log_prob=True,
                    )
                    next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, new_next_actions, fp=None)

                    target_q_values = torch.min(torch.stack([
                        self.target_qf10s[i](next_obs, new_next_actions, next_tau_hat) for i in self.random_index
                    ])
                    ) - alpha * new_log_pi
                    
                    q_target = self.reward_scale * rewards + \
                        (1. - terminals) * self.discount * target_q_values
                tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=None)
                q10s_pred = [self.qf10s[i](obs, actions, tau_hat) for i in range(10)]
                qf10s_loss = [quantile_regression_loss(q10s_pred[i], q_target.detach(), tau_hat, next_presum_tau) for i in range(10)]

                """
                Update networks
                """
                for i in range(10):
                    self.qf10s_optimizer[i].zero_grad()
                    qf10s_loss[i].backward(retain_graph=True)
                    self.qf10s_optimizer[i].step()
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # print("policy_loss.backward()")
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.policy, self.target_policy, self.soft_target_tau)
            if not self.redq:
                ptu.soft_update_from_to(
                    self.qf1, self.target_qf1, self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2, self.target_qf2, self.soft_target_tau
                )
            else:
                for i in range(10):
                    ptu.soft_update_from_to(
                        self.qf10s[i], self.target_qf10s[i], self.soft_target_tau
                    )   

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            if self.redq:
                q1_pred = torch.mean(torch.stack(q10s_pred)).unsqueeze(0)
                q2_pred = q10s_pred[0]
                qf1_loss = torch.mean(torch.stack(qf10s_loss)).unsqueeze(0)
                qf2_loss = qf10s_loss[1]
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            # print(q1_pred)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1
        

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        if not self.redq:
            return [
                self.policy,
                self.qf1,
                self.qf2,
                self.target_qf1,
                self.target_qf2,
            ]
        else:
            return [
                self.policy,
                self.qf10s[0],
                self.qf10s[1],
                self.qf10s[2],
                self.qf10s[3],
                self.qf10s[4],
                self.qf10s[5],
                self.qf10s[6],
                self.qf10s[7],
                self.qf10s[8],
                self.qf10s[9],
                self.target_qf10s[0],
                self.target_qf10s[1],
                self.target_qf10s[2],
                self.target_qf10s[3],
                self.target_qf10s[4],
                self.target_qf10s[5],
                self.target_qf10s[6],
                self.target_qf10s[7],
                self.target_qf10s[8],
                self.target_qf10s[9],
            ]

    def get_snapshot(self):
        if not self.redq:
            snap = dict(
                policy_state_dict=self.policy.state_dict(),
                policy_optim_state_dict=self.policy_optimizer.state_dict(),

                qf1_state_dict=self.qf1.state_dict(),
                qf1_optim_state_dict=self.qf1_optimizer.state_dict(),
                target_qf1_state_dict=self.target_qf1.state_dict(),

                qf2_state_dict=self.qf2.state_dict(),
                qf2_optim_state_dict=self.qf2_optimizer.state_dict(),
                target_qf2_state_dict=self.target_qf2.state_dict(),

                log_alpha=self.log_alpha,
                alpha_optim_state_dict=self.alpha_optimizer.state_dict(),

                eval_statistics=self.eval_statistics,
                _n_train_steps_total=self._n_train_steps_total,
                _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )
        else:
            snap = dict(
                policy_state_dict=self.policy.state_dict(),
                policy_optim_state_dict=self.policy_optimizer.state_dict(),

                qf1_state_dict=self.qf10s[0].state_dict(),
                qf1_optim_state_dict=self.qf10s_optimizer[0].state_dict(),
                target_qf1_state_dict=self.target_qf10s[0].state_dict(),

                qf2_state_dict=self.qf10s[1].state_dict(),
                qf2_optim_state_dict=self.qf10s_optimizer[1].state_dict(),
                target_qf2_state_dict=self.target_qf10s[1].state_dict(),

                qf3_state_dict=self.qf10s[2].state_dict(),
                qf3_optim_state_dict=self.qf10s_optimizer[2].state_dict(),
                target_qf3_state_dict=self.target_qf10s[2].state_dict(),

                qf4_state_dict=self.qf10s[3].state_dict(),
                qf4_optim_state_dict=self.qf10s_optimizer[3].state_dict(),
                target_qf4_state_dict=self.target_qf10s[3].state_dict(),

                qf5_state_dict=self.qf10s[4].state_dict(),
                qf5_optim_state_dict=self.qf10s_optimizer[4].state_dict(),
                target_qf5_state_dict=self.target_qf10s[4].state_dict(),

                qf6_state_dict=self.qf10s[5].state_dict(),
                qf6_optim_state_dict=self.qf10s_optimizer[5].state_dict(),
                target_qf6_state_dict=self.target_qf10s[5].state_dict(),

                qf7_state_dict=self.qf10s[6].state_dict(),
                qf7_optim_state_dict=self.qf10s_optimizer[6].state_dict(),
                target_qf7_state_dict=self.target_qf10s[6].state_dict(),

                qf8_state_dict=self.qf10s[7].state_dict(),
                qf8_optim_state_dict=self.qf10s_optimizer[7].state_dict(),
                target_qf8_state_dict=self.target_qf10s[7].state_dict(),

                qf9_state_dict=self.qf10s[8].state_dict(),
                qf9_optim_state_dict=self.qf10s_optimizer[8].state_dict(),
                target_qf9_state_dict=self.target_qf10s[8].state_dict(),

                qf10_state_dict=self.qf10s[9].state_dict(),
                qf10_optim_state_dict=self.qf10s_optimizer[9].state_dict(),
                target_qf10_state_dict=self.target_qf10s[9].state_dict(),

                log_alpha=self.log_alpha,
                alpha_optim_state_dict=self.alpha_optimizer.state_dict(),

                eval_statistics=self.eval_statistics,
                _n_train_steps_total=self._n_train_steps_total,
                _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )
        return snap

    def restore_from_snapshot(self, ss):
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)


        log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']

        self.log_alpha.data.copy_(log_alpha)
        self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
        if not self.redq:
            qf1_state_dict, qf1_optim_state_dict = ss['qf1_state_dict'], ss['qf1_optim_state_dict']
            target_qf1_state_dict = ss['target_qf1_state_dict']

            self.qf1.load_state_dict(qf1_state_dict)
            self.qf1_optimizer.load_state_dict(qf1_optim_state_dict)
            self.target_qf1.load_state_dict(target_qf1_state_dict)

            qf2_state_dict, qf2_optim_state_dict = ss['qf2_state_dict'], ss['qf2_optim_state_dict']
            target_qf2_state_dict = ss['target_qf2_state_dict']

            self.qf2.load_state_dict(qf2_state_dict)
            self.qf2_optimizer.load_state_dict(qf2_optim_state_dict)
            self.target_qf2.load_state_dict(target_qf2_state_dict)
        else:  
            for i in range(10):
                qf1_state_dict, qf1_optim_state_dict = ss[f'qf{i+1}_state_dict'], ss[f'qf{i+1}_optim_state_dict']
                target_qf1_state_dict = ss[f'target_qf{i+1}_state_dict']

                self.qf10s[i].load_state_dict(qf1_state_dict)
                self.qf10s_optimizer[i].load_state_dict(qf1_optim_state_dict)
                self.target_qf10s[i].load_state_dict(target_qf1_state_dict)
