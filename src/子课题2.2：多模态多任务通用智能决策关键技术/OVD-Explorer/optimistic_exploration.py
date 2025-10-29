import torch
import utils.pytorch_util as ptu
import torch.optim as optim
from trainer.policies import TanhNormal
import math
import numpy as np
from datetime import datetime 
import time
from tbrecorder import Writer
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
FIXINGALPHA = False
DETERMINSTIC = True 
SAMPLEEXP = True
PRINTDETAILEDLOG = False 
DETERMINSTICEXPLORATION = False 
PRETANHADD = True
NOONEALEA = True
ALEANOSQRT = True
print("config", FIXINGALPHA, DETERMINSTIC, SAMPLEEXP, DETERMINSTICEXPLORATION, PRETANHADD, NOONEALEA, ALEANOSQRT)
ALEOSCALE = 6
tau_type = 'iqn'
N_Q = 20
num_quantiles=N_Q 
s_=0
def get_tau(obs, actions, fp=None):
    if tau_type == 'fix':
        presum_tau = ptu.zeros(len(actions), num_quantiles) + 1. / num_quantiles
    elif tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
        presum_tau = ptu.rand(len(actions), num_quantiles) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
    elif tau_type == 'fqf':
        if fp is None:
            fp = fp
        presum_tau = fp(obs, actions)
    tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
    with torch.no_grad():
        tau_hat = ptu.zeros_like(tau)
        tau_hat[:, 0:1] = tau[:, 0:1] / 2.
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
    # print('actions, tau, tau_hat, presum_tau')
    # print(actions, tau, tau_hat, presum_tau)
    # print(actions.shape, tau.shape, tau_hat.shape, presum_tau.shape)
    return tau, tau_hat, presum_tau

def get_entropy_based_exploration_action(observation, sam = True, zo=None, policy=None, qfs=None, hyper_params=None):
    """
    input current observation
    output the exploration action according to entropy based policy.
    """
    global s_
    global z_auto
    t0 = datetime.now() 
    epsilon_12 = 1e-5
    s_ += 1
    # print(hyper_params)
    use_automatic_z_tuning = hyper_params['use_automatic_z_tuning']
    use_aleatoric = hyper_params['use_aleatoric']
    if use_automatic_z_tuning:
        z=zo.get_z()
        # print("z", z)
    else:
        z=hyper_params['z']
    sigma=hyper_params['sigma']
    use_quantile_cdf=hyper_params['use_quantile_cdf']
    alpha=hyper_params['alpha']
    beta=hyper_params['beta']
    # alpha_2 = hyper_params['alpha_2']
    # nor = hyper_params['nor']
    version = hyper_params['version']
    seed = hyper_params['seed']
    f = open(f"./logs/ovde{version}_{seed}_{alpha}.log", 'a+')
    print("-- \t", datetime.now() , file=f)

    writer = Writer.instance(version, seed)
    ob = ptu.from_numpy(observation)
    # print('observation', observation)
    # ob = observation
    sampled_action, pre_tanh_mu_T, _, _, std, _ = policy(ob)
    # print(sampled_action)
    t1 = datetime.now() 
    # pre_tanh_mu_T = sampled_action
    pre_tanh_mu_T.requires_grad_()
    tanh_mu_T = torch.tanh(pre_tanh_mu_T)
    g_cdf, g_pdf = None, None
    t3 = datetime.now() 
    if DETERMINSTIC:
        if SAMPLEEXP:
            d = TanhNormal(pre_tanh_mu_T, std)
            sampled_actions = []
            sampled_actions_minq = []
            for sample_item in range(1):
                pre_action_sampled = pre_tanh_mu_T
                # pre_action_sampled = d.sample()
                pre_action_sampled.requires_grad_()
                action_sampled = torch.tanh(pre_action_sampled)
                tau, tau_hat, presum_tau = get_tau(ob, sampled_action.unsqueeze(0), fp=None)
                tau_hat = tau_hat.squeeze(0)
                args = list(torch.unsqueeze(i, dim=0) for i in (ob, action_sampled.cuda(), tau_hat.cuda()))
                Q1 = qfs[0](*args)
                Q2 = qfs[1](*args)
                if use_aleatoric:
                    mu_Q = (torch.mean(Q1) + torch.mean(Q2)) / 2.0
                    uncertainties_epistemic = torch.abs(torch.mean(Q1) - torch.mean(Q2)) / 2.0
                    sigma_Q = uncertainties_epistemic
                    if not use_quantile_cdf:
                        if not NOONEALEA:
                            aleatoric = torch.var((Q1+Q2)/2, 1)
                            max_var = torch.max(torch.var(Q1, 1), torch.var(Q2, 1))
                            alea = aleatoric / max_var
                            uncertainties_aleatoric =  alea  * ALEOSCALE + 6
                        else :
                            aleatoric = torch.var((Q1+Q2)/2, 1)
                            max_var = torch.max(torch.var(Q1, 1), torch.var(Q2, 1))
                            alea = aleatoric# / max_var
                            uncertainties_aleatoric =  torch.sqrt(alea)#  * ALEOSCALE + 6
                            if not ALEANOSQRT:
                                uncertainties_aleatoric =  alea#  * ALEOSCALE + 6
                        if PRINTDETAILEDLOG:
                            print("- 2\t", "{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(uncertainties_epistemic.item(), uncertainties_aleatoric.item(), aleatoric.item(), max_var.item(), alea.item()), observation, file=f)
                        else:
                            pass
                        sigma = uncertainties_aleatoric
                    else:
                        if PRINTDETAILEDLOG:
                            print("- 2\t", "{:.4f}".format(uncertainties_epistemic.item()), observation, file=f)
                        else:
                            pass
                else:
                    mu_Q = (Q1 + Q2) / 2.0
                    sigma_Q = torch.abs(Q1 - Q2) / 2.0
                q_star_sampled = mu_Q + beta * sigma_Q
                qs_a = torch.autograd.grad(q_star_sampled, pre_action_sampled)
                qs_a = qs_a[0].cuda()
                qs_a /= torch.norm(qs_a)

                g_cdf, g_pdf = None, None
                if not use_quantile_cdf:
                    q_pi = torch.distributions.Normal(mu_Q - beta * sigma_Q, sigma)
                    g_cdf = q_pi.cdf(q_star_sampled) / z
                else:
                    assert use_aleatoric
                    mu_Q_quant, _ = torch.min(Q1, Q2).sort()
                    index_up = torch.where(mu_Q_quant[0]>=q_star_sampled)[0][0] if len(torch.where(mu_Q_quant[0]>=q_star_sampled)[0]) is not 0 else torch.tensor([-1])[0]
                    index_down = torch.where(mu_Q_quant[0]<=q_star_sampled)[0][-1]
                    if index_up == index_down:
                        if index_up != num_quantiles - 1:
                            g_cdf = torch.tensor([float(index_up + 1) / num_quantiles]).cuda()
                        else:
                            g_cdf = torch.tensor([1.0]).cuda()
                    elif index_up == -1:
                        g_cdf = torch.tensor([1.0]).cuda()
                    else:
                        rate = (q_star_sampled - mu_Q_quant[0][index_down]) / (mu_Q_quant[0][index_up] - mu_Q_quant[0][index_down])
                        g_cdf = torch.tensor([float(index_down + 1 + rate) / num_quantiles]).cuda()
                    g_cdf /= z
                g_pdf = torch.tensor([1.0]).cuda()
                p = (torch.log(g_cdf) + 1)  
                grad = p * qs_a
                if FIXINGALPHA:
                    if pre_action_sampled[0] * grad[0] > 0:
                        alpha *= -1
                t4 = datetime.now() 
                mu_E = pre_action_sampled.cuda() + alpha * grad
                dist = TanhNormal(mu_E, std)
                ac = dist.sample()
                ac_np = ptu.get_numpy(ac)
                sampled_actions.append(ac_np)
                t5 = datetime.now() 

                t6 = datetime.now() 
            ac_np = sampled_actions[0]
            return ac_np, {}
    else:
        grad = (torch.log(g_cdf) + 1 - torch.log(math.sqrt(2*math.pi)*std.detach()))*q_a
        Sigma_T = torch.pow(std, 2)
        denom = torch.sqrt(
            torch.sum(
                torch.mul(torch.pow(grad, 2), Sigma_T)
            )
        ) + 10e-6
        grad = torch.mul(Sigma_T, grad) / denom
        t4 = datetime.now() 
        if FIXINGALPHA:
            if pre_tanh_mu_T[0] * grad[0] > 0:
                alpha *= -1
    if PRETANHADD:
        mu_E = pre_tanh_mu_T.cuda() + alpha * grad
    else:
        mu_E = tanh_mu_T.cuda() + alpha * grad
    if not DETERMINSTICEXPLORATION:
        assert mu_E.shape == std.shape
        dist = TanhNormal(mu_E, std)
        ac = dist.sample()
        ac_np = ptu.get_numpy(ac)
    else: 
        ac_np = ptu.get_numpy(mu_E)
    t5 = datetime.now() 
    print("- L218 ep\t", ac_np, "mu({:.4f}, {:.4f}), g({:.4f}, {:.4f}), q({:.4f}, {:.4f}) c{:.4f}".format(mu_E[0].detach().item(), mu_E[1].detach().item(), grad[0].detach().item(), grad[1].detach().item(), q_a[0].detach().item(), q_a[1].detach().item(), g_cdf.detach().item()), file=f)# std, tanh_mu_T)
    t6 = datetime.now() 
    return ac_np, {}

def get_optimistic_exploration_action(ob_np, policy=None, qfs=None, qfs_target=None, hyper_params=None):

    assert ob_np.ndim == 1
    use_aleatoric = hyper_params['use_aleatoric']

    beta_UB = hyper_params['beta_UB']
    delta = hyper_params['delta']
    version = hyper_params['version']
    seed = hyper_params['seed']
    f = open(f"./logs/doac{version}_{seed}.log", 'a+')
    print(datetime.now() , file=f)

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    sampled_action, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    # Ensure that pretanh_mu_T is not batched
    assert len(list(pre_tanh_mu_T.shape)) == 1, pre_tanh_mu_T
    assert len(list(std.shape)) == 1

    pre_tanh_mu_T.requires_grad_()
    tanh_mu_T = torch.tanh(pre_tanh_mu_T)

    tau, tau_hat, presum_tau = get_tau(ob, sampled_action.unsqueeze(0), fp=None)
    tau_hat = tau_hat.squeeze(0)
    # print('ob, tanh_mu_T.cuda(), tau_hat.cuda()')
    # print(ob, tanh_mu_T.cuda(), tau_hat.cuda())
    # print(ob.shape, tanh_mu_T.cuda().shape, tau_hat.cuda().shape)
    # Get the upper bound of the Q estimate
    # args = list(torch.unsqueeze(i, dim=0) for i in (ob, tanh_mu_T))
    args = None
    if use_aleatoric:
        args = list(torch.unsqueeze(i, dim=0) for i in (ob, tanh_mu_T.cuda(), tau_hat.cuda()))
    else:
        args = list(torch.unsqueeze(i, dim=0) for i in (ob, tanh_mu_T))
    Q1 = qfs[0](*args)
    Q2 = qfs[1](*args)


    # 1. mean(Q1) - mean(Q2)
    # 2. mean(Q1-Q2)
    if use_aleatoric:
        mu_Q = (torch.mean(Q1) + torch.mean(Q2)) / 2.0
        # print("- mu_Q", mu_Q)
        sigma_Q = torch.abs(torch.mean(Q1) - torch.mean(Q2)) / 2.0
    else:
        mu_Q = (Q1 + Q2) / 2.0
        sigma_Q = torch.abs(Q1 - Q2) / 2.0

    Q_UB = mu_Q + beta_UB * sigma_Q
    if PRINTDETAILEDLOG:
        print("- 2", "{:.4f}".format(sigma_Q.item()), ob_np, file=f)
    else:
        pass
        # print("- 2", "{:.4f}".format(sigma_Q.item()), file=f)

    # Obtain the gradient of Q_UB wrt to a
    # with a evaluated at mu_t
    grad = torch.autograd.grad(Q_UB, pre_tanh_mu_T)
    grad = grad[0]

    assert grad is not None
    assert pre_tanh_mu_T.shape == grad.shape

    # Obtain Sigma_T (the covariance of the normal distribution)
    Sigma_T = torch.pow(std, 2)

    # The dividor is (g^T Sigma g) ** 0.5
    # Sigma is diagonal, so this works out to be
    # ( sum_{i=1}^k (g^(i))^2 (sigma^(i))^2 ) ** 0.5
    denom = torch.sqrt(
        torch.sum(
            torch.mul(torch.pow(grad, 2), Sigma_T)
        )
    ) + 10e-6

    # Obtain the change in mu
    mu_C = math.sqrt(2.0 * delta) * torch.mul(Sigma_T, grad) / denom

    assert mu_C.shape == pre_tanh_mu_T.shape

    mu_E = pre_tanh_mu_T + mu_C

    # Construct the tanh normal distribution and sample the exploratory action from it
    assert mu_E.shape == std.shape

    dist = TanhNormal(mu_E, std)

    ac = dist.sample()

    ac_np = ptu.get_numpy(ac)
    # if PRINTDETAILEDLOG:
        # print("- L362 ep", ac_np, "({:.4f}, {:.4f}), ({:.4f}, {:.4f})".format(mu_E[0].detach().item(), mu_E[1].detach().item(), grad[0].detach().item(), grad[1].detach().item()), file=f)# std, tanh_mu_T)
    # else:
        # print("- L362 ep", "({:.4f}, {:.4f}), ({:.4f}, {:.4f})".format(mu_E[0].detach().item(), mu_E[1].detach().item(), grad[0].detach().item(), grad[1].detach().item()), file=f)# std, tanh_mu_T)

    # mu_T_np = ptu.get_numpy(pre_tanh_mu_T)
    # mu_C_np = ptu.get_numpy(mu_C)
    # mu_E_np = ptu.get_numpy(mu_E)
    # dict(
    #     mu_T=mu_T_np,
    #     mu_C=mu_C_np,
    #     mu_E=mu_E_np
    # )

    # Return an empty dict, and do not log
    # stats for now
    return ac_np, {}
