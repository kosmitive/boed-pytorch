import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import matplotlib.pyplot as plt

from src.params_to_flat import params_to_flat

p = 2
n = 10
od = 1
dist_w = dist.Laplace(0, 0.1)
dist_o = dist.Exponential(1)

# number of samples
N = 10 ** 3

# create parameters
xi = nn.Parameter(torch.zeros(n, p), requires_grad=True)
old_ig = None

# network
vmarg_l_1 = nn.Linear(p * n, 64)
vmarg_l_l2 = nn.Linear(64, 64)
vmarg_l_mu = nn.Linear(64, od)
vmarg_l_s = nn.Linear(64, int(od * (od + 1) / 2))
vmarg_nn_params = list(vmarg_l_1.parameters()) + list(vmarg_l_l2.parameters()) + list(vmarg_l_mu.parameters()) + list(vmarg_l_s.parameters())

# network
vpost_l_1 = nn.Linear((p + 1) * n, 64)
vpost_l_l2 = nn.Linear(64, 64)
vpost_l_mu = nn.Linear(64, p)
vpost_l_s = nn.Linear(64, int(p * (p + 1) / 2))
vpost_nn_params = list(vpost_l_1.parameters()) + list(vpost_l_l2.parameters()) + list(vpost_l_mu.parameters()) + list(vpost_l_s.parameters())


def calc_eig_vpost(xi):

    # structure
    nxi = torch.tanh(xi)
    w = dist_w.rsample([p, N])
    o = dist_o.rsample([N])
    prior_mu = nxi @ w

    # obtain observations
    dist_y = dist.Normal(prior_mu, o)
    y = dist_y.rsample()

    # build network
    reshaped_xi = nxi.reshape([1, -1]).repeat([N, 1])
    reshaped_y = y.T
    comb = torch.cat([reshaped_xi, reshaped_y], dim=1)
    ac1 = torch.tanh(vpost_l_1(comb))
    ac2 = torch.tanh(vpost_l_l2(ac1))
    mu = vpost_l_mu(ac2)
    s = vpost_l_s(ac2)

    # build dist
    SigmaR = torch.zeros([N, p, p])
    SigmaR[:, range(p), range(p)] = F.softplus(s[:, range(p)]) + 1e-6
    iv = od
    for ip in range(1, p):
        l = p - 1
        SigmaR[:, ip, :ip - 1] = s[:, iv:iv + l]
        iv += l

    p_dist_th = dist.MultivariateNormal(mu, scale_tril=SigmaR)
    post_lp = p_dist_th.log_prob(w.T)
    prior_lp = dist_w.log_prob(w.T).sum(1)
    w = prior_lp - post_lp
    eig = torch.mean(w, dim=0)
    cur_eig = eig.detach().numpy()

    g1 = torch.autograd.grad(eig, xi, retain_graph=True)[0]
    g2 = torch.autograd.grad(eig, vpost_nn_params)

    return cur_eig, g1, g2


opt_xi = optim.Adam([xi])
opt_nn = optim.Adam(vpost_nn_params)
vpost_opt_nn = optim.Adam(vpost_nn_params)
best_ig = None
T = 1.0

for it in range(100):

    opt_xi.zero_grad()
    opt_nn.zero_grad()

    fr = True

    cur_ig, xi_grad, th_grad = calc_eig_vpost(xi)
    print(cur_ig)

    xi.grad = -xi_grad
    for pa, v in zip(vpost_nn_params, th_grad):
        pa.grad = -v.detach()

    opt_xi.step()
    opt_nn.step()

des = torch.tanh(xi).detach().numpy()
plt.figure()
plt.scatter(des[:, 0], des[:, 1])
plt.xlim([np.min(des[:, 0]) - 0.005, np.max(des[:, 0]) + 0.005])
plt.ylim([np.min(des[:, 1]) - 0.005, np.max(des[:, 1]) + 0.005])
plt.show()
