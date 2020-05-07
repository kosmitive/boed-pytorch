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


class MarginalNetwork(nn.Module):

    def __init__(self, inp_dim, out_dim, hidden_dim=64, batch_shape=1):

        nn.Module.__init__(self)
        self.l1 = nn.Linear(inp_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l_mu = nn.Linear(64, out_dim)
        self.l_sigma = nn.Linear(64, int(out_dim * (out_dim + 1) / 2))
        self.out_dim = out_dim
        self.N = batch_shape

    def forward(self, x):

        if len(x.shape) == 1: x = x.reshape([1, -1])
        a1 = torch.tanh(self.l1(x))
        a2 = torch.tanh(self.l2(a1))
        mu = self.l_mu(a2)
        sigma = self.l_sigma(a2)

        # build dist
        Sigma = torch.zeros([self.N, self.out_dim, self.out_dim])
        Sigma[:, range(self.out_dim), range(self.out_dim)] = F.softplus(sigma[:, :self.out_dim]) + 1e-6
        iv = self.out_dim
        for ip in range(1, self.out_dim):
            l = ip - 1
            Sigma[:, ip, :ip-1] = sigma[:, iv:iv + l]
            iv += l

        p_dist_y = dist.MultivariateNormal(mu, scale_tril=Sigma)
        return p_dist_y

qmarg = MarginalNetwork(n * p, n, batch_shape=1)

def calc_eig_vmarg(xi):

    # structure
    nxi = torch.tanh(xi)
    w = dist_w.rsample([p, N])
    o = dist_o.rsample([N])
    prior_mu = nxi @ w

    # obtain observations
    dist_y = dist.Normal(prior_mu, o)
    y = dist_y.rsample()

    # build network
    reshaped_xi = nxi.reshape([-1])
    p_dist_y = qmarg(reshaped_xi)

    post_lp = p_dist_y.log_prob(y.unsqueeze(-1)).sum(0)
    prior_lp = dist_y.log_prob(y).sum(0) / 0.5
    w = prior_lp - post_lp
    eig = torch.mean(w, dim=0)
    cur_eig = eig.detach().numpy()

    g1 = torch.autograd.grad(eig, xi, retain_graph=True)[0]
    g2 = torch.autograd.grad(eig, qmarg.parameters())

    return cur_eig, g1, g2

opt_xi = optim.Adam([xi])
opt_nn = optim.Adam(qmarg.parameters())
best_ig = None
T = 1.0

for it in range(100):

    opt_xi.zero_grad()
    opt_nn.zero_grad()

    fr = True

    cur_ig, xi_grad, th_grad = calc_eig_vmarg(xi)
    print(cur_ig)

    xi.grad = -xi_grad
    for pa, v in zip(qmarg.parameters(), th_grad):
        pa.grad = -v.detach()

    opt_xi.step()
    opt_nn.step()

des = torch.tanh(xi).detach().numpy()
plt.figure()
plt.scatter(des[:, 0], des[:, 1])
plt.xlim([np.min(des[:, 0]) - 0.005, np.max(des[:, 0]) + 0.005])
plt.ylim([np.min(des[:, 1]) - 0.005, np.max(des[:, 1]) + 0.005])
plt.show()
