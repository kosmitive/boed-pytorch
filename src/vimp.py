import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg

from src.params_to_flat import params_to_flat

p = 2
n = 30
od = 1
dist_w = dist.Laplace(0, 0.1)

# number of samples
N = 1500

# create parameters
xi = nn.Parameter(torch.zeros(n, p), requires_grad=True)
old_ig = None

class MarginalNetwork(nn.Module):

    def __init__(self, inp_dim, out_dim, hidden_dim=16, batch_shape=1):

        nn.Module.__init__(self)
        self.l1 = nn.Linear(inp_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l_mu = nn.Linear(hidden_dim, out_dim)
        self.l_sigma = nn.Linear(hidden_dim, int(out_dim * (out_dim + 1) / 2))
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

# define the marginal networks
ql = MarginalNetwork(n * p + p, n, batch_shape=N)
qm = MarginalNetwork(n * p, n)

def eig(xi, ql, qm):

    # structure
    nxi = torch.tanh(xi)
    w = dist_w.rsample([p, N])
    o = 0.01
    prior_mu = nxi @ w

    # obtain observations
    dist_y = dist.Normal(prior_mu, o)
    y = dist_y.rsample()

    inp = nxi.reshape([1, -1])
    qmd = qm(inp)
    qld = ql(torch.cat([inp.repeat([N, 1]), w.T], dim=1))
    lqmd = qmd.log_prob(y.T)
    lqld = qld.log_prob(y.T)
    ig = torch.mean(lqld - lqmd, dim=0)
    loss = torch.mean(lqmd + lqld, dim=0)

    g1 = torch.autograd.grad(ig, xi, retain_graph=True, create_graph=True)[0]
    g2_o = torch.autograd.grad(loss, ql.parameters(), retain_graph=True, create_graph=True)
    g3_o = torch.autograd.grad(loss, qm.parameters(), retain_graph=True, create_graph=True)

    use_implicit_gradients = False
    if use_implicit_gradients:

        g2 = params_to_flat(g2_o, ql.parameters())
        g3 = params_to_flat(g3_o, qm.parameters())

        def hvp_l(v):
            v1 = torch.autograd.grad(torch.sum(torch.tensor(v) * g2), ql.parameters(), retain_graph=True)
            return params_to_flat(v1, ql.parameters()).detach().numpy()

        def hvp_m(v):
            v1 = torch.autograd.grad(torch.sum(torch.tensor(v) * g3), qm.parameters(), retain_graph=True)
            return params_to_flat(v1, ql.parameters()).detach().numpy()

        def jvp(v, g):
            v1 = torch.autograd.grad(torch.sum(v.detach() * g), xi, retain_graph=True)[0]
            return v1

        num_l = sum([int(np.prod(para.shape)) for para in ql.parameters()])
        num_m = sum([int(np.prod(para.shape)) for para in qm.parameters()])
        obj = LinearOperator(matvec=hvp_l, dtype=np.float32, shape=[num_l, num_l])
        x = g2.detach().numpy()
        cg_iter = 70
        imp_l, _ = cg(obj, x, maxiter=cg_iter)

        obj = LinearOperator(matvec=hvp_m, dtype=np.float32, shape=[num_m, num_m])
        x = g3.detach().numpy()
        imp_m, _ = cg(obj, x, maxiter=cg_iter)

        imp_l = torch.tensor(imp_l)
        imp_m = torch.tensor(imp_m)
        g1 = g1 + jvp(imp_l, g2) + jvp(imp_m, g3)

    return ig, g1, g2_o, g3_o, loss


opt_xi = optim.SGD([xi], lr=1e-2)
opt_qm = optim.SGD(qm.parameters(), lr=1e-5)
opt_ql = optim.SGD(ql.parameters(), lr=1e-5)
best_ig = None
T = 1.0

for out in range(100):
    for it in range(10):

        opt_xi.zero_grad()
        opt_qm.zero_grad()
        opt_ql.zero_grad()

        fr = True

        cur_ig, xi_grad, th_grad, th_grad_m, loss = eig(xi, ql, qm)
        print("ig=" + str(cur_ig) + ", loss=" + str(loss))


        xi.grad = -xi_grad
        for pa, v in zip(ql.parameters(), th_grad):
            pa.grad = -v.detach()

        for pa, v in zip(qm.parameters(), th_grad_m):
            pa.grad = -v.detach()

        opt_xi.step()
        #opt_qm.step()
        #opt_ql.step()

    for it in range(10):

        opt_xi.zero_grad()
        opt_qm.zero_grad()
        opt_ql.zero_grad()

        fr = True

        cur_ig, xi_grad, th_grad, th_grad_m, loss = eig(xi, ql, qm)
        print("ig=" + str(cur_ig) + ", loss=" + str(loss))

        xi.grad = -xi_grad
        for pa, v in zip(ql.parameters(), th_grad):
            pa.grad = -v.detach()

        for pa, v in zip(qm.parameters(), th_grad_m):
            pa.grad = -v.detach()

        #opt_xi.step()
        opt_qm.step()
        opt_ql.step()

des = torch.tanh(xi).detach().numpy()
plt.figure()
plt.scatter(des[:, 0], des[:, 1])
x_lim = [np.min(des[:, 0]) - 0.005, np.max(des[:, 0]) + 0.005]
y_lim = [np.min(des[:, 1]) - 0.005, np.max(des[:, 1]) + 0.005]
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.show()
