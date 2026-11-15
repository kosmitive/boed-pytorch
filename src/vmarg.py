"""Variational marginal upper bound on the EIG (Foster et al., 2019).

The script ascends a design matrix for a Laplace-prior linear model.
Run from the repository root::

    python -m src.vmarg --steps 100
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

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
        self.l_mu = nn.Linear(hidden_dim, out_dim)
        self.l_sigma = nn.Linear(hidden_dim, int(out_dim * (out_dim + 1) / 2))
        self.out_dim = out_dim
        self.N = batch_shape

    def forward(self, x):

        if len(x.shape) == 1:
            x = x.reshape([1, -1])
        a1 = torch.tanh(self.l1(x))
        a2 = torch.tanh(self.l2(a1))
        mu = self.l_mu(a2)
        sigma = self.l_sigma(a2)

        # Full Cholesky factor: diagonal first, then row-wise strict lower entries.
        factor = torch.zeros([self.N, self.out_dim, self.out_dim])
        factor[:, range(self.out_dim), range(self.out_dim)] = F.softplus(sigma[:, :self.out_dim]) + 1e-6
        cursor = self.out_dim
        for row in range(1, self.out_dim):
            width = row
            factor[:, row, :row] = sigma[:, cursor:cursor + width]
            cursor += width

        p_dist_y = dist.MultivariateNormal(mu, scale_tril=factor)
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

    # y is (n, N); q(y | design) is one n-dimensional Gaussian.
    post_lp = p_dist_y.log_prob(y.T)
    prior_lp = dist_y.log_prob(y).sum(0) / 0.5
    w = prior_lp - post_lp
    eig = torch.mean(w, dim=0)
    cur_eig = eig.detach().numpy()

    g1 = torch.autograd.grad(eig, xi, retain_graph=True)[0]
    g2 = torch.autograd.grad(eig, qmarg.parameters())

    return cur_eig, g1, g2


def _save_design_plot(name: str) -> None:
    """Write the current figure under ``artifacts/``."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    path = out / name
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Wrote {path}")


def main(steps: int = 100) -> None:
    """Ascend the design and the marginal network for ``steps`` iterations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opt_xi = optim.Adam([xi])
    opt_nn = optim.Adam(qmarg.parameters())
    for it in range(steps):
        opt_xi.zero_grad()
        opt_nn.zero_grad()

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
    _save_design_plot("vmarg_design.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    main(steps=parser.parse_args().steps)
