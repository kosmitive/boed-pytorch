"""Variational nested Monte Carlo upper bound on the EIG (Foster et al., 2019).

The script ascends a design matrix for a Laplace-prior linear model.
Run from the repository root::

    python -m src.vnmc --steps 100
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
L = 100

# create parameters
xi = nn.Parameter(torch.zeros(n, p), requires_grad=True)
old_ig = None

T = 1.0

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


def _scale_tril(raw: torch.Tensor, dim: int, batch: int) -> torch.Tensor:
    """Lower-triangular factor; diagonal first, then the strict lower triangle."""
    factor = torch.zeros([batch, dim, dim])
    factor[:, range(dim), range(dim)] = F.softplus(raw[:, range(dim)]) + 1e-6
    cursor = dim
    for row in range(1, dim):
        width = row
        factor[:, row, :row] = raw[:, cursor:cursor + width]
        cursor += width
    return factor


def calc_eig_vnmc(xi):

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

    p_dist_th = dist.MultivariateNormal(mu, scale_tril=_scale_tril(s, p, N))
    th_posterior = p_dist_th.rsample([L])
    post_lp = p_dist_th.log_prob(th_posterior)
    prior_lp = dist_w.log_prob(th_posterior).sum(-1)

    # structure
    prior_mu_po = torch.einsum('kd,abd->abk', nxi, th_posterior)
    prior_o_po = dist_o.rsample([L, N])

    # obtain observations
    dist_y_po = dist.Normal(prior_mu_po, prior_o_po.unsqueeze(-1).repeat([1, 1, n]))
    ll_lp = dist_y_po.log_prob(y.T).sum(-1)

    log_dist = (prior_lp + ll_lp) / T - post_lp
    rv = torch.logsumexp(log_dist, dim=0) - np.log(L+1)
    lv = dist_y.log_prob(y).sum(0)
    eig = torch.mean(lv - rv)
    cur_eig = eig.detach().numpy()

    g1 = torch.autograd.grad(eig, xi, retain_graph=True)[0]
    g2 = torch.autograd.grad(eig, vpost_nn_params)

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
    """Ascend the design and the VNMC proposal for ``steps`` iterations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opt_xi = optim.Adam([xi])
    opt_nn = optim.Adam(vpost_nn_params)
    for it in range(steps):
        opt_xi.zero_grad()
        opt_nn.zero_grad()

        cur_ig, xi_grad, th_grad = calc_eig_vnmc(xi)
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
    _save_design_plot("vnmc_design.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    main(steps=parser.parse_args().steps)
