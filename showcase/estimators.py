"""Posterior, marginal, VNMC, and nested Monte Carlo estimators of EIG.

The objectives follow Foster et al., NeurIPS 2019. For the linear-Gaussian
model in :mod:`showcase.model` the posterior and the marginal are both
Gaussian, so a small network that emits a Gaussian can make each bound tight.

* Posterior lower bound: ``E[log q(θ | y, x) − log p(θ)]``.
* Marginal upper bound: ``E[log p(y | θ, x) − log q(y | x)]``.
* VNMC upper bound: the nested Monte Carlo contrast, with inner samples drawn
  from ``q(θ | y, x)`` instead of the prior.
* Nested Monte Carlo: the same contrast with inner samples drawn from the prior.
  It is biased high for a finite inner-sample count.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

from showcase.model import (
    DESIGN_HIGH,
    DESIGN_LOW,
    SIGMA,
    gaussian_log_prob,
    log_likelihood,
    sample_gaussian,
    sample_outcome,
)

_SOFTPLUS_OF_ONE = math.log(math.exp(1.0) - 1.0)


def scale_tril(raw: Tensor, dim: int) -> Tensor:
    """Build a batched Cholesky factor from an unconstrained vector.

    The first ``dim`` entries are diagonal logits, passed through softplus.
    The rest fill the strict lower triangle in row-major order.

    Args:
        raw: Factors of shape ``(..., dim * (dim + 1) // 2)``.
        dim: Matrix size.

    Returns:
        Lower-triangular factors of shape ``(..., dim, dim)``.
    """
    diag = F.softplus(raw[..., :dim]) + 1e-5
    factor = torch.diag_embed(diag)
    if dim == 1:
        return factor
    rows, cols = torch.tril_indices(row=dim, col=dim, offset=-1, device=raw.device)
    lower = raw[..., dim:]
    flat_index = rows * dim + cols
    leading = raw.shape[:-1]
    index = flat_index.view(*([1] * len(leading)), -1).expand(*leading, flat_index.numel())
    scattered = raw.new_zeros(*leading, dim * dim).scatter(dim=-1, index=index, src=lower)
    return factor + scattered.reshape(*leading, dim, dim)


class PosteriorNetwork(nn.Module):
    """Amortized Gaussian ``q(θ | y, x)``."""

    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 2)
        self.scale_head = nn.Linear(hidden, 3)
        self.initialize_to_gaussian(torch.zeros(2), torch.eye(2))

    def initialize_to_gaussian(self, mean: Tensor, cov: Tensor) -> None:
        """Set the readout so the network starts as ``Normal(mean, cov)``.

        Head weights are zero, so the initial Gaussian ignores ``(y, x)``.
        That is the prior, and the posterior lower bound then starts at zero
        when ``mean`` and ``cov`` are the prior.

        Args:
            mean: Initial mean, shape (2,).
            cov: Initial covariance, shape (2, 2).
        """
        eye = torch.eye(2, dtype=cov.dtype, device=cov.device)
        chol = torch.linalg.cholesky(cov.detach() + 1e-5 * eye)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.scale_head.weight)
        with torch.no_grad():
            self.mean_head.bias.copy_(mean.detach())
            self.scale_head.bias.zero_()
            self.scale_head.bias[0] = _softplus_inverse(chol[0, 0])
            self.scale_head.bias[1] = _softplus_inverse(chol[1, 1])
            self.scale_head.bias[2] = chol[1, 0]

    def forward(self, y: Tensor, x: Tensor) -> MultivariateNormal:
        """Return the variational posterior at each ``(y, x)`` pair.

        Args:
            y: Observations of shape (batch,).
            x: Designs of shape (batch,).

        Returns:
            A batch of full-covariance normals over θ.
        """
        hidden = self.backbone(torch.stack((y, x), dim=-1))
        return MultivariateNormal(self.mean_head(hidden), scale_tril=scale_tril(self.scale_head(hidden), 2))


class MarginalNetwork(nn.Module):
    """Amortized univariate Gaussian ``q(y | x)``."""

    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _SOFTPLUS_OF_ONE)

    def forward(self, x: Tensor) -> Normal:
        """Return the variational marginal at each design.

        Args:
            x: Designs of shape (batch,).

        Returns:
            A batch of univariate normals over ``y``.
        """
        hidden = self.backbone(x.unsqueeze(-1))
        mean = self.mean_head(hidden).squeeze(-1)
        std = F.softplus(self.scale_head(hidden).squeeze(-1)) + 1e-5
        return Normal(mean, std)


def estimate_posterior(
    net: PosteriorNetwork,
    x: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    n_samples: int = 4096,
) -> Tensor:
    """Monte Carlo estimate of the posterior lower bound at each design.

    Args:
        net: Trained variational posterior.
        x: Designs of shape (g,).
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        n_samples: Joint samples per design.

    Returns:
        Lower bound per design, shape (g,).
    """
    return _repeat_estimate(
        x,
        n_samples,
        lambda x_rep, theta, y: (
            net(y, x_rep).log_prob(theta) - gaussian_log_prob(theta, prior_mean, prior_cov)
        ),
        prior_mean,
        prior_cov,
        sigma,
    )


def estimate_marginal(
    net: MarginalNetwork,
    x: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    n_samples: int = 4096,
) -> Tensor:
    """Monte Carlo estimate of the marginal upper bound at each design.

    Args:
        net: Trained variational marginal.
        x: Designs of shape (g,).
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        n_samples: Joint samples per design.

    Returns:
        Upper bound per design, shape (g,).
    """
    return _repeat_estimate(
        x,
        n_samples,
        lambda x_rep, theta, y: log_likelihood(y, theta, x_rep, sigma) - net(x_rep).log_prob(y),
        prior_mean,
        prior_cov,
        sigma,
    )


def estimate_vnmc(
    net: PosteriorNetwork,
    x: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    n_outer: int = 2048,
    n_inner: int = 16,
) -> Tensor:
    """Monte Carlo estimate of the VNMC upper bound at each design.

    Args:
        net: Proposal ``q(θ | y, x)``.
        x: Designs of shape (g,).
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        n_outer: Outer samples per design.
        n_inner: Importance samples per outer sample.

    Returns:
        Upper bound per design, shape (g,).
    """
    with torch.no_grad():
        designs = x.reshape(-1)
        groups = designs.shape[0]
        x_rep = designs.repeat_interleave(n_outer)
        theta0 = sample_gaussian(prior_mean, prior_cov, groups * n_outer)
        y = sample_outcome(theta0, x_rep, sigma)
        bound = _vnmc_contrast(net, theta0, y, x_rep, prior_mean, prior_cov, sigma, n_inner)
        return bound.view(groups, n_outer).mean(dim=-1)


def nested_monte_carlo(
    x: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    n_outer: int = 2048,
    n_inner: int = 64,
) -> Tensor:
    """Nested Monte Carlo EIG at each design.

    Inner samples are drawn from the prior, which is VNMC with the proposal
    fixed at the prior. The estimate is high-biased for finite ``n_inner``.

    Args:
        x: Designs of shape (g,).
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        n_outer: Outer samples per design.
        n_inner: Prior samples used for each marginal likelihood.

    Returns:
        EIG estimate per design, shape (g,).
    """
    with torch.no_grad():
        designs = x.reshape(-1)
        groups = designs.shape[0]
        x_rep = designs.repeat_interleave(n_outer)
        theta0 = sample_gaussian(prior_mean, prior_cov, groups * n_outer)
        y = sample_outcome(theta0, x_rep, sigma)
        theta_inner = sample_gaussian(prior_mean, prior_cov, n_inner, groups * n_outer)
        log_inner = log_likelihood(y, theta_inner, x_rep, sigma)
        log_marginal = torch.logsumexp(log_inner, dim=0) - math.log(n_inner)
        log_outer = log_likelihood(y, theta0, x_rep, sigma)
        return (log_outer - log_marginal).view(groups, n_outer).mean(dim=-1)


def nmc_versus_inner(
    x_value: float,
    inner_counts: Sequence[int],
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    n_outer: int = 2000,
    n_reps: int = 5,
) -> Tensor:
    """Nested Monte Carlo on a grid of inner-sample sizes.

    Replicates share no randomness. Within one replicate, every inner-sample
    size is a prefix of the same inner sample, so the curve is paired.

    Args:
        x_value: Scalar design.
        inner_counts: Inner-sample sizes, each at least 1.
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        n_outer: Outer samples per replicate.
        n_reps: Independent replicates.

    Returns:
        Estimates of shape ``(n_reps, len(inner_counts))``.
    """
    rows = [
        _nmc_prefixes(x_value, inner_counts, prior_mean, prior_cov, sigma, n_outer)
        for _ in range(n_reps)
    ]
    return torch.stack(rows)


def vnmc_versus_inner(
    net: PosteriorNetwork,
    x_value: float,
    inner_counts: Sequence[int],
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    n_outer: int = 2000,
    n_reps: int = 5,
) -> Tensor:
    """VNMC on a grid of inner-sample sizes, paired within each replicate.

    Args:
        net: Proposal ``q(θ | y, x)``.
        x_value: Scalar design.
        inner_counts: Inner-sample sizes, each at least 1.
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        n_outer: Outer samples per replicate.
        n_reps: Independent replicates.

    Returns:
        Estimates of shape ``(n_reps, len(inner_counts))``.
    """
    rows = [
        _vnmc_prefixes(net, x_value, inner_counts, prior_mean, prior_cov, sigma, n_outer)
        for _ in range(n_reps)
    ]
    return torch.stack(rows)


def train_posterior(
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    steps: int = 400,
    batch_size: int = 512,
    lr: float = 4e-3,
    x_fixed: float | None = None,
    reference_x: float | None = None,
    eval_every: int = 0,
    eval_samples: int = 2048,
    init_at_prior: bool = False,
    verbose: bool = False,
    low: float = DESIGN_LOW,
    high: float = DESIGN_HIGH,
) -> tuple[PosteriorNetwork, list[tuple[int, float]]]:
    """Maximize the variational posterior lower bound.

    Args:
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        steps: Gradient steps.
        batch_size: Joint samples per step.
        lr: Adam learning rate.
        x_fixed: Train at one design. ``None`` draws designs uniformly.
        reference_x: Design at which the bound is recorded. Defaults to
            ``x_fixed``.
        eval_every: Record the bound every this many steps. Zero disables it.
        eval_samples: Samples used for each recorded bound.
        init_at_prior: Start ``q`` at the supplied prior.
        verbose: Print recorded bounds.
        low: Lower end of the design range used when ``x_fixed`` is omitted.
        high: Upper end of that design range.

    Returns:
        The trained network and a list of ``(step, bound)`` pairs. Step 0 is
        the bound before any update.
    """
    mean = prior_mean.detach()
    cov = prior_cov.detach()
    net = PosteriorNetwork()
    if init_at_prior:
        net.initialize_to_gaussian(mean, cov)
    anchor = reference_x if reference_x is not None else x_fixed

    def loss_fn(batch: int) -> Tensor:
        x = _sample_designs(batch, x_fixed, low, high)
        theta = sample_gaussian(mean, cov, batch)
        y = sample_outcome(theta, x, sigma)
        bound = net(y, x).log_prob(theta) - gaussian_log_prob(theta, mean, cov)
        return -bound.mean()

    def eval_fn() -> float:
        assert anchor is not None
        with torch.no_grad():
            value = estimate_posterior(net, torch.tensor([anchor]), mean, cov, sigma, eval_samples)
        return float(value[0])

    history = _fit(net, loss_fn, steps, batch_size, lr, eval_every, eval_fn if anchor is not None else None, verbose, "posterior")
    return net, history


def train_marginal(
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    steps: int = 400,
    batch_size: int = 512,
    lr: float = 4e-3,
    x_fixed: float | None = None,
    reference_x: float | None = None,
    eval_every: int = 0,
    eval_samples: int = 2048,
    verbose: bool = False,
    low: float = DESIGN_LOW,
    high: float = DESIGN_HIGH,
) -> tuple[MarginalNetwork, list[tuple[int, float]]]:
    """Minimize the variational marginal upper bound.

    Args:
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        steps: Gradient steps.
        batch_size: Joint samples per step.
        lr: Adam learning rate.
        x_fixed: Train at one design. ``None`` draws designs uniformly.
        reference_x: Design at which the bound is recorded.
        eval_every: Record the bound every this many steps. Zero disables it.
        eval_samples: Samples used for each recorded bound.
        verbose: Print recorded bounds.
        low: Lower end of the design range used when ``x_fixed`` is omitted.
        high: Upper end of that design range.

    Returns:
        The trained network and a list of ``(step, bound)`` pairs.
    """
    mean = prior_mean.detach()
    cov = prior_cov.detach()
    net = MarginalNetwork()
    anchor = reference_x if reference_x is not None else x_fixed

    def loss_fn(batch: int) -> Tensor:
        x = _sample_designs(batch, x_fixed, low, high)
        theta = sample_gaussian(mean, cov, batch)
        y = sample_outcome(theta, x, sigma)
        upper = log_likelihood(y, theta, x, sigma) - net(x).log_prob(y)
        return upper.mean()

    def eval_fn() -> float:
        assert anchor is not None
        with torch.no_grad():
            value = estimate_marginal(net, torch.tensor([anchor]), mean, cov, sigma, eval_samples)
        return float(value[0])

    history = _fit(net, loss_fn, steps, batch_size, lr, eval_every, eval_fn if anchor is not None else None, verbose, "marginal")
    return net, history


def train_vnmc(
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
    steps: int = 300,
    batch_size: int = 512,
    lr: float = 2e-3,
    n_inner: int = 8,
    x_fixed: float | None = None,
    reference_x: float | None = None,
    eval_every: int = 0,
    eval_samples: int = 1024,
    eval_inner: int = 16,
    init_at_prior: bool = True,
    verbose: bool = False,
    low: float = DESIGN_LOW,
    high: float = DESIGN_HIGH,
) -> tuple[PosteriorNetwork, list[tuple[int, float]]]:
    """Minimize the VNMC upper bound to train the importance proposal.

    Args:
        prior_mean: Prior mean.
        prior_cov: Prior covariance.
        sigma: Observation-noise standard deviation.
        steps: Gradient steps on the VNMC bound.
        batch_size: Outer samples per step.
        lr: Adam learning rate.
        n_inner: Inner samples used while training.
        x_fixed: Train at one design. ``None`` draws designs uniformly.
        reference_x: Design at which the bound is recorded.
        eval_every: Record the bound every this many steps. Zero disables it.
        eval_samples: Outer samples used for each recorded bound.
        eval_inner: Inner samples used for each recorded bound.
        init_at_prior: Start the proposal at the supplied prior.
        verbose: Print recorded bounds.
        low: Lower end of the design range used when ``x_fixed`` is omitted.
        high: Upper end of that design range.

    Returns:
        The trained proposal and a list of ``(step, bound)`` pairs.
    """
    mean = prior_mean.detach()
    cov = prior_cov.detach()
    net = PosteriorNetwork()
    if init_at_prior:
        net.initialize_to_gaussian(mean, cov)
    anchor = reference_x if reference_x is not None else x_fixed

    def loss_fn(batch: int) -> Tensor:
        x = _sample_designs(batch, x_fixed, low, high)
        theta = sample_gaussian(mean, cov, batch)
        y = sample_outcome(theta, x, sigma)
        return _vnmc_contrast(net, theta, y, x, mean, cov, sigma, n_inner).mean()

    def eval_fn() -> float:
        assert anchor is not None
        with torch.no_grad():
            value = estimate_vnmc(
                net,
                torch.tensor([anchor]),
                mean,
                cov,
                sigma,
                n_outer=eval_samples,
                n_inner=eval_inner,
            )
        return float(value[0])

    history = _fit(net, loss_fn, steps, batch_size, lr, eval_every, eval_fn if anchor is not None else None, verbose, "vnmc")
    return net, history


def _fit(
    net: nn.Module,
    loss_fn: Callable[[int], Tensor],
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    eval_fn: Callable[[], float] | None,
    verbose: bool,
    name: str,
) -> list[tuple[int, float]]:
    """Adam-minimize ``loss_fn`` and optionally record an evaluation."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    history: list[tuple[int, float]] = []
    if eval_fn is not None and eval_every:
        history.append((0, eval_fn()))
        if verbose:
            print(f"  {name} step 0: {history[-1][1]:.4f}", flush=True)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(batch_size)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()
        schedule.step()
        if eval_fn is not None and eval_every and step % eval_every == 0:
            history.append((step, eval_fn()))
            if verbose:
                print(f"  {name} step {step}: {history[-1][1]:.4f}", flush=True)
    return history


def _sample_designs(batch: int, x_fixed: float | None, low: float, high: float) -> Tensor:
    if x_fixed is None:
        return torch.empty(batch).uniform_(low, high)
    return torch.full((batch,), float(x_fixed))


def _repeat_estimate(
    x: Tensor,
    n_samples: int,
    score_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float,
) -> Tensor:
    with torch.no_grad():
        designs = x.reshape(-1)
        groups = designs.shape[0]
        x_rep = designs.repeat_interleave(n_samples)
        theta = sample_gaussian(prior_mean, prior_cov, groups * n_samples)
        y = sample_outcome(theta, x_rep, sigma)
        scores = score_fn(x_rep, theta, y)
        return scores.view(groups, n_samples).mean(dim=-1)


def _vnmc_contrast(
    net: PosteriorNetwork,
    theta0: Tensor,
    y: Tensor,
    x: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float,
    n_inner: int,
) -> Tensor:
    """Per-sample VNMC contrast ``log p(y|θ0) − log p̂(y)``.

    ``p̂(y)`` is the importance-sampling estimate of the marginal likelihood
    that uses ``n_inner`` draws from ``q(θ | y, x)``.
    """
    proposal = net(y, x)
    theta_inner = proposal.rsample((n_inner,))
    log_weight = (
        log_likelihood(y, theta_inner, x, sigma)
        + gaussian_log_prob(theta_inner, prior_mean, prior_cov)
        - proposal.log_prob(theta_inner)
    )
    log_marginal = torch.logsumexp(log_weight, dim=0) - math.log(n_inner)
    return log_likelihood(y, theta0, x, sigma) - log_marginal


def _nmc_prefixes(
    x_value: float,
    inner_counts: Sequence[int],
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float,
    n_outer: int,
) -> Tensor:
    with torch.no_grad():
        x = torch.full((n_outer,), float(x_value))
        theta0 = sample_gaussian(prior_mean, prior_cov, n_outer)
        y = sample_outcome(theta0, x, sigma)
        theta_inner = sample_gaussian(prior_mean, prior_cov, max(inner_counts), n_outer)
        log_inner = log_likelihood(y, theta_inner, x, sigma)
        log_outer = log_likelihood(y, theta0, x, sigma)
        estimates = []
        for count in inner_counts:
            log_marginal = torch.logsumexp(log_inner[:count], dim=0) - math.log(count)
            estimates.append((log_outer - log_marginal).mean())
        return torch.stack(estimates)


def _vnmc_prefixes(
    net: PosteriorNetwork,
    x_value: float,
    inner_counts: Sequence[int],
    prior_mean: Tensor,
    prior_cov: Tensor,
    sigma: float,
    n_outer: int,
) -> Tensor:
    with torch.no_grad():
        x = torch.full((n_outer,), float(x_value))
        theta0 = sample_gaussian(prior_mean, prior_cov, n_outer)
        y = sample_outcome(theta0, x, sigma)
        proposal = net(y, x)
        theta_inner = proposal.sample((max(inner_counts),))
        log_weight = (
            log_likelihood(y, theta_inner, x, sigma)
            + gaussian_log_prob(theta_inner, prior_mean, prior_cov)
            - proposal.log_prob(theta_inner)
        )
        log_outer = log_likelihood(y, theta0, x, sigma)
        estimates = []
        for count in inner_counts:
            log_marginal = torch.logsumexp(log_weight[:count], dim=0) - math.log(count)
            estimates.append((log_outer - log_marginal).mean())
        return torch.stack(estimates)


def _softplus_inverse(value: Tensor) -> Tensor:
    """Inverse of softplus, for initializing a Cholesky diagonal."""
    return value.expm1().clamp_min(1e-12).log()
