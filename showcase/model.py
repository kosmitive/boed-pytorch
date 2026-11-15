"""Conjugate Bayesian linear regression with a closed-form EIG.

The latent parameter is two-dimensional and one experiment is a scalar design
``x``. The observation model is

    θ ~ Normal(μ, Σ),
    y | θ, x ~ Normal(θ₀ + θ₁ x, σ²).

A design matrix whose rows are ``[1, x]`` then has

    EIG(X) = 0.5 * logdet(I + X Σ Xᵀ / σ²),

which is the mutual information between θ and the Gaussian observation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

DESIGN_LOW = -2.0
DESIGN_HIGH = 2.0
SIGMA = 1.0
THETA_DIM = 2


def standard_prior() -> tuple[Tensor, Tensor]:
    """Return the standard normal prior mean and covariance.

    Returns:
        Mean of shape (2,) and covariance of shape (2, 2).
    """
    return torch.zeros(THETA_DIM), torch.eye(THETA_DIM)


def features(x: Tensor) -> Tensor:
    """Map a scalar design to the linear-regression feature ``[1, x]``.

    Args:
        x: Design, of any shape.

    Returns:
        Features with a trailing dimension of size 2.
    """
    return torch.stack((torch.ones_like(x), x), dim=-1)


def analytic_eig(x: Tensor, prior_cov: Tensor, sigma: float = SIGMA) -> Tensor:
    """Closed-form EIG of one observation at each design.

    Designs are scored separately. The EIG of several observations taken
    together is :func:`analytic_eig_matrix`.

    Args:
        x: Scalar designs of any shape.
        prior_cov: Prior covariance of θ, shape (2, 2).
        sigma: Observation-noise standard deviation.

    Returns:
        EIG in nats, with the same shape as ``x``.
    """
    feat = features(x)
    quad = torch.einsum("...i,ij,...j->...", feat, prior_cov, feat)
    return 0.5 * torch.log1p(quad / sigma**2)


def analytic_eig_matrix(
    feature_matrix: Tensor,
    prior_cov: Tensor,
    sigma: float = SIGMA,
) -> Tensor:
    """Closed-form EIG of a joint linear-Gaussian design.

    Args:
        feature_matrix: Matrix ``X`` of shape (n, d), one row per observation.
        prior_cov: Prior covariance Σ of shape (d, d).
        sigma: Observation-noise standard deviation.

    Returns:
        Scalar EIG, ``0.5 * logdet(I + X Σ Xᵀ / σ²)``.
    """
    scaled = feature_matrix @ prior_cov @ feature_matrix.transpose(-1, -2)
    scaled = scaled / sigma**2
    eye = torch.eye(feature_matrix.shape[-2], dtype=feature_matrix.dtype, device=feature_matrix.device)
    _sign, logdet = torch.linalg.slogdet(eye + scaled)
    return 0.5 * logdet


def sample_gaussian(mean: Tensor, cov: Tensor, *batch: int) -> Tensor:
    """Draw samples from a full-covariance normal.

    Args:
        mean: Mean of shape (d,).
        cov: Covariance of shape (d, d).
        batch: Leading sample dimensions.

    Returns:
        Samples of shape ``(*batch, d)``.
    """
    dim = mean.shape[0]
    eye = torch.eye(dim, dtype=mean.dtype, device=mean.device)
    chol = torch.linalg.cholesky(cov + 1e-8 * eye)
    epsilon = torch.randn(*batch, dim, dtype=mean.dtype, device=mean.device)
    return mean + epsilon @ chol.transpose(-1, -2)


def sample_outcome(theta: Tensor, x: Tensor, sigma: float = SIGMA) -> Tensor:
    """Sample ``y | θ, x`` from the linear observation model.

    Args:
        theta: Parameters with trailing dimension 2.
        x: Designs broadcastable against ``theta`` without the last axis.
        sigma: Observation-noise standard deviation.

    Returns:
        Observations with the same shape as the mean ``[1, x] · θ``.
    """
    feat = features(x)
    while feat.ndim < theta.ndim:
        feat = feat.unsqueeze(0)
    mean = (feat * theta).sum(dim=-1)
    return mean + sigma * torch.randn_like(mean)


def log_likelihood(y: Tensor, theta: Tensor, x: Tensor, sigma: float = SIGMA) -> Tensor:
    """Gaussian log-likelihood of an observation.

    Args:
        y: Observations.
        theta: Parameters with trailing dimension 2.
        x: Designs broadcastable to ``theta``.
        sigma: Observation-noise standard deviation.

    Returns:
        Log-likelihood with the broadcast shape of ``y`` and the mean.
    """
    feat = features(x)
    while feat.ndim < theta.ndim:
        feat = feat.unsqueeze(0)
    mean = (feat * theta).sum(dim=-1)
    variance = sigma**2
    return -0.5 * ((y - mean) ** 2 / variance + math.log(2.0 * math.pi * variance))


def gaussian_log_prob(value: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
    """Log-density of a normal with a full covariance.

    Args:
        value: Points with trailing dimension ``d``.
        mean: Mean of shape (d,).
        cov: Covariance of shape (d, d).

    Returns:
        Log-density with the leading shape of ``value``.
    """
    dim = mean.shape[-1]
    eye = torch.eye(dim, dtype=cov.dtype, device=cov.device)
    scale = cov + 1e-8 * eye
    diff = value - mean
    solved = torch.linalg.solve(scale, diff.unsqueeze(-1)).squeeze(-1)
    quad = (diff * solved).sum(dim=-1)
    _sign, logdet = torch.linalg.slogdet(scale)
    return -0.5 * (quad + logdet + dim * math.log(2.0 * math.pi))


def update_posterior(
    mean: Tensor,
    cov: Tensor,
    x: Tensor,
    y: Tensor,
    sigma: float = SIGMA,
) -> tuple[Tensor, Tensor]:
    """Apply one conjugate Gaussian observation update.

    Args:
        mean: Current posterior mean, shape (2,).
        cov: Current posterior covariance, shape (2, 2).
        x: Scalar design.
        y: Scalar observation.
        sigma: Observation-noise standard deviation.

    Returns:
        Updated mean and covariance.
    """
    feat = features(x.reshape(()))
    precision = torch.linalg.inv(cov)
    precision = 0.5 * (precision + precision.transpose(-1, -2))
    new_precision = precision + torch.outer(feat, feat) / sigma**2
    new_precision = 0.5 * (new_precision + new_precision.transpose(-1, -2))
    cov_new = torch.linalg.inv(new_precision)
    cov_new = 0.5 * (cov_new + cov_new.transpose(-1, -2))
    mean_new = cov_new @ (precision @ mean + feat * (y.reshape(()) / sigma**2))
    return mean_new, cov_new


def logdet_covariance(cov: Tensor) -> float:
    """Return ``log det`` of a covariance, the posterior volume up to a shift.

    Args:
        cov: Covariance matrix.

    Returns:
        Log-determinant as a Python float.
    """
    _sign, logdet = torch.linalg.slogdet(cov)
    return float(logdet)
