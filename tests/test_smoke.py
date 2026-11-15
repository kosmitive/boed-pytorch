"""Cheap checks for the closed form and for the original scripts."""

from __future__ import annotations

import math

import torch
from torch.distributions import MultivariateNormal

from showcase.estimators import (
    estimate_marginal,
    estimate_posterior,
    nested_monte_carlo,
    scale_tril,
    train_marginal,
    train_posterior,
)
from showcase.model import (
    analytic_eig,
    analytic_eig_matrix,
    features,
    gaussian_log_prob,
    logdet_covariance,
    standard_prior,
    update_posterior,
)
from src.params_to_flat import params_to_flat


def test_closed_form_matches_the_matrix_formula() -> None:
    cov = torch.eye(2)
    designs = torch.tensor([-1.5, 0.0, 0.7, 2.0])
    got = analytic_eig(designs, cov, sigma=1.0)
    rows = features(designs)
    for index in range(designs.shape[0]):
        expect = analytic_eig_matrix(rows[index : index + 1], cov, sigma=1.0)
        assert torch.allclose(got[index], expect, atol=1e-5)
    assert abs(float(got[1]) - 0.5 * math.log(2.0)) < 1e-6


def test_joint_design_matches_the_small_determinant() -> None:
    cov = torch.eye(2)
    matrix = features(torch.tensor([-2.0, 2.0]))
    by_observation = analytic_eig_matrix(matrix, cov, sigma=1.0)
    gram = cov @ matrix.T @ matrix
    _sign, logdet = torch.linalg.slogdet(torch.eye(2) + gram)
    assert torch.allclose(by_observation, 0.5 * logdet, atol=1e-5)


def test_information_grows_toward_the_boundary() -> None:
    cov = torch.eye(2)
    values = analytic_eig(torch.tensor([0.0, 0.5, 1.0, 2.0]), cov, sigma=1.0)
    assert torch.all(values[1:] > values[:-1])


def test_gaussian_log_prob_matches_pytorch() -> None:
    mean = torch.tensor([0.2, -0.4])
    cov = torch.tensor([[1.2, 0.3], [0.3, 0.8]])
    theta = torch.randn(4, 10, 2)
    got = gaussian_log_prob(theta, mean, cov)
    reference = MultivariateNormal(mean, cov + 1e-8 * torch.eye(2)).log_prob(theta)
    assert got.shape == (4, 10)
    assert torch.allclose(got, reference, atol=1e-4)


def test_scale_tril_places_the_off_diagonal() -> None:
    raw = torch.tensor([[0.0, 0.0, 0.4]], requires_grad=True)
    factor = scale_tril(raw, 2)
    diag = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-5
    expected = torch.tensor([[diag, 0.0], [0.4, diag]])
    assert torch.allclose(factor[0], expected, atol=1e-6)
    factor.square().sum().backward()
    assert raw.grad is not None
    assert float(raw.grad.abs().sum()) > 0.0


def test_posterior_update_matches_the_precision_form() -> None:
    mean = torch.zeros(2)
    cov = torch.eye(2)
    updated_mean, updated_cov = update_posterior(mean, cov, torch.tensor(1.0), torch.tensor(0.5), sigma=1.0)
    feat = features(torch.tensor(1.0))
    precision = torch.eye(2) + torch.outer(feat, feat)
    expect_cov = torch.linalg.inv(precision)
    expect_mean = expect_cov @ (feat * 0.5)
    assert torch.allclose(updated_cov, expect_cov, atol=1e-5)
    assert torch.allclose(updated_mean, expect_mean, atol=1e-5)


def test_boundary_designs_shrink_the_posterior_faster() -> None:
    extreme_cov = torch.eye(2)
    weak_cov = torch.eye(2)
    mean = torch.zeros(2)
    for _ in range(4):
        _mean, extreme_cov = update_posterior(mean, extreme_cov, torch.tensor(2.0), torch.tensor(0.0), sigma=1.0)
        _mean, weak_cov = update_posterior(mean, weak_cov, torch.tensor(0.0), torch.tensor(0.0), sigma=1.0)
    assert logdet_covariance(extreme_cov) < logdet_covariance(weak_cov)


def test_nmc_is_high_biased_for_one_inner_sample() -> None:
    torch.manual_seed(0)
    _mean, cov = standard_prior()
    truth = float(analytic_eig(torch.tensor(1.5), cov, sigma=1.0))
    estimate = float(nested_monte_carlo(torch.tensor([1.5]), _mean, cov, 1.0, n_outer=1500, n_inner=1)[0])
    assert estimate > truth + 0.2


def test_nmc_with_more_inner_samples_is_close() -> None:
    torch.manual_seed(0)
    mean, cov = standard_prior()
    truth = float(analytic_eig(torch.tensor(1.5), cov, sigma=1.0))
    estimate = float(nested_monte_carlo(torch.tensor([1.5]), mean, cov, 1.0, n_outer=2500, n_inner=64)[0])
    assert abs(estimate - truth) < 0.06


def test_short_training_moves_the_bounds_toward_the_truth() -> None:
    torch.manual_seed(0)
    mean, cov = standard_prior()
    truth = float(analytic_eig(torch.tensor(1.5), cov, sigma=1.0))
    _net, posterior_history = train_posterior(
        mean,
        cov,
        steps=40,
        batch_size=256,
        lr=5e-3,
        x_fixed=1.5,
        eval_every=40,
        eval_samples=2000,
    )
    _net, marginal_history = train_marginal(
        mean,
        cov,
        steps=40,
        batch_size=256,
        lr=5e-3,
        x_fixed=1.5,
        eval_every=40,
        eval_samples=2000,
    )
    posterior_start, posterior_end = posterior_history[0][1], posterior_history[-1][1]
    marginal_start, marginal_end = marginal_history[0][1], marginal_history[-1][1]
    assert posterior_end > posterior_start + 0.15
    assert posterior_end < truth + 0.08
    assert marginal_end < marginal_start - 0.1
    assert marginal_end > truth - 0.08


def test_params_to_flat_replaces_a_missing_gradient() -> None:
    weight = torch.nn.Parameter(torch.zeros(2, 3))
    flat = params_to_flat([None], [weight])
    assert flat.shape == (6,)
    assert torch.count_nonzero(flat) == 0


def test_implicit_likelihood_estimator_evaluates() -> None:
    import src.vimp as vimp

    ig, design_grad, likelihood_grads, marginal_grads, loss = vimp.eig(vimp.xi, vimp.ql, vimp.qm)
    assert torch.isfinite(ig)
    assert torch.isfinite(loss)
    assert design_grad.shape == vimp.xi.shape
    assert all(grad is not None for grad in likelihood_grads)
    assert all(grad is not None for grad in marginal_grads)


def test_original_estimators_evaluate() -> None:
    import src.vmarg as vmarg
    import src.vnmc as vnmc
    import src.vpost as vpost

    marginal_value, marginal_design_grad, marginal_net_grad = vmarg.calc_eig_vmarg(vmarg.xi)
    assert math.isfinite(float(marginal_value))
    assert marginal_design_grad.shape == vmarg.xi.shape
    assert float(marginal_net_grad[-2][vmarg.n :].abs().sum()) > 0.0

    posterior_value, posterior_design_grad, posterior_net_grad = vpost.calc_eig_vpost(vpost.xi)
    assert math.isfinite(float(posterior_value))
    assert posterior_design_grad.shape == vpost.xi.shape
    assert posterior_net_grad[-2].shape[0] == 3
    assert float(posterior_net_grad[-2][2].abs().sum()) > 0.0

    vnmc_value, vnmc_design_grad, _vnmc_net_grad = vnmc.calc_eig_vnmc(vnmc.xi)
    assert math.isfinite(float(vnmc_value))
    assert vnmc_design_grad.shape == vnmc.xi.shape

    # The recorded bounds are finite on a fresh evaluation as well.
    mean, cov = standard_prior()
    torch.manual_seed(1)
    posterior_net, _ = train_posterior(mean, cov, steps=1, batch_size=32, eval_every=0)
    marginal_net, _ = train_marginal(mean, cov, steps=1, batch_size=32, eval_every=0)
    point = torch.tensor([0.5])
    assert torch.isfinite(estimate_posterior(posterior_net, point, mean, cov, n_samples=32)).all()
    assert torch.isfinite(estimate_marginal(marginal_net, point, mean, cov, n_samples=32)).all()
