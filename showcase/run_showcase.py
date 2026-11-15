"""Run the linear-regression EIG showcase and write the figures.

From the repository root::

    python -m showcase.run_showcase

The command is meant to finish on CPU in about a minute. It writes

* ``assets/eig_vs_design.png``
* ``assets/estimator_convergence.png``
* ``assets/sequential_design.png``
* ``assets/results.json``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Ellipse
from torch import Tensor

from showcase.estimators import (
    ImplicitLikelihood,
    MarginalNetwork,
    PosteriorNetwork,
    estimate_implicit,
    estimate_marginal,
    estimate_posterior,
    estimate_vnmc,
    nested_monte_carlo,
    nmc_versus_inner,
    train_implicit,
    train_marginal,
    train_posterior,
    train_vnmc,
    vnmc_versus_inner,
)
from showcase.model import (
    DESIGN_HIGH,
    DESIGN_LOW,
    SIGMA,
    analytic_eig,
    features,
    logdet_covariance,
    standard_prior,
    update_posterior,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

SEED = 0
REFERENCE_X = 1.5
TRAIN_STEPS = 450
VNMC_STEPS = 350
BATCH_SIZE = 512
LEARNING_RATE = 4e-3
EVAL_EVERY = 25
EVAL_SAMPLES = 12000
HISTORY_SAMPLES = 8192
TRAIN_LOW = -2.4
TRAIN_HIGH = 2.4
GRID_POINTS = 25
NMC_OUTER = 4000
NMC_INNER = 96
VNMC_EVAL_OUTER = 3072
VNMC_EVAL_INNER = 24
INNER_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
INNER_REPS = 6
INNER_OUTER = 4000
SEQUENTIAL_EXPERIMENTS = 8
SEQUENTIAL_CANDIDATES = 21
SEQUENTIAL_TRAIN_STEPS = 220
SEQUENTIAL_EVAL_SAMPLES = 4096
SEQUENTIAL_RANDOM_RUNS = 40
TRUE_THETA = (0.3, -0.8)
CHI2_95 = 5.991464547107979

COLOR_ANALYTIC = "#222222"
COLOR_POSTERIOR = "#0072B2"
COLOR_MARGINAL = "#E69F00"
COLOR_VNMC = "#009E73"
COLOR_NMC = "#D55E00"
COLOR_IMPLICIT = "#56B4E9"
COLOR_RANDOM = "#0072B2"
COLOR_VARIATIONAL = "#CC79A7"


def main(argv: list[str] | None = None) -> None:
    """Generate the showcase figures selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        choices=("all", "eig", "convergence", "sequential"),
        default="all",
        help="Which figure to write. The default writes all three.",
    )
    args = parser.parse_args(argv)
    _apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    match args.figures:
        case "all":
            selected = ("eig", "convergence", "sequential")
        case "eig" | "convergence" | "sequential":
            selected = (args.figures,)
        case _:
            raise ValueError(f"Unknown figure set: {args.figures}")

    results: dict[str, object] = {"seed": SEED, "sigma": SIGMA, "reference_design": REFERENCE_X}
    # EIG and convergence share one training run. Sequential is separate.
    trained: dict[str, object] | None = None
    if "eig" in selected or "convergence" in selected:
        print("Training amortized estimators...", flush=True)
        trained = _train_amortized()
    for name in selected:
        match name:
            case "eig":
                assert trained is not None
                results["eig"] = _write_eig_figure(trained)
            case "convergence":
                assert trained is not None
                results["convergence"] = _write_convergence_figure(trained)
            case "sequential":
                print("Running sequential design...", flush=True)
                results["sequential"] = _write_sequential_figure()
            case _:
                raise ValueError(name)
    elapsed = time.perf_counter() - started
    results["seconds"] = round(elapsed, 1)
    if args.figures == "all":
        (ASSETS / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Finished in {elapsed:.1f}s", flush=True)
    _print_summary(results)


def _train_amortized() -> dict[str, object]:
    """Fit the variational estimators under the standard prior."""
    _seed(SEED)
    mean, cov = standard_prior()
    common = {
        "sigma": SIGMA,
        "batch_size": BATCH_SIZE,
        "reference_x": REFERENCE_X,
        "eval_every": EVAL_EVERY,
        "eval_samples": HISTORY_SAMPLES,
        "verbose": True,
        "low": TRAIN_LOW,
        "high": TRAIN_HIGH,
    }
    posterior, posterior_history = train_posterior(
        mean, cov, steps=TRAIN_STEPS, lr=LEARNING_RATE, **common
    )
    marginal, marginal_history = train_marginal(
        mean, cov, steps=TRAIN_STEPS, lr=LEARNING_RATE, **common
    )
    vnmc, vnmc_history = train_vnmc(
        mean,
        cov,
        steps=VNMC_STEPS,
        lr=2e-3,
        n_inner=8,
        eval_inner=16,
        **common,
    )
    implicit, implicit_history = train_implicit(mean, cov, steps=TRAIN_STEPS, lr=LEARNING_RATE, **common)
    return {
        "mean": mean,
        "cov": cov,
        "posterior": posterior,
        "marginal": marginal,
        "vnmc": vnmc,
        "implicit": implicit,
        "posterior_history": posterior_history,
        "marginal_history": marginal_history,
        "vnmc_history": vnmc_history,
        "implicit_history": implicit_history,
    }


def _write_eig_figure(trained: dict[str, object]) -> dict[str, object]:
    """Compare every estimator with the closed form across the design range."""
    mean = trained["mean"]
    cov = trained["cov"]
    assert isinstance(mean, Tensor) and isinstance(cov, Tensor)
    posterior = trained["posterior"]
    marginal = trained["marginal"]
    vnmc = trained["vnmc"]
    implicit = trained["implicit"]
    assert isinstance(posterior, PosteriorNetwork)
    assert isinstance(marginal, MarginalNetwork)
    assert isinstance(vnmc, PosteriorNetwork)
    assert isinstance(implicit, ImplicitLikelihood)

    grid = torch.linspace(DESIGN_LOW, DESIGN_HIGH, GRID_POINTS)
    truth_grid = torch.linspace(DESIGN_LOW, DESIGN_HIGH, 201)
    truth = analytic_eig(grid, cov, SIGMA)
    truth_line = analytic_eig(truth_grid, cov, SIGMA)
    print("Evaluating estimators on the design grid...", flush=True)
    posterior_est = estimate_posterior(posterior, grid, mean, cov, SIGMA, EVAL_SAMPLES)
    marginal_est = estimate_marginal(marginal, grid, mean, cov, SIGMA, EVAL_SAMPLES)
    vnmc_est = estimate_vnmc(vnmc, grid, mean, cov, SIGMA, VNMC_EVAL_OUTER, VNMC_EVAL_INNER)
    implicit_est = estimate_implicit(implicit, grid, mean, cov, SIGMA, EVAL_SAMPLES)
    nmc_est = nested_monte_carlo(grid, mean, cov, SIGMA, NMC_OUTER, NMC_INNER)

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8.4, 6.4),
        gridspec_kw={"height_ratios": [2.15, 1.0]},
        constrained_layout=True,
    )
    ax.plot(truth_grid.numpy(), truth_line.numpy(), color=COLOR_ANALYTIC, label="Closed form", zorder=2)
    _mark(ax, grid, posterior_est, COLOR_POSTERIOR, "o", "Posterior lower bound")
    _mark(ax, grid, marginal_est, COLOR_MARGINAL, "s", "Marginal upper bound")
    _mark(ax, grid, vnmc_est, COLOR_VNMC, "^", "VNMC")
    _mark(ax, grid, implicit_est, COLOR_IMPLICIT, "D", "Implicit likelihood")
    _mark(ax, grid, nmc_est, COLOR_NMC, "x", f"NMC (M = {NMC_INNER})")
    ax.set_ylabel("EIG (nats)")
    ax.set_title(r"One measurement,  $y = \theta_0 + \theta_1 x + \varepsilon$")
    ax.legend(loc="upper center", ncol=3)

    errors = {
        "posterior": posterior_est - truth,
        "marginal": marginal_est - truth,
        "vnmc": vnmc_est - truth,
        "implicit": implicit_est - truth,
        "nmc": nmc_est - truth,
    }
    _mark(ax_err, grid, errors["posterior"], COLOR_POSTERIOR, "o", "Posterior")
    _mark(ax_err, grid, errors["marginal"], COLOR_MARGINAL, "s", "Marginal")
    _mark(ax_err, grid, errors["vnmc"], COLOR_VNMC, "^", "VNMC")
    _mark(ax_err, grid, errors["implicit"], COLOR_IMPLICIT, "D", "Implicit likelihood")
    _mark(ax_err, grid, errors["nmc"], COLOR_NMC, "x", "NMC")
    ax_err.axhline(0.0, color=COLOR_ANALYTIC, linewidth=1.0)
    ax_err.set_xlabel("design  x")
    ax_err.set_ylabel("estimate − truth")
    ax_err.set_xlim(DESIGN_LOW, DESIGN_HIGH)

    path = ASSETS / "eig_vs_design.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)

    summary = {
        name: {
            "max_abs_error": round(float(err.abs().max()), 4),
            "mean_error": round(float(err.mean()), 4),
        }
        for name, err in errors.items()
    }
    reference = float(analytic_eig(torch.tensor(REFERENCE_X), cov, SIGMA))
    summary["reference_x"] = REFERENCE_X
    summary["reference_eig"] = round(reference, 4)
    return summary


def _write_convergence_figure(trained: dict[str, object]) -> dict[str, object]:
    """Show bound convergence in training steps and inner-sample bias."""
    mean = trained["mean"]
    cov = trained["cov"]
    assert isinstance(mean, Tensor) and isinstance(cov, Tensor)
    vnmc = trained["vnmc"]
    assert isinstance(vnmc, PosteriorNetwork)
    truth = float(analytic_eig(torch.tensor(REFERENCE_X), cov, SIGMA))
    posterior_history = trained["posterior_history"]
    marginal_history = trained["marginal_history"]
    implicit_history = trained["implicit_history"]
    assert isinstance(posterior_history, list) and isinstance(marginal_history, list)
    assert isinstance(implicit_history, list)

    print("Estimating inner-sample bias...", flush=True)
    _seed(SEED + 1)
    nmc_reps = nmc_versus_inner(REFERENCE_X, INNER_COUNTS, mean, cov, SIGMA, INNER_OUTER, INNER_REPS)
    vnmc_reps = vnmc_versus_inner(vnmc, REFERENCE_X, INNER_COUNTS, mean, cov, SIGMA, INNER_OUTER, INNER_REPS)

    fig, (ax_train, ax_inner) = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    post_steps, post_values = _history_arrays(posterior_history)
    marg_steps, marg_values = _history_arrays(marginal_history)
    implicit_steps, implicit_values = _history_arrays(implicit_history)
    ax_train.plot(post_steps, post_values, color=COLOR_POSTERIOR, label="Posterior lower bound")
    ax_train.plot(marg_steps, marg_values, color=COLOR_MARGINAL, label="Marginal upper bound")
    ax_train.plot(implicit_steps, implicit_values, color=COLOR_IMPLICIT, label="Implicit likelihood")
    ax_train.axhline(truth, color=COLOR_ANALYTIC, linestyle="--", label=f"Closed form ({truth:.3f})")
    ax_train.set_xlabel("training step")
    ax_train.set_ylabel("EIG (nats)")
    ax_train.set_title(f"Training at  x = {REFERENCE_X:g}")
    ax_train.legend(loc="best")

    counts = list(INNER_COUNTS)
    _draw_inner_band(ax_inner, counts, nmc_reps, COLOR_NMC, "NMC")
    _draw_inner_band(ax_inner, counts, vnmc_reps, COLOR_VNMC, "VNMC")
    ax_inner.axhline(truth, color=COLOR_ANALYTIC, linestyle="--", label="Closed form")
    ax_inner.set_xscale("log", base=2)
    ax_inner.set_xlabel("inner samples  M")
    ax_inner.set_ylabel("EIG (nats)")
    ax_inner.set_title("Inner samples  (band: ±1 s.d.)")
    ax_inner.legend(loc="best")
    # Keep the closed form readable when NMC at M = 1 is far above it.
    nmc_mean = nmc_reps.mean(dim=0)
    ax_inner.set_ylim(truth - 0.15, 1.95)
    ax_inner.text(
        0.98,
        0.96,
        f"NMC at M = 1 is {float(nmc_mean[0]):.2f} nats",
        transform=ax_inner.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLOR_NMC,
    )

    path = ASSETS / "estimator_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)

    def _final(history: list[tuple[int, float]]) -> float:
        return round(history[-1][1], 4)

    return {
        "reference_eig": round(truth, 4),
        "posterior_final": _final(posterior_history),
        "marginal_final": _final(marginal_history),
        "implicit_final": _final(implicit_history),
        "nmc_mean_at_M": {str(m): round(float(v), 4) for m, v in zip(INNER_COUNTS, nmc_mean)},
        "vnmc_mean_at_M": {str(m): round(float(v), 4) for m, v in zip(INNER_COUNTS, vnmc_reps.mean(dim=0))},
    }


def _write_sequential_figure() -> dict[str, object]:
    """Compare myopic maximum-EIG designs with random designs."""
    _seed(SEED + 2)
    candidates = torch.linspace(DESIGN_LOW, DESIGN_HIGH, SEQUENTIAL_CANDIDATES)
    true_theta = torch.tensor(TRUE_THETA)

    def choose_analytic(mean: Tensor, cov: Tensor) -> Tensor:
        scores = analytic_eig(candidates, cov, SIGMA)
        return candidates[int(torch.argmax(scores))]

    def choose_variational(mean: Tensor, cov: Tensor) -> Tensor:
        net, _history = train_posterior(
            mean,
            cov,
            sigma=SIGMA,
            steps=SEQUENTIAL_TRAIN_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            init_at_prior=True,
            verbose=False,
        )
        scores = estimate_posterior(net, candidates, mean, cov, SIGMA, SEQUENTIAL_EVAL_SAMPLES)
        return candidates[int(torch.argmax(scores))]

    def choose_implicit(mean: Tensor, cov: Tensor) -> Tensor:
        model, _history = train_implicit(
            mean,
            cov,
            sigma=SIGMA,
            steps=SEQUENTIAL_TRAIN_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            verbose=False,
        )
        scores = estimate_implicit(model, candidates, mean, cov, SIGMA, SEQUENTIAL_EVAL_SAMPLES)
        return candidates[int(torch.argmax(scores))]

    def choose_random(mean: Tensor, cov: Tensor) -> Tensor:
        del mean, cov
        return candidates[int(torch.randint(candidates.shape[0], ()))]

    print("  closed-form policy", flush=True)
    analytic = _rollout(choose_analytic, true_theta, SEED + 20, "analytic")
    print("  variational policy", flush=True)
    variational = _rollout(choose_variational, true_theta, SEED + 21, "variational")
    print("  implicit-likelihood policy", flush=True)
    implicit = _rollout(choose_implicit, true_theta, SEED + 22, "implicit")
    random_runs = [
        _rollout(choose_random, true_theta, SEED + 100 + index, "random", quiet=True)
        for index in range(SEQUENTIAL_RANDOM_RUNS)
    ]
    random_logdet = torch.tensor([run["logdet"] for run in random_runs], dtype=torch.float32)
    random_median = random_logdet.median(dim=0).values
    random_low = torch.quantile(random_logdet, 0.1, dim=0)
    random_high = torch.quantile(random_logdet, 0.9, dim=0)
    median_index = int(torch.argmin((random_logdet[:, -1] - random_median[-1]).abs()))
    typical_random = random_runs[median_index]
    experiments = list(range(SEQUENTIAL_EXPERIMENTS + 1))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), constrained_layout=True)
    ax_vol, ax_des, ax_ell = axes
    ax_vol.plot(experiments, analytic["logdet"], color=COLOR_ANALYTIC, marker="o", label="Max EIG, closed form")
    ax_vol.plot(
        experiments,
        variational["logdet"],
        color=COLOR_VARIATIONAL,
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgewidth=1.7,
        markersize=8,
        label="Max EIG, posterior bound",
    )
    ax_vol.plot(
        experiments,
        implicit["logdet"],
        color=COLOR_IMPLICIT,
        marker="s",
        linestyle="none",
        markerfacecolor="none",
        markeredgewidth=1.5,
        markersize=7,
        label="Max EIG, implicit likelihood",
    )
    ax_vol.fill_between(
        experiments,
        random_low.numpy(),
        random_high.numpy(),
        color=COLOR_RANDOM,
        alpha=0.22,
        label="Random, 10–90%",
    )
    ax_vol.plot(experiments, random_median.numpy(), color=COLOR_RANDOM, label="Random, median")
    ax_vol.set_xlabel("experiments so far")
    ax_vol.set_ylabel(r"log det $\Sigma$")
    ax_vol.set_title("Posterior volume")
    ax_vol.legend(loc="best", fontsize=8)

    _draw_designs(ax_des, random_runs, analytic["designs"], variational["designs"], implicit["designs"])
    ax_des.set_xlabel("experiment")
    ax_des.set_ylabel("chosen design  x")
    ax_des.set_title("Where each policy measures")
    ax_des.set_ylim(DESIGN_LOW - 0.15, DESIGN_HIGH + 0.15)

    prior_mean, prior_cov = standard_prior()
    _add_ellipse(ax_ell, prior_mean, prior_cov, color="0.55", linestyle=":", label="Prior")
    _add_ellipse(ax_ell, typical_random["mean"], typical_random["cov"], color=COLOR_RANDOM, label="Random")
    _add_ellipse(ax_ell, analytic["mean"], analytic["cov"], color=COLOR_ANALYTIC, label="Closed-form EIG")
    _add_ellipse(
        ax_ell,
        variational["mean"],
        variational["cov"],
        color=COLOR_VARIATIONAL,
        linestyle="--",
        label="Posterior-bound EIG",
    )
    _add_ellipse(
        ax_ell,
        implicit["mean"],
        implicit["cov"],
        color=COLOR_IMPLICIT,
        linestyle=(0, (2, 1.5)),
        label="Implicit-likelihood EIG",
    )
    ax_ell.scatter([TRUE_THETA[0]], [TRUE_THETA[1]], color="black", marker="*", s=80, zorder=5, label="True θ")
    ax_ell.set_xlim(-3.2, 3.2)
    ax_ell.set_ylim(-3.2, 3.2)
    ax_ell.set_aspect("equal", adjustable="box")
    ax_ell.grid(False)
    ax_ell.set_xlabel(r"$\theta_0$")
    ax_ell.set_ylabel(r"$\theta_1$")
    ax_ell.set_title("95% posterior after 8 experiments")
    ax_ell.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)

    path = ASSETS / "sequential_design.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)
    return {
        "analytic_designs": [round(float(x), 4) for x in analytic["designs"]],
        "variational_designs": [round(float(x), 4) for x in variational["designs"]],
        "implicit_designs": [round(float(x), 4) for x in implicit["designs"]],
        "final_logdet": {
            "analytic": round(analytic["logdet"][-1], 4),
            "variational": round(variational["logdet"][-1], 4),
            "implicit": round(implicit["logdet"][-1], 4),
            "random_median": round(float(random_median[-1]), 4),
        },
    }


def _rollout(
    choose: Callable[[Tensor, Tensor], Tensor],
    true_theta: Tensor,
    seed: int,
    label: str,
    quiet: bool = False,
) -> dict[str, object]:
    """Run one sequential experiment and record volume, designs, and the fit.

    Args:
        choose: Maps the current posterior mean and covariance to a design.
        true_theta: Parameter that generates the observations.
        seed: Seed for the observation noise and for any randomness in ``choose``.
        label: Name printed with each accepted design.
        quiet: Skip the per-experiment log line.
    """
    _seed(seed)
    mean, cov = standard_prior()
    logdet = [logdet_covariance(cov)]
    designs: list[float] = []
    for _ in range(SEQUENTIAL_EXPERIMENTS):
        choice = choose(mean.detach(), cov.detach()).reshape(())
        feat = features(choice)
        observation = torch.dot(feat, true_theta) + SIGMA * torch.randn(())
        mean, cov = update_posterior(mean, cov, choice, observation, SIGMA)
        logdet.append(logdet_covariance(cov))
        designs.append(float(choice))
        if not quiet:
            print(f"    {label} x = {float(choice):+.2f}   logdet = {logdet[-1]:.3f}", flush=True)
    return {"logdet": logdet, "designs": designs, "mean": mean.detach(), "cov": cov.detach()}


def _draw_designs(
    ax,
    random_runs: list[dict[str, object]],
    analytic: list[float],
    variational: list[float],
    implicit: list[float],
) -> None:
    """Draw chosen designs for the random cloud and the three EIG policies."""
    steps = list(range(1, SEQUENTIAL_EXPERIMENTS + 1))
    for run in random_runs:
        designs = run["designs"]
        assert isinstance(designs, list)
        ax.scatter(steps, designs, s=10, color=COLOR_RANDOM, alpha=0.18, linewidths=0)
    ax.plot(steps, analytic, color=COLOR_ANALYTIC, marker="o", label="Closed-form EIG")
    ax.plot(steps, variational, color=COLOR_VARIATIONAL, marker="o", linestyle="--", label="Posterior-bound EIG")
    ax.plot(steps, implicit, color=COLOR_IMPLICIT, marker="s", linestyle=":", label="Implicit-likelihood EIG")
    ax.scatter([], [], s=18, color=COLOR_RANDOM, label="Random")
    ax.legend(loc="best", fontsize=8)


def _add_ellipse(ax, mean: Tensor, cov: Tensor, **kwargs) -> None:
    values, vectors = torch.linalg.eigh(cov)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    angle = math.degrees(math.atan2(float(vectors[1, 0]), float(vectors[0, 0])))
    radius = math.sqrt(CHI2_95)
    ellipse = Ellipse(
        xy=(float(mean[0]), float(mean[1])),
        width=2 * radius * math.sqrt(float(values[0])),
        height=2 * radius * math.sqrt(float(values[1])),
        angle=angle,
        fill=False,
        linewidth=1.8,
        **kwargs,
    )
    ax.add_patch(ellipse)
    ax.plot(float(mean[0]), float(mean[1]), marker="+", color=kwargs.get("color", "black"), markersize=6)


def _draw_inner_band(ax, counts: list[int], reps: Tensor, color: str, label: str) -> None:
    center = reps.mean(dim=0)
    spread = reps.std(dim=0, unbiased=False)
    ax.plot(counts, center.numpy(), color=color, marker="o", label=label)
    ax.fill_between(counts, (center - spread).numpy(), (center + spread).numpy(), color=color, alpha=0.18, linewidth=0)


def _mark(ax, grid: Tensor, values: Tensor, color: str, marker: str, label: str) -> None:
    ax.plot(grid.numpy(), values.detach().numpy(), linestyle="none", marker=marker, color=color, markersize=5.5, label=label)


def _history_arrays(history: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    steps = [step for step, _value in history]
    values = [value for _step, value in history]
    return steps, values


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def _seed(seed: int) -> None:
    torch.manual_seed(seed)


def _print_summary(results: dict[str, object]) -> None:
    eig = results.get("eig")
    if isinstance(eig, dict):
        print("EIG grid, max |estimate − truth|:", flush=True)
        for name in ("posterior", "marginal", "vnmc", "implicit", "nmc"):
            stats = eig.get(name)
            if isinstance(stats, dict):
                print(f"  {name:10s}  {stats['max_abs_error']:.4f}   mean error {stats['mean_error']:+.4f}", flush=True)
    sequential = results.get("sequential")
    if isinstance(sequential, dict):
        final = sequential.get("final_logdet")
        if isinstance(final, dict):
            print("Final log det Σ:", flush=True)
            for name, value in final.items():
                print(f"  {name:16s}  {value:.4f}", flush=True)


if __name__ == "__main__":
    main()
