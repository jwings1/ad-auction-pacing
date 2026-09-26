"""Reproduce every table and figure in the README.

    python experiments/run_all.py

Protocol: the step size of the pacing bidder and the participation rate of the
throttling bidder are chosen on PILOT days only; all reported numbers come from
separate TEST days. The daily budget is a fixed euro amount, set as a fraction
of what a greedy bidder would spend in an unconstrained pilot day.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adpacing import (AdaptivePacingBidder, GreedyBidder, MarketConfig, ThrottlingBidder,  # noqa: E402
                      generate_stream, hindsight_optimum, simulate)
from adpacing.simulate import estimate_values  # noqa: E402

PILOT_SEEDS = list(range(0, 5))
TEST_SEEDS = list(range(100, 120))
BUDGET_FRACTION = 0.3
STEP_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
COLORS = {"greedy": "#c0392b", "throttling": "#e67e22", "adaptive_pacing": "#2471a3"}


def day(seed):
    return generate_stream(MarketConfig(seed=seed))


def make(name, budget, n, seed, step):
    if name == "greedy":
        return GreedyBidder(budget, n)
    if name == "throttling":
        return ThrottlingBidder(budget, n, participation=budget / UNCONSTRAINED, seed=seed)
    return AdaptivePacingBidder(budget, n, step_size=step)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---- pilot: budget scale and step size --------------------------------------
pilot_days = {s: day(s) for s in PILOT_SEEDS}
UNCONSTRAINED = float(np.mean([simulate(d, GreedyBidder(1e12, len(d)), seed=s).spend for s, d in pilot_days.items()]))
BUDGET = BUDGET_FRACTION * UNCONSTRAINED

step_rows = []
for eps in STEP_GRID:
    scores = []
    for s, d in pilot_days.items():
        r = simulate(d, AdaptivePacingBidder(BUDGET, len(d), step_size=eps), seed=s)
        scores.append(r.expected_value / hindsight_optimum(d, BUDGET)["value"])
    step_rows.append({"step_size": eps, "pct_of_optimum": round(100 * np.mean(scores), 1),
                      "std": round(100 * np.std(scores), 1)})
write_csv(RES / "step_size_pilot.csv", step_rows)
STEP = max(step_rows, key=lambda r: r["pct_of_optimum"])["step_size"]
print(f"unconstrained pilot spend {UNCONSTRAINED:.1f}, budget {BUDGET:.1f}, chosen step {STEP}")

# ---- experiment 1: strategy comparison at a 30% budget ----------------------
test_days = {s: day(s) for s in TEST_SEEDS}
strategies = ["greedy", "throttling", "adaptive_pacing"]
per_run, examples = [], {}
for s, d in test_days.items():
    opt = hindsight_optimum(d, BUDGET)["value"]
    for name in strategies:
        bidder = make(name, BUDGET, len(d), s, STEP)
        r = simulate(d, bidder, seed=s)
        per_run.append({"seed": s, "strategy": name, "pct_of_optimum": 100 * r.expected_value / opt,
                        "expected_value": r.expected_value, "spend": r.spend, "utilization": r.utilization,
                        "roi": r.roi, "exhausted_hour": r.exhausted_hour})
        if s == TEST_SEEDS[0]:
            examples[name] = (r, getattr(bidder, "mu_path", None))
write_csv(RES / "comparison_runs.csv", per_run)

summary = []
for name in strategies:
    rows = [r for r in per_run if r["strategy"] == name]
    ex = [r["exhausted_hour"] for r in rows if not np.isnan(r["exhausted_hour"])]
    summary.append({
        "strategy": name,
        "pct_of_optimum": f'{np.mean([r["pct_of_optimum"] for r in rows]):.1f} ± {np.std([r["pct_of_optimum"] for r in rows]):.1f}',
        "budget_used_pct": f'{100 * np.mean([r["utilization"] for r in rows]):.1f}',
        "value_per_euro": f'{np.mean([r["roi"] for r in rows]):.2f}',
        "budget_runs_out": f"{np.median(ex):02.0f}:00 (median)" if len(ex) > len(rows) / 2 else "never (most days)",
    })
write_csv(RES / "comparison_summary.csv", summary)

# ---- experiment 2: budget sweep ---------------------------------------------
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
sweep = []
for f in fractions:
    B = f * UNCONSTRAINED
    for name in strategies:
        vals = []
        for s in TEST_SEEDS[:10]:
            d = test_days[s]
            vals.append(100 * simulate(d, make(name, B, len(d), s, STEP), seed=s).expected_value
                        / hindsight_optimum(d, B)["value"])
        sweep.append({"budget_fraction": f, "strategy": name, "pct_of_optimum": round(float(np.mean(vals)), 1)})
write_csv(RES / "budget_sweep.csv", sweep)

# ---- experiment 3: value-model miscalibration --------------------------------
biases, noises = [0.7, 1.0, 1.3], [0.0, 0.3, 0.6]
mis = []
for name in ["greedy", "adaptive_pacing"]:
    for a in biases:
        for sig in noises:
            vals = []
            for s in TEST_SEEDS[:10]:
                d = test_days[s]
                v_hat = estimate_values(d, bias=a, noise=sig, seed=s)
                r = simulate(d, make(name, BUDGET, len(d), s, STEP), value_est=v_hat, seed=s)
                vals.append(100 * r.expected_value / hindsight_optimum(d, BUDGET)["value"])
            mis.append({"strategy": name, "bias": a, "noise": sig, "pct_of_optimum": round(float(np.mean(vals)), 1)})
write_csv(RES / "miscalibration.csv", mis)

# ---- figures -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name in strategies:
    r, _ = examples[name]
    ax[0].plot(np.arange(24), np.cumsum(r.hourly_spend), label=name.replace("_", " "), color=COLORS[name], lw=2)
ax[0].axhline(BUDGET, ls="--", c="k", lw=1, label="daily budget")
ax[0].set(xlabel="hour of day", ylabel="cumulative spend (€)", title="When does the budget run out?")
ax[0].legend(frameon=False, loc="lower right")
traffic = np.bincount(test_days[TEST_SEEDS[0]].hour, minlength=24)
ax[1].bar(np.arange(24), traffic, color="#d5d8dc")
ax[1].set(xlabel="hour of day", ylabel="auctions per hour", title="Traffic peaks in the evening")
fig.tight_layout()
fig.savefig(FIG / "spend_over_day.png", dpi=150)

r, mu = examples["adaptive_pacing"]
d = test_days[TEST_SEEDS[0]]
fig, ax = plt.subplots(figsize=(6, 3.5))
counts = np.bincount(d.hour, minlength=24)
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
clock = d.hour + (np.arange(len(d)) - starts[d.hour]) / counts[d.hour]      # fractional hour of each auction
ax.plot(clock, 1 + np.array(mu), lw=1.2, color=COLORS["adaptive_pacing"], label="online: 1 + μ")
ax.axhline(hindsight_optimum(d, BUDGET)["threshold_ratio"], ls="--", c="k", lw=1, label="hindsight value/price cut-off")
ax.set(xlabel="hour of day", ylabel="value/price threshold", title="Pacing learns the optimal bid threshold")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "pacing_multiplier.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.8))
for name in strategies:
    ys = [r["pct_of_optimum"] for r in sweep if r["strategy"] == name]
    ax.plot([100 * f for f in fractions], ys, marker="o", color=COLORS[name], label=name.replace("_", " "))
ax.set(xlabel="daily budget (% of unconstrained spend)", ylabel="% of hindsight optimum", ylim=(0, 105),
       title="Pacing matters most when budgets are tight")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "budget_sweep.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, name in zip(axes, ["greedy", "adaptive_pacing"]):
    grid = np.array([[next(r["pct_of_optimum"] for r in mis if r["strategy"] == name and r["bias"] == a and r["noise"] == sg)
                      for sg in noises] for a in biases])
    im = ax.imshow(grid, vmin=40, vmax=100, cmap="Blues")
    for i in range(len(biases)):
        for j in range(len(noises)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", color="k" if grid[i, j] < 80 else "w")
    ax.set(xticks=range(len(noises)), xticklabels=noises, yticks=range(len(biases)), yticklabels=biases,
           xlabel="value-model noise (σ)", title=name.replace("_", " "))
axes[0].set_ylabel("value-model bias (×)")
fig.suptitle("% of hindsight optimum under a miscalibrated value model")
fig.tight_layout()
fig.savefig(FIG / "miscalibration.png", dpi=150)

print("summary:", *summary, sep="\n")
print("sweep:", sweep, sep="\n")
print("miscalibration:", mis, sep="\n")
print("steps:", step_rows, sep="\n")
