"""Hindsight optimum: the best possible outcome with perfect knowledge.

Knowing every competing bid in advance, winning auction i costs ctr_i * d_i in
expectation and yields ctr_i * v_i. Maximising value under the budget is a
fractional knapsack, solved exactly by taking auctions in decreasing order of
v_i / d_i. No online strategy can beat this in expectation, so it is the
benchmark every strategy is scored against.
"""
from __future__ import annotations

import numpy as np

from .market import AuctionStream


def hindsight_optimum(stream: AuctionStream, budget: float) -> dict:
    exp_cost = stream.ctr * stream.competitor_bid
    exp_value = stream.ctr * stream.value
    worth = stream.value > stream.competitor_bid          # never pay more than value
    order = np.argsort(-(stream.value / stream.competitor_bid))
    order = order[worth[order]]

    cum_cost = np.cumsum(exp_cost[order])
    full = cum_cost <= budget
    n_full = int(full.sum())
    value = float(exp_value[order[:n_full]].sum())
    spend = float(cum_cost[n_full - 1]) if n_full else 0.0
    if n_full < len(order):                               # fractional last item
        frac = (budget - spend) / exp_cost[order[n_full]]
        value += frac * exp_value[order[n_full]]
        spend = budget
    return {"value": value, "spend": spend, "threshold_ratio": float(stream.value[order[n_full - 1]] / stream.competitor_bid[order[n_full - 1]]) if n_full else np.inf}
