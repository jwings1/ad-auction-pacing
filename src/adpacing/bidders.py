"""Bidding strategies for a budget-constrained advertiser.

All bidders see an estimated value per click for each auction and must keep
total spend within a daily budget B over T auctions. A bid is only placed if
the remaining budget covers it: in a second-price auction the price per click
never exceeds our bid, so the budget can never be overspent.
"""
from __future__ import annotations

import numpy as np


class Bidder:
    name = "base"

    def __init__(self, budget: float, n_auctions: int):
        self.budget = float(budget)
        self.remaining = float(budget)
        self.n_auctions = n_auctions

    def bid(self, value_est: float, t: int) -> float:
        raise NotImplementedError

    def update(self, cost: float, t: int) -> None:
        self.remaining -= cost

    def cap(self, b: float) -> float:
        return b if b <= self.remaining else 0.0


class GreedyBidder(Bidder):
    """Bids its value (truthful in a second-price auction) until the budget runs out."""

    name = "greedy"

    def bid(self, value_est: float, t: int) -> float:
        return self.cap(value_est)


class ThrottlingBidder(Bidder):
    """Bids truthfully in a random fraction p of auctions.

    p is set from a pilot day: p = budget / (spend a greedy bidder would have
    with no budget). This spreads spend across the day but ignores which
    auctions are good value.
    """

    name = "throttling"

    def __init__(self, budget: float, n_auctions: int, participation: float, seed: int = 0):
        super().__init__(budget, n_auctions)
        self.p = float(np.clip(participation, 0.0, 1.0))
        self.rng = np.random.default_rng([seed, 1])  # independent of the click stream

    def bid(self, value_est: float, t: int) -> float:
        if self.rng.random() >= self.p:
            return 0.0
        return self.cap(value_est)


class AdaptivePacingBidder(Bidder):
    """Dual-descent budget pacing (Balseiro & Gur, 2019).

    Bid b_t = v_t / (1 + mu_t). The multiplier mu is the shadow price of the
    budget: it rises when we spend faster than the target rate rho = B / T and
    falls when we spend slower,

        mu_{t+1} = clip(mu_t - eps * (1 - c_t / rho), 0, mu_max).

    In a second-price auction we win iff v_t / d_t > 1 + mu_t, so pacing is an
    online version of the hindsight rule "buy the auctions with the best
    value-to-price ratio until the budget is spent".
    """

    name = "adaptive_pacing"

    def __init__(self, budget: float, n_auctions: int, step_size: float = 0.002, mu_max: float = 20.0, mu_init: float = 0.0):
        super().__init__(budget, n_auctions)
        self.eps = step_size
        self.mu_max = mu_max
        self.mu = mu_init
        self.rho = budget / n_auctions
        self.mu_path: list[float] = []

    def bid(self, value_est: float, t: int) -> float:
        return self.cap(value_est / (1.0 + self.mu))

    def update(self, cost: float, t: int) -> None:
        super().update(cost, t)
        self.mu = min(max(self.mu - self.eps * (1.0 - cost / self.rho), 0.0), self.mu_max)
        self.mu_path.append(self.mu)
