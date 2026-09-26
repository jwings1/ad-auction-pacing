"""Run one bidder over one day of auctions and collect metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .auction import second_price
from .bidders import Bidder
from .market import AuctionStream


@dataclass
class RunResult:
    strategy: str
    expected_value: float      # sum of ctr * value over auctions won
    realized_value: float      # value of clicks actually received
    spend: float
    budget: float
    wins: int
    exhausted_hour: float      # hour the budget ran out (nan if never)
    hourly_spend: np.ndarray = field(repr=False)

    @property
    def utilization(self) -> float:
        return self.spend / self.budget if self.budget else float("nan")

    @property
    def roi(self) -> float:
        return self.expected_value / self.spend if self.spend else float("nan")


def estimate_values(stream: AuctionStream, bias: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """The bidder's value model: true value times a bias and mean-one lognormal noise."""
    if noise == 0.0:
        return bias * stream.value
    rng = np.random.default_rng([seed, 2])  # value-model noise
    return bias * stream.value * np.exp(rng.normal(-0.5 * noise**2, noise, len(stream)))


def simulate(stream: AuctionStream, bidder: Bidder, value_est: np.ndarray | None = None,
             click_noise: bool = True, seed: int = 0, hours: int = 24) -> RunResult:
    rng = np.random.default_rng([seed, 0])  # click stream
    v_hat = stream.value if value_est is None else value_est
    clicks_u = rng.random(len(stream))

    exp_value = realized = spend = 0.0
    wins = 0
    hourly = np.zeros(hours)
    exhausted = float("nan")

    for t in range(len(stream)):
        b = bidder.bid(float(v_hat[t]), t)
        cost = 0.0
        if b > 0.0:
            won, price = second_price(b, float(stream.competitor_bid[t]))
            if won:
                wins += 1
                ctr = float(stream.ctr[t])
                exp_value += ctr * float(stream.value[t])
                if click_noise:
                    if clicks_u[t] < ctr:
                        cost = price
                        realized += float(stream.value[t])
                else:
                    cost = ctr * price
                    realized += ctr * float(stream.value[t])
        bidder.update(cost, t)
        spend += cost
        hourly[stream.hour[t]] += cost
        if np.isnan(exhausted) and bidder.remaining < 0.01 * bidder.budget:
            exhausted = float(stream.hour[t])

    return RunResult(bidder.name, exp_value, realized, spend, bidder.budget, wins, exhausted, hourly)
