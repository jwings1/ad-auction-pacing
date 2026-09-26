import numpy as np
import pytest

from adpacing import (AdaptivePacingBidder, GreedyBidder, MarketConfig, ThrottlingBidder,
                      generate_stream, hindsight_optimum, second_price, simulate)


@pytest.fixture(scope="module")
def day():
    return generate_stream(MarketConfig(n_auctions=4_000, seed=7))


def test_second_price_pays_competing_bid_not_own_bid():
    assert second_price(2.0, 1.2) == (True, 1.2)
    assert second_price(1.0, 1.2) == (False, 0.0)
    assert second_price(2.0, 1.2, reserve=1.5) == (True, 1.5)


@pytest.mark.parametrize("make", [
    lambda B, n: GreedyBidder(B, n),
    lambda B, n: ThrottlingBidder(B, n, participation=0.5, seed=3),
    lambda B, n: AdaptivePacingBidder(B, n, step_size=0.002),
])
@pytest.mark.parametrize("click_noise", [True, False])
def test_budget_is_never_exceeded(day, make, click_noise):
    budget = 20.0
    r = simulate(day, make(budget, len(day)), click_noise=click_noise, seed=1)
    assert r.spend <= budget + 1e-9


def test_no_strategy_beats_the_hindsight_optimum(day):
    budget = 20.0
    opt = hindsight_optimum(day, budget)["value"]
    for bidder in [GreedyBidder(budget, len(day)), AdaptivePacingBidder(budget, len(day))]:
        r = simulate(day, bidder, click_noise=False)   # deterministic costs: the bound is exact
        assert r.expected_value <= opt + 1e-9


def test_pacing_equals_greedy_when_budget_does_not_bind(day):
    huge = 1e9
    g = simulate(day, GreedyBidder(huge, len(day)), seed=2)
    p = simulate(day, AdaptivePacingBidder(huge, len(day)), seed=2)
    assert p.wins == g.wins and p.spend == pytest.approx(g.spend)


def test_pacing_bid_is_a_value_to_price_threshold():
    b = AdaptivePacingBidder(100.0, 10)
    b.mu = 0.8
    value, price = 3.0, 1.5                       # ratio 2.0 > 1 + mu -> win
    assert second_price(b.bid(value, 0), price)[0]
    value = 2.5                                   # ratio 1.67 < 1.8 -> lose
    assert not second_price(b.bid(value, 0), price)[0]


def test_pacing_beats_greedy_under_a_tight_budget(day):
    greedy_spend = simulate(day, GreedyBidder(1e9, len(day)), seed=4).spend
    budget = 0.3 * greedy_spend
    g = simulate(day, GreedyBidder(budget, len(day)), seed=4)
    p = simulate(day, AdaptivePacingBidder(budget, len(day)), seed=4)
    assert p.expected_value > g.expected_value


def test_market_is_reproducible():
    a = generate_stream(MarketConfig(n_auctions=500, seed=11))
    b = generate_stream(MarketConfig(n_auctions=500, seed=11))
    assert np.array_equal(a.competitor_bid, b.competitor_bid)
