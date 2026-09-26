"""Synthetic stream of search-ad auctions for one day of traffic.

Each auction is a single ad slot on a search results page. The advertiser (us)
bids a cost-per-click; the ad is shown if our bid beats the highest competing
bid, and we pay that competing bid per click (second-price rule).

What the stream encodes:
  * hour      - time of day; traffic peaks in the evening
  * segment   - keyword group (e.g. "hotels rome", "cheap flights") with its own
                click-through rate (CTR), conversion rate and booking value
  * value     - expected value of one click = P(booking | click) * commission,
                with per-auction noise for user intent (evening users convert more)
  * competitor_bid - highest competing per-click bid; competitors bid around the
                segment's average value and bid harder at peak hours
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketConfig:
    n_auctions: int = 20_000
    n_segments: int = 20
    hours: int = 24
    competitor_sigma: float = 0.5   # dispersion of competing bids
    intent_sigma: float = 0.3       # per-auction dispersion of click value
    seed: int = 0


@dataclass(frozen=True)
class AuctionStream:
    hour: np.ndarray
    segment: np.ndarray
    ctr: np.ndarray
    value: np.ndarray
    competitor_bid: np.ndarray

    def __len__(self) -> int:
        return len(self.value)


def traffic_intensity(hours: int) -> np.ndarray:
    """Relative auction volume per hour, peaking around 19:00."""
    h = np.arange(hours)
    return 1.0 + 0.8 * np.sin((h - 13) / hours * 2 * np.pi)


def generate_stream(cfg: MarketConfig) -> AuctionStream:
    rng = np.random.default_rng(cfg.seed)

    intensity = traffic_intensity(cfg.hours)
    hour = np.sort(rng.choice(cfg.hours, size=cfg.n_auctions, p=intensity / intensity.sum()))

    seg_ctr = rng.uniform(0.02, 0.10, cfg.n_segments)
    seg_cvr = rng.uniform(0.01, 0.08, cfg.n_segments)
    seg_commission = rng.lognormal(np.log(25.0), 0.4, cfg.n_segments)
    segment = rng.integers(0, cfg.n_segments, cfg.n_auctions)

    intent = np.exp(
        0.25 * np.sin((hour - 14) / cfg.hours * 2 * np.pi)
        + rng.normal(0.0, cfg.intent_sigma, cfg.n_auctions)
    )
    seg_value = seg_cvr * seg_commission
    value = seg_value[segment] * intent

    pressure = 0.6 + 0.3 * intensity[hour] / intensity.max()
    competitor_bid = seg_value[segment] * pressure * rng.lognormal(0.0, cfg.competitor_sigma, cfg.n_auctions)

    return AuctionStream(
        hour=hour,
        segment=segment,
        ctr=seg_ctr[segment],
        value=value,
        competitor_bid=competitor_bid,
    )
