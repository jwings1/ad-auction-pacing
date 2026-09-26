"""Single-slot second-price (Vickrey) auction with an optional reserve price."""
from __future__ import annotations


def second_price(bid: float, competitor_bid: float, reserve: float = 0.0) -> tuple[bool, float]:
    """Return (won, price_per_click).

    We win if our bid strictly exceeds the highest competing bid and the
    reserve; we then pay the larger of the two, never our own bid.
    """
    threshold = max(competitor_bid, reserve)
    if bid > threshold:
        return True, threshold
    return False, 0.0
