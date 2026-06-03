"""
Trade Market Scanner — Phase 3.

Scans every team in the league and evaluates all 1-for-1 and 2-for-1 trade
combinations (from our perspective: we send 1 or 2 players, we receive 1).

Pipeline per candidate trade
----------------------------
1. Pre-filter by asset value: skip trades where net_asset_value < asset_value_floor
   (using per-player values already computed, so no optimizer call needed yet).
2. Run evaluate_trade on trades that pass pre-filtering.
3. Discard any trade where net_lineup_score <= 0.
4. Collect, sort by net_lineup_score descending, return top_n.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .formulas import BASE_RATINGS
from .trade_engine import calculate_asset_value, evaluate_trade, SALARY_SCALE

# ---------------------------------------------------------------------------
# Defaults (module-level so tests can reference them)
# ---------------------------------------------------------------------------

ASSET_FLOOR_DEFAULT: float = -5.0
"""Minimum acceptable net_asset_value.  Trades that cost us more than this
in asset value are pre-filtered before the optimizer is ever called."""

TOP_N_DEFAULT: int = 5
"""Maximum number of results returned by find_best_trades."""


# ---------------------------------------------------------------------------
# Internal enumeration helpers
# ---------------------------------------------------------------------------


def _precompute_values(
    roster_df: pd.DataFrame, salary_scale: float
) -> list[float]:
    return [
        calculate_asset_value(roster_df.iloc[i], salary_scale)
        for i in range(len(roster_df))
    ]


def _enumerate_1for1(
    n_mine: int,
    n_theirs: int,
    my_vals: list[float],
    their_vals: list[float],
    floor: float,
) -> list[tuple[list[int], list[int]]]:
    """All (my_indices, their_indices) pairs for 1-for-1 that pass the value floor."""
    out = []
    for i in range(n_mine):
        for j in range(n_theirs):
            if their_vals[j] - my_vals[i] >= floor:
                out.append(([i], [j]))
    return out


def _enumerate_2for1(
    n_mine: int,
    n_theirs: int,
    my_vals: list[float],
    their_vals: list[float],
    floor: float,
) -> list[tuple[list[int], list[int]]]:
    """All (my_indices, their_indices) for 2-for-1 (we send 2, receive 1) that pass
    the value floor."""
    out = []
    for i1, i2 in itertools.combinations(range(n_mine), 2):
        combined_my = my_vals[i1] + my_vals[i2]
        for j in range(n_theirs):
            if their_vals[j] - combined_my >= floor:
                out.append(([i1, i2], [j]))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_best_trades(
    my_roster_df: pd.DataFrame,
    league_rosters_dict: dict[str, pd.DataFrame],
    salary_scale: float = SALARY_SCALE,
    asset_value_floor: float = ASSET_FLOOR_DEFAULT,
    top_n: int = TOP_N_DEFAULT,
) -> list[dict]:
    """
    Scan the league for the best available trades for my team.

    Evaluates every 1-for-1 and 2-for-1 combination across all teams in
    league_rosters_dict.  "2-for-1" means we send 2 players and receive 1
    (a package deal — commonly used to consolidate depth into a star).

    Filtering (applied in this order for efficiency)
    -------------------------------------------------
    1. Asset value pre-filter: candidates where
           incoming_asset_value - outgoing_asset_value < asset_value_floor
       are discarded before the expensive optimizer is invoked.
    2. Lineup score filter: trades with net_lineup_score <= 0 are discarded
       after evaluate_trade runs.

    Parameters
    ----------
    my_roster_df        : my team's roster DataFrame (one row per player).
    league_rosters_dict : dict mapping team name → roster DataFrame.
    salary_scale        : passed through to calculate_asset_value and
                          evaluate_trade (default: 1 000, salary in $K).
    asset_value_floor   : minimum acceptable net_asset_value delta (default −5).
                          Prevents giving up significantly more value than received.
    top_n               : maximum number of results to return (default 5).

    Returns
    -------
    list of up to top_n dicts sorted by net_lineup_score descending.  Each dict:
        'team'             : str   — trade-partner key from league_rosters_dict
        'trade_type'       : str   — '1-for-1' or '2-for-1'
        'incoming'         : list  — player dicts arriving on my roster
        'outgoing'         : list  — player dicts leaving my roster
        'net_lineup_score' : float — improvement to my optimal lineup score (> 0)
        'net_asset_value'  : float — change in my total asset value
        'new_score'        : float — my new optimal lineup score after the trade
    """
    if len(my_roster_df) < 5:
        raise ValueError(
            f"My roster has only {len(my_roster_df)} players; at least 5 required."
        )

    my_vals = _precompute_values(my_roster_df, salary_scale)
    n_mine = len(my_roster_df)

    passing: list[dict] = []

    for team_name, their_roster_df in league_rosters_dict.items():
        if their_roster_df is None or len(their_roster_df) < 1:
            continue

        their_vals = _precompute_values(their_roster_df, salary_scale)
        n_theirs = len(their_roster_df)

        # Build list of (my_indices, their_indices, trade_type) candidates
        candidates: list[tuple[list[int], list[int], str]] = []

        # 1-for-1: roster size is unchanged — valid when n_mine >= 5
        for my_idx, their_idx in _enumerate_1for1(
            n_mine, n_theirs, my_vals, their_vals, asset_value_floor
        ):
            candidates.append((my_idx, their_idx, "1-for-1"))

        # 2-for-1: roster shrinks by 1 — need n_mine >= 6 to keep ≥ 5 after trade
        if n_mine >= 6:
            for my_idx, their_idx in _enumerate_2for1(
                n_mine, n_theirs, my_vals, their_vals, asset_value_floor
            ):
                candidates.append((my_idx, their_idx, "2-for-1"))

        for my_idxs, their_idxs, trade_type in candidates:
            # Guard: post-trade roster must have ≥ 5 players
            post_size = n_mine - len(my_idxs) + len(their_idxs)
            if post_size < 5:
                continue

            incoming = [their_roster_df.iloc[j] for j in their_idxs]

            try:
                result = evaluate_trade(
                    my_roster_df,
                    incoming_players=incoming,
                    outgoing_players=my_idxs,   # integer positional indices
                    salary_scale=salary_scale,
                )
            except Exception:
                continue

            if result["net_lineup_score"] <= 0:
                continue

            passing.append({
                "team":             team_name,
                "trade_type":       trade_type,
                "incoming":         [their_roster_df.iloc[j].to_dict() for j in their_idxs],
                "outgoing":         [my_roster_df.iloc[i].to_dict() for i in my_idxs],
                "net_lineup_score": result["net_lineup_score"],
                "net_asset_value":  result["net_asset_value"],
                "new_score":        result["new_score"],
            })

    passing.sort(key=lambda t: t["net_lineup_score"], reverse=True)
    return passing[:top_n]
