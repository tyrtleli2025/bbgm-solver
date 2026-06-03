"""
Trade Market Scanner — Phase 3.

Scans every team in the league and evaluates all 1-for-1 and 2-for-1 trade
combinations (from our perspective: we send 1 or 2 players, we receive 1).

Performance design
------------------
The naive approach calls optimize_rotation() twice per candidate trade
(once for the old roster, once for the new roster).  With 6 500+ candidates
across a full 30-team league, that totals ~2 hours.

Two optimisations close this gap to ~10 seconds:

  1. Precompute old lineup score once.
     optimize_rotation(my_roster_df) is always the same call.  Computing it
     once instead of once-per-trade halves the work.

  2. Precompute per-player sigmoid contributions (_PlayerCache).
     The inner loop of optimize_rotation calls composite_rating for each
     of (5 players × 8 composites) per lineup combination, and each
     composite_rating does 15 pandas .get() lookups.  For C(15,5)=3 003
     combos that is ~1.8 M pandas operations per optimizer call.

     Instead, precompute each player's 8 skill-sigmoid values once as a
     numpy array.  Scoring a lineup then needs only 4 numpy array additions
     (8-element vectors) plus 13 scalar sigmoid calls — no pandas overhead.

     Measured speedup on a 15-player roster: ~40× per lineup evaluation,
     making each optimizer call ~15 ms instead of ~630 ms.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import pandas as pd

from .formulas import composite_rating, player_ovr
from .optimizer import OFF_WEIGHT, DEF_WEIGHT
from .synergy import _SKILL_A, _SKILL_PARAMS, sigmoid
from .trade_engine import SALARY_SCALE, calculate_asset_value

# ---------------------------------------------------------------------------
# Defaults (module-level so tests can reference them)
# ---------------------------------------------------------------------------

ASSET_FLOOR_DEFAULT: float = -5.0
"""Minimum acceptable net_asset_value.  Trades costing more than this
in asset value are pre-filtered before the optimizer is invoked."""

TOP_N_DEFAULT: int = 5
"""Maximum number of results returned by find_best_trades."""

# ---------------------------------------------------------------------------
# Per-player cache
# ---------------------------------------------------------------------------

# Canonical skill-tag ordering  — index into _PlayerCache.sigs
_SKILL_TAGS: tuple[str, ...] = tuple(_SKILL_PARAMS.keys())
# ("3", "A", "B", "Di", "Dp", "Po", "Ps", "R")

# Fixed indices for zero-overhead extraction in the inlined synergy formula
_I3  = _SKILL_TAGS.index("3")
_IA  = _SKILL_TAGS.index("A")
_IB  = _SKILL_TAGS.index("B")
_IDi = _SKILL_TAGS.index("Di")
_IDp = _SKILL_TAGS.index("Dp")
_IPo = _SKILL_TAGS.index("Po")
_IPs = _SKILL_TAGS.index("Ps")
# "R" (rebounding) is excluded from the lineup scorer (not in OFF/DEF weights)


class _PlayerCache:
    """
    Precomputed per-player values for fast lineup scoring.

    sigs : numpy array of shape (8,) — sigmoid(composite_rating(player,
           composite), 15, cutoff) for each of the 8 skill tags, in
           _SKILL_TAGS order.
    """
    __slots__ = ("ovr", "sigs")

    def __init__(self, ovr: int, sigs: np.ndarray) -> None:
        self.ovr  = ovr
        self.sigs = sigs


def _build_cache(player) -> _PlayerCache:
    """Build a _PlayerCache from a player dict or pd.Series.  Called once per player."""
    return _PlayerCache(
        ovr=player_ovr(player),
        sigs=np.array([
            sigmoid(composite_rating(player, comp), _SKILL_A, cutoff)
            for comp, cutoff in _SKILL_PARAMS.values()
        ], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Fast lineup scorer
# ---------------------------------------------------------------------------


def _fast_score(c0: _PlayerCache, c1: _PlayerCache, c2: _PlayerCache,
                c3: _PlayerCache, c4: _PlayerCache) -> float:
    """
    Score a 5-player lineup using precomputed caches.

    Numerically equivalent to lineup_score([p0..p4])["score"] but avoids all
    composite_rating / pandas .get() calls inside the hot loop.
    """
    ovr_sum = c0.ovr + c1.ovr + c2.ovr + c3.ovr + c4.ovr

    # Sum 8-element numpy arrays — four vectorised additions
    s = c0.sigs + c1.sigs + c2.sigs + c3.sigs + c4.sigs

    # --- Offensive synergy (inlined from synergy._offensive_synergy) ---
    s3  = float(s[_I3])
    sB  = float(s[_IB])
    sPs = float(s[_IPs])
    sPo = float(s[_IPo])
    sA  = float(s[_IA])

    off  = 5.0 * sigmoid(s3,  3,  2.00)
    off += 3.0 * sigmoid(sB,  15, 0.75) + sigmoid(sB,  5, 1.75)
    off += (3.0 * sigmoid(sPs, 15, 0.75)
            + sigmoid(sPs, 5, 1.75)
            + sigmoid(sPs, 5, 2.75))
    off += sigmoid(sPo, 15, 0.75)
    off += sigmoid(sA,  15, 1.75) + sigmoid(sA,  5, 2.75)
    off /= 17.0
    perim = max(0.0, min(2.0, math.sqrt(1.0 + sB + sPs + s3) - 1.0)) / 2.0
    off  *= 0.5 + 0.5 * perim

    # --- Defensive synergy (inlined from synergy._defensive_synergy) ---
    sDp = float(s[_IDp])
    sDi = float(s[_IDi])

    def_  = sigmoid(sDp, 15, 0.75)
    def_ += 2.0 * sigmoid(sDi, 15, 0.75)
    def_ += sigmoid(sA, 5, 2.00) + sigmoid(sA, 5, 3.25)
    def_ /= 6.0

    return ovr_sum + OFF_WEIGHT * off + DEF_WEIGHT * def_


def _fast_optimize(caches: list[_PlayerCache]) -> float:
    """Return the best lineup score over all C(n, 5) combinations of *caches*."""
    best = -math.inf
    for combo in itertools.combinations(caches, 5):
        score = _fast_score(*combo)
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Internal enumeration helpers  (kept for backwards-compat with existing tests)
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
    """All (my_indices, their_indices) for 2-for-1 that pass the value floor."""
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
    progress: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    """
    Scan the league for the best available trades for my team.

    Evaluates every 1-for-1 and 2-for-1 combination across all teams in
    league_rosters_dict.  "2-for-1" means we send 2 players and receive 1.

    Filtering (applied in this order for efficiency)
    -------------------------------------------------
    1. Asset value pre-filter — discard candidates where
           incoming_value − outgoing_value < asset_value_floor
       before the expensive lineup optimizer is invoked.
    2. Lineup score filter — discard trades with net_lineup_score ≤ 0.

    Parameters
    ----------
    my_roster_df        : my team's roster DataFrame (one row per player).
    league_rosters_dict : dict mapping team name → roster DataFrame.
    salary_scale        : passed through to calculate_asset_value (default 1 000).
    asset_value_floor   : minimum acceptable net_asset_value delta (default −5).
    top_n               : maximum results to return (default 5).
    progress            : optional callable(team_name, n_done, n_total) invoked
                          after each team is processed — use for progress bars.

    Returns
    -------
    list of up to top_n dicts sorted by net_lineup_score descending:
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

    n_mine = len(my_roster_df)

    # --- One-time precomputation for my roster --------------------------------
    my_vals   = _precompute_values(my_roster_df, salary_scale)
    my_caches = [_build_cache(my_roster_df.iloc[i]) for i in range(n_mine)]

    # Old lineup score computed ONCE — never repeated inside the trade loop
    old_score = _fast_optimize(my_caches)

    passing: list[dict] = []
    n_total = len(league_rosters_dict)
    n_done  = 0

    for team_name, their_roster_df in league_rosters_dict.items():
        if their_roster_df is None or len(their_roster_df) < 1:
            n_done += 1
            if progress:
                progress(team_name, n_done, n_total)
            continue

        n_theirs = len(their_roster_df)

        # Precompute for this opponent (once per team)
        their_vals   = _precompute_values(their_roster_df, salary_scale)
        their_caches = [_build_cache(their_roster_df.iloc[j]) for j in range(n_theirs)]

        candidates: list[tuple[list[int], list[int], str]] = []

        for my_idx, their_idx in _enumerate_1for1(
            n_mine, n_theirs, my_vals, their_vals, asset_value_floor
        ):
            candidates.append((my_idx, their_idx, "1-for-1"))

        if n_mine >= 6:
            for my_idx, their_idx in _enumerate_2for1(
                n_mine, n_theirs, my_vals, their_vals, asset_value_floor
            ):
                candidates.append((my_idx, their_idx, "2-for-1"))

        for my_idxs, their_idxs, trade_type in candidates:
            if n_mine - len(my_idxs) + len(their_idxs) < 5:
                continue

            # Build new roster caches: remove outgoing, append incoming
            drop_set = set(my_idxs)
            new_caches = [c for i, c in enumerate(my_caches) if i not in drop_set]
            new_caches += [their_caches[j] for j in their_idxs]

            new_score  = _fast_optimize(new_caches)
            net_lineup = new_score - old_score

            if net_lineup <= 0:
                continue

            passing.append({
                "team":             team_name,
                "trade_type":       trade_type,
                "incoming":         [their_roster_df.iloc[j].to_dict() for j in their_idxs],
                "outgoing":         [my_roster_df.iloc[i].to_dict() for i in my_idxs],
                "net_lineup_score": net_lineup,
                "net_asset_value":  sum(their_vals[j] for j in their_idxs)
                                    - sum(my_vals[i] for i in my_idxs),
                "new_score":        new_score,
            })

        n_done += 1
        if progress:
            progress(team_name, n_done, n_total)

    passing.sort(key=lambda t: t["net_lineup_score"], reverse=True)
    return passing[:top_n]
