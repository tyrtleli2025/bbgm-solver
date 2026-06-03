"""
Lineup Optimizer — Phase 3.

Enumerates every possible 5-man lineup from a roster DataFrame, scores each
using a Net Rating Proxy, and returns the optimal starting unit.

Net Rating Proxy
----------------
    score = Σ player_ovr  +  OFF_WEIGHT × syn["off"]  +  DEF_WEIGHT × syn["def"]

Rebounding synergy is captured inside the game simulation and already partially
reflected in individual OVRs, so it is not double-counted here.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from .formulas import player_ovr, PlayerRatings
from .synergy import calculate_lineup_synergy

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

OFF_WEIGHT: float = 15.0   # points added per unit of offensive synergy (0–1)
DEF_WEIGHT: float = 15.0   # points added per unit of defensive synergy (0–1)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def lineup_score(players: list[PlayerRatings]) -> dict:
    """
    Compute the Net Rating Proxy for a specific 5-man lineup.

    Parameters
    ----------
    players : list of exactly 5 player rating dicts or pd.Series.

    Returns
    -------
    dict with keys 'score', 'ovr_sum', 'synergy'.
    """
    if len(players) != 5:
        raise ValueError(f"Exactly 5 players required, got {len(players)}")

    ovr_sum = sum(player_ovr(p) for p in players)
    syn = calculate_lineup_synergy(players)
    score = ovr_sum + OFF_WEIGHT * syn["off"] + DEF_WEIGHT * syn["def"]
    return {"score": score, "ovr_sum": ovr_sum, "synergy": syn}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def optimize_rotation(roster_df: pd.DataFrame) -> dict:
    """
    Find the optimal 5-man starting lineup from a roster DataFrame.

    Evaluates all C(n, 5) combinations and returns the one with the highest
    Net Rating Proxy score.  For a 15-man roster this is C(15,5) = 3 003
    combinations — fast enough to run without caching.

    Parameters
    ----------
    roster_df : pd.DataFrame
        One row per player.  Columns must include the 15 BASE_RATINGS keys.
        Additional columns (e.g. 'name', 'salary') are ignored.

    Returns
    -------
    dict with keys:
        'lineup'  : pd.DataFrame  — the 5 selected players (index reset to 0–4)
        'synergy' : dict          — {'off', 'def', 'reb'} synergy multipliers
        'score'   : float         — Net Rating Proxy of the optimal lineup
        'ovr_sum' : int           — sum of the 5 individual player OVRs
    """
    n = len(roster_df)
    if n < 5:
        raise ValueError(f"Roster must have at least 5 players, got {n}")

    # Pre-compute each player's OVR once — avoids n^4 redundant regression calls.
    cached_ovr = [player_ovr(roster_df.iloc[i]) for i in range(n)]

    best_score = -math.inf
    best_indices: tuple[int, ...] = ()
    best_synergy: dict[str, float] = {}

    for combo in itertools.combinations(range(n), 5):
        players = [roster_df.iloc[i] for i in combo]

        ovr_sum = sum(cached_ovr[i] for i in combo)
        syn = calculate_lineup_synergy(players)
        score = ovr_sum + OFF_WEIGHT * syn["off"] + DEF_WEIGHT * syn["def"]

        if score > best_score:
            best_score = score
            best_indices = combo
            best_synergy = syn

    lineup = roster_df.iloc[list(best_indices)].reset_index(drop=True)
    return {
        "lineup":  lineup,
        "synergy": best_synergy,
        "score":   best_score,
        "ovr_sum": sum(cached_ovr[i] for i in best_indices),
    }
