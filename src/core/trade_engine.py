"""
Trade Asset and Evaluation Engine — Phase 2.

Provides per-player trade value scores and a trade evaluator that compares
the optimal lineup score before and after a hypothetical roster swap.

Trade Value formula
-------------------
    value = base_ovr
          + potential_bonus      (young + high-pot only)
          - age_decline_penalty  (age >= 31)
          - contract_penalty     (salary above OVR-implied fair market)
"""

from __future__ import annotations

import math
from typing import Union

import pandas as pd

from .formulas import BASE_RATINGS, player_ovr, PlayerRatings
from .optimizer import optimize_rotation

# ---------------------------------------------------------------------------
# Tuning constants  (module-level so callers can read them in assertions)
# ---------------------------------------------------------------------------

POT_WEIGHT: float = 0.5
"""Multiplier on (pot − ovr) × youth_factor for the potential bonus."""

AGE_PENALTY: float = 3.0
"""Points deducted per full year past age 30 (age 34 → −12 pts)."""

CONTRACT_PENALTY: float = 1.5
"""Points deducted per $1M above OVR-implied fair-market salary."""

SALARY_SCALE: float = 1_000.0
"""Divide raw salary column by this to get $M.  Default assumes $1 000 = $1K."""

# Fair-market salary curve: linear from $0M at OVR 40 → $30M at OVR 100.
# Calibrated to typical BBGM market rates.
_FAIR_SLOPE: float = 0.5      # $M per OVR point above the floor
_FAIR_FLOOR_OVR: float = 40.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_float(player: PlayerRatings, key: str, default: float) -> float:
    val = player.get(key) if isinstance(player, dict) else player.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def _fair_salary_m(ovr: float) -> float:
    """OVR-implied fair-market salary in $M."""
    return max(0.0, (ovr - _FAIR_FLOOR_OVR) * _FAIR_SLOPE)


# ---------------------------------------------------------------------------
# Per-player asset value
# ---------------------------------------------------------------------------


def calculate_asset_value(
    player: PlayerRatings,
    salary_scale: float = SALARY_SCALE,
) -> float:
    """
    Compute a single player's Trade Value score.

    Components
    ----------
    Base OVR
        Current player_ovr() output — the primary value driver.

    Potential bonus
        (pot − ovr) × youth_factor × POT_WEIGHT
        youth_factor = clamp((26 − age) / 6, 0, 1)
        Applied only when pot > ovr AND age < 26.

    Age decline penalty
        (age − 30) × AGE_PENALTY, applied only when age ≥ 31.

    Contract penalty
        max(0, salary_$M − fair_salary_$M) × CONTRACT_PENALTY
        Fair market: OVR 40 → $0M, OVR 100 → $30M (linear).

    Parameters
    ----------
    player       : dict or pd.Series with base ratings and optional metadata:
                   'age'    — player age in years (default 27)
                   'pot'    — potential rating 0–100 (default = current ovr)
                   'salary' — raw salary divided by salary_scale to get $M (default 0)
    salary_scale : divisor converting the raw salary value to $M.

    Returns
    -------
    float — trade value (can be negative for albatross contracts).
    """
    ovr = player_ovr(player)
    age = _get_float(player, "age", 27.0)
    pot = _get_float(player, "pot", float(ovr))
    salary_raw = _get_float(player, "salary", 0.0)

    value = float(ovr)

    # Potential bonus — only for young, still-developing players
    if pot > ovr and age < 26.0:
        youth_factor = max(0.0, (26.0 - age) / 6.0)   # 1.0 at 20, 0.0 at 26
        value += (pot - ovr) * youth_factor * POT_WEIGHT

    # Age decline penalty — linear past age 30
    if age >= 31.0:
        value -= (age - 30.0) * AGE_PENALTY

    # Contract efficiency penalty — overpaying erodes trade leverage
    salary_m = salary_raw / salary_scale
    if salary_m > 0.0:
        overpay_m = max(0.0, salary_m - _fair_salary_m(float(ovr)))
        value -= overpay_m * CONTRACT_PENALTY

    return value


def total_asset_value(
    roster_df: pd.DataFrame,
    salary_scale: float = SALARY_SCALE,
) -> float:
    """Sum of calculate_asset_value over every player in a roster DataFrame."""
    return sum(
        calculate_asset_value(roster_df.iloc[i], salary_scale)
        for i in range(len(roster_df))
    )


# ---------------------------------------------------------------------------
# Outgoing-player resolution
# ---------------------------------------------------------------------------


def _resolve_drop_positions(
    roster_df: pd.DataFrame,
    outgoing_players: list[Union[int, PlayerRatings]],
) -> list[int]:
    """
    Convert outgoing_players to a list of 0-based positional row indices.

    Each item may be:
    - int  → used directly as a positional index (0-based).
    - dict / pd.Series → matched against the roster by comparing all 15
      BASE_RATINGS values; the first matching row is used.
    """
    positions: list[int] = []
    already_dropped: set[int] = set()

    for p in outgoing_players:
        if isinstance(p, int):
            if not (0 <= p < len(roster_df)):
                raise ValueError(
                    f"Outgoing index {p} is out of range for roster of size {len(roster_df)}."
                )
            if p in already_dropped:
                raise ValueError(f"Outgoing index {p} appears more than once.")
            already_dropped.add(p)
            positions.append(p)
        else:
            matched = False
            for pos in range(len(roster_df)):
                if pos in already_dropped:
                    continue
                row = roster_df.iloc[pos]
                if all(
                    _get_float(row, r, 50.0) == _get_float(p, r, 50.0)
                    for r in BASE_RATINGS
                ):
                    already_dropped.add(pos)
                    positions.append(pos)
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    "Could not match an outgoing player to any remaining roster row "
                    "by base ratings. Pass integer positional indices for unambiguous removal."
                )

    return positions


# ---------------------------------------------------------------------------
# Trade evaluator
# ---------------------------------------------------------------------------


def evaluate_trade(
    roster_df: pd.DataFrame,
    incoming_players: list[PlayerRatings],
    outgoing_players: list[Union[int, PlayerRatings]],
    salary_scale: float = SALARY_SCALE,
) -> dict:
    """
    Evaluate the impact of a proposed trade on lineup quality and asset value.

    Parameters
    ----------
    roster_df        : current team roster — one row per player.
    incoming_players : list of players being acquired (dicts or pd.Series).
    outgoing_players : list of players being traded away — either 0-based
                       positional integers or player rating dicts/Series.
    salary_scale     : passed through to calculate_asset_value.

    Returns
    -------
    dict with keys:
        'net_lineup_score' : float — change in optimal lineup score (+ = improved)
        'net_asset_value'  : float — change in total trade value (+ = improved)
        'old_score'        : float — optimal lineup score before the trade
        'new_score'        : float — optimal lineup score after the trade
        'old_lineup'       : dict  — full optimize_rotation result for current roster
        'new_lineup'       : dict  — full optimize_rotation result for new roster
        'incoming_value'   : float — total asset value of arriving players
        'outgoing_value'   : float — total asset value of departing players
    """
    drop_positions = _resolve_drop_positions(roster_df, outgoing_players)
    drop_set = set(drop_positions)

    # Compute asset values while the outgoing players are still on the roster
    outgoing_value = sum(
        calculate_asset_value(roster_df.iloc[pos], salary_scale)
        for pos in drop_positions
    )
    incoming_value = sum(
        calculate_asset_value(p, salary_scale) for p in incoming_players
    )

    # Build the hypothetical post-trade roster
    kept_rows = [
        roster_df.iloc[i] for i in range(len(roster_df)) if i not in drop_set
    ]
    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)

    if incoming_players:
        incoming_df = pd.DataFrame([
            dict(p) if isinstance(p, dict) else p.to_dict()
            for p in incoming_players
        ])
        new_roster_df = pd.concat([kept_df, incoming_df], ignore_index=True)
    else:
        new_roster_df = kept_df

    if len(new_roster_df) < 5:
        raise ValueError(
            f"Post-trade roster has only {len(new_roster_df)} players; "
            "at least 5 are required to run the optimizer."
        )

    old_lineup = optimize_rotation(roster_df)
    new_lineup = optimize_rotation(new_roster_df)

    return {
        "net_lineup_score": new_lineup["score"] - old_lineup["score"],
        "net_asset_value":  incoming_value - outgoing_value,
        "old_score":        old_lineup["score"],
        "new_score":        new_lineup["score"],
        "old_lineup":       old_lineup,
        "new_lineup":       new_lineup,
        "incoming_value":   incoming_value,
        "outgoing_value":   outgoing_value,
    }
