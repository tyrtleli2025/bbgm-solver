"""
Deterministic player rating projection (develop_freeagency_draft_reference.md §1).

All projections use expected values — no noise:
  - base change: mean from age table in §1.2
  - uniform(0.4, 1.4) noise multiplier → replaced by midpoint 0.9
  - endurance young boost uniform(0, 9) → replaced by mean 4.5

Height never changes (tiny probability, negligible for projection).
"""

from __future__ import annotations

import math
from src.core.formulas import BASE_RATINGS, player_ovr

# ---------------------------------------------------------------------------
# Base change by age (§1.2 table, column "Base Value")
# ---------------------------------------------------------------------------

def _base_change_expected(age: float) -> float:
    a = int(age)
    if a <= 21:  return  2.0
    if a <= 25:  return  1.0
    if a <= 27:  return  0.0
    if a <= 29:  return -1.0
    if a <= 31:  return -2.0
    if a <= 34:  return -3.0
    if a <= 40:  return -4.0
    if a <= 43:  return -5.0
    return -6.0


# ---------------------------------------------------------------------------
# Coaching effect (§1.5) — at default level 34 this is ~0
# ---------------------------------------------------------------------------

def _coaching_effect(coaching_level: int = 34) -> float:
    """Additive multiplier on positive base change; applied symmetrically to decline."""
    x = (3 * (coaching_level - 1)) / 99 - 1
    effect_norm = 1.1 * x if x < 0 else 1.1 * math.tanh(x)
    return 0.09 * effect_norm


# ---------------------------------------------------------------------------
# Per-rating age modifiers (§1.2)
# ---------------------------------------------------------------------------

_SHOOTING  = frozenset({"ins", "ft", "fg", "tp"})
_IQ        = frozenset({"oiq", "diq"})
_ATHLETIC  = frozenset({"spd", "jmp"})
_SKILL     = frozenset({"drb", "pss", "reb"})


def _age_modifier(rating: str, age: float) -> float:
    """Modifier added to base_change before the uniform(0.4,1.4) multiplier."""
    a = int(age)

    if rating == "hgt":
        return 0.0  # not modified by normal formula

    if rating in _SHOOTING:
        if a <= 27:  return 0.0
        if a <= 29:  return 0.5
        if a <= 31:  return 1.5
        return 2.0

    if rating in _IQ:
        if a <= 21:  return 4.0
        if a <= 23:  return 3.0
        if a <= 27:  return 0.0
        if a <= 29:  return 0.5
        if a <= 31:  return 1.5
        return 2.0

    if rating == "spd":
        if a <= 27:  return  0.0
        if a <= 30:  return -2.0
        if a <= 35:  return -3.0
        if a <= 40:  return -4.0
        return -8.0

    if rating == "jmp":
        if a <= 26:  return  0.0
        if a <= 30:  return -3.0
        if a <= 35:  return -4.0
        if a <= 40:  return -5.0
        return -10.0

    if rating == "endu":
        if a <= 23:  return  4.5   # E[uniform(0, 9)]
        if a <= 30:  return  0.0
        if a <= 35:  return -2.0
        if a <= 40:  return -4.0
        return -8.0

    if rating == "stre":
        return 0.0   # only base change applies

    if rating in _SKILL:
        # Same protection as shooting, but tighter limits
        if a <= 27:  return 0.0
        if a <= 29:  return 0.5
        if a <= 31:  return 1.5
        return 2.0

    if rating == "dnk":
        if a <= 27:  return 0.0
        return 0.5   # less protection than shooting

    return 0.0


# ---------------------------------------------------------------------------
# Per-rating, per-age change limits (§1.2)
# ---------------------------------------------------------------------------

def _change_limits(rating: str, age: float) -> tuple[float, float]:
    """(lo, hi) clamp applied to the raw change before it is added to the rating."""
    a = int(age)

    if rating == "hgt":
        return (0.0, 0.0)  # never changes

    if rating in _IQ:
        if a <= 19:  return (-3.0, 32.0)
        if a <= 20:  return (-3.0, 27.0)
        if a <= 21:  return (-3.0, 22.0)
        if a <= 22:  return (-3.0, 17.0)
        if a <= 23:  return (-3.0, 12.0)
        return (-3.0, 9.0)

    if rating in ("spd", "jmp"):
        return (-12.0, 2.0)

    if rating == "endu":
        return (-11.0, 19.0)

    if rating in _SKILL:
        return (-2.0, 5.0)

    if rating == "dnk":
        return (-3.0, 13.0)

    # stre, shooting, and any others: no explicit limits in the source
    return (-15.0, 15.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project_ratings(
    ratings: dict,
    age: float,
    years_forward: int,
    coaching_level: int = 34,
) -> dict:
    """
    Project a player's 15 base ratings deterministically *years_forward* seasons.

    Uses expected values throughout (no Monte Carlo noise):
      change = clamp( (base_change + age_modifier) × 0.9,  limit_min, limit_max )
      new_rating = clamp( old_rating + change,  0, 100 )

    Parameters
    ----------
    ratings       : dict mapping each key in BASE_RATINGS to its current value.
    age           : player's age at the *start* of the current season.
    years_forward : number of offseason development cycles to apply.
    coaching_level: 1–100; 34 is the default (≈0 effect).

    Returns
    -------
    New ratings dict (values are floats).
    """
    projected = {r: float(ratings.get(r, 50.0)) for r in BASE_RATINGS}
    coaching = _coaching_effect(coaching_level)

    for year in range(years_forward):
        current_age = age + year   # age at start of this development cycle

        base = _base_change_expected(current_age)
        # Apply coaching: amplifies positive growth, slows decline
        if base > 0:
            base *= (1.0 + coaching)
        elif base < 0:
            base *= (1.0 - coaching)

        for r in BASE_RATINGS:
            if r == "hgt":
                continue  # height is fixed

            mod         = _age_modifier(r, current_age)
            raw_change  = (base + mod) * 0.9    # midpoint of uniform(0.4, 1.4)
            lo, hi      = _change_limits(r, current_age)
            change      = max(lo, min(hi, raw_change))
            projected[r] = max(0.0, min(100.0, projected[r] + change))

    return projected


def project_ovr(ratings: dict) -> float:
    """Compute the ZenGM OVR from a ratings dict (reuses formulas.player_ovr)."""
    return float(player_ovr(ratings))


def project_contract_status(
    contract_exp: int,
    current_season: int,
    target_season: int,
) -> bool:
    """
    True if the player is still under contract in *target_season*.

    A contract expiring in season X covers through the end of that season,
    so the player is under contract if contract_exp >= target_season.
    """
    return contract_exp >= target_season


def estimate_next_contract(
    projected_ovr: float,
    age: float,
    is_my_team: bool = False,
    ovr_mean: float = 65.0,
    ovr_std: float = 10.0,
    max_contract: float = 50_000.0,
    min_contract: float = 1_200.0,
) -> float:
    """
    Estimate the annual salary (in $K) a player will demand when their contract
    expires, using ZenGM's genContract formula (free-agency reference §2.1):

        factor = 2 × 1.7 = 3.4   (soft-cap × basketball multiplier)
        amount = ((p.value / 100) - 0.47) × 3.4 × (maxContract - minContract)
                 + minContract

    p.value is approximated from projected_ovr via the standard normalization
    (§ 3 of the trade-AI reference).

    Parameters
    ----------
    projected_ovr : OVR at the time of free agency.
    age           : player age at the time of free agency.
    is_my_team    : apply the ~0.85× re-signing loyalty discount (§4.3).
    ovr_mean, ovr_std : league-level normalization constants (defaults: 65, 10).
    """
    from src.core.ai_trade_value import _value_combine_ovr_pot

    # Normalize ovr to the ZenGM value scale (mean ~47, std ~10)
    ovr_norm = (projected_ovr - ovr_mean) / ovr_std * 10.0 + 47.0
    # Assume pot ≈ ovr at the time of re-signing (player is near or past peak)
    player_value = _value_combine_ovr_pot(age, ovr_norm, ovr_norm)

    factor = 2.0 * 1.7   # 3.4
    amount = ((player_value / 100.0) - 0.47) * factor * (max_contract - min_contract) + min_contract
    amount = max(min_contract, min(max_contract, amount))

    if is_my_team:
        amount *= 0.85   # loyalty bonus reduces asking price by ~15%

    return amount
