"""
Layer 3 — Team Composites and Synergy.

Computes the three synergy multipliers (off, def, reb) for a five-player lineup.
These are added to team composite ratings scaled by synergyFactor (default 0.1):
    team_composite += 0.1 * synergy[key]

Source: zengm-games/zengm — index.ts → updateSynergy()
"""

from __future__ import annotations

import math

from .formulas import PlayerRatings, composite_rating

# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------


def sigmoid(x: float, a: float, b: float) -> float:
    """
    Logistic sigmoid used throughout the synergy model.

        sigmoid(x, a, b) = 1 / (1 + exp(-a * (x - b)))

    Parameters
    ----------
    x : input value
    a : steepness (15 for per-player skill curves, lower for count-level terms)
    b : inflection point (cutoff)
    """
    return 1.0 / (1.0 + math.exp(-a * (x - b)))


# ---------------------------------------------------------------------------
# Step 1 — Skill counts
# ---------------------------------------------------------------------------

# a=15 is hardcoded for all per-player skill-detection sigmoids.
# At this steepness each player contributes ≈0 or ≈1 to each skill count.
_SKILL_A: float = 15.0

# Maps skill tag → (composite_name, cutoff_b)
_SKILL_PARAMS: dict[str, tuple[str, float]] = {
    "3":  ("shootingThreePointer", 0.59),
    "A":  ("athleticism",          0.63),
    "B":  ("dribbling",            0.68),
    "Di": ("defenseInterior",      0.57),
    "Dp": ("defensePerimeter",     0.61),
    "Po": ("shootingLowPost",      0.61),
    "Ps": ("passing",              0.63),
    "R":  ("rebounding",           0.61),
}


def skills_count(players: list[PlayerRatings]) -> dict[str, float]:
    """
    Step 1: For each of the 8 skill tags, sum sigmoid(composite, 15, cutoff)
    across all on-court players.  Each count is in [0, 5].

    Parameters
    ----------
    players : list of player rating dicts/Series (base ratings 0–100).
    """
    counts: dict[str, float] = {tag: 0.0 for tag in _SKILL_PARAMS}
    for player in players:
        for tag, (composite, cutoff) in _SKILL_PARAMS.items():
            c = composite_rating(player, composite)
            counts[tag] += sigmoid(c, _SKILL_A, cutoff)
    return counts


# ---------------------------------------------------------------------------
# Step 2 — Offensive synergy
# ---------------------------------------------------------------------------


def _offensive_synergy(counts: dict[str, float]) -> float:
    """
    Step 2: Offensive synergy in [0, 1].

    The /17 denominator is the theoretical maximum of the raw sum (each sigmoid
    term approaching 1 at count=5).  A perimFactor then multiplies the result,
    rewarding lineups with shooters + passers + ball-handlers and punishing
    non-perimeter lineups (perimFactor=0 → off halved; perimFactor=1 → no penalty).
    """
    c3  = counts["3"]
    cB  = counts["B"]
    cPs = counts["Ps"]
    cPo = counts["Po"]
    cA  = counts["A"]

    off  = 5.0 * sigmoid(c3,  3,  2.00)                        # shooting
    off += 3.0 * sigmoid(cB,  15, 0.75) + sigmoid(cB,  5, 1.75)          # ball handling
    off += (3.0 * sigmoid(cPs, 15, 0.75)
            + sigmoid(cPs, 5, 1.75)
            + sigmoid(cPs, 5, 2.75))                            # passing
    off += sigmoid(cPo, 15, 0.75)                               # post play
    off += sigmoid(cA,  15, 1.75) + sigmoid(cA,  5, 2.75)      # athleticism
    off /= 17.0

    perim_raw = math.sqrt(1.0 + cB + cPs + c3) - 1.0
    perim_factor = max(0.0, min(2.0, perim_raw)) / 2.0
    off *= 0.5 + 0.5 * perim_factor

    return off


# ---------------------------------------------------------------------------
# Step 3 — Defensive synergy
# ---------------------------------------------------------------------------


def _defensive_synergy(counts: dict[str, float]) -> float:
    """
    Step 3: Defensive synergy in [0, 1].

    Interior defense is weighted 2× vs. perimeter; athleticism provides a
    smaller bonus.  Normalized by /6 (theoretical max of the raw sum).
    """
    cDp = counts["Dp"]
    cDi = counts["Di"]
    cA  = counts["A"]

    d  = sigmoid(cDp, 15, 0.75)                                 # perimeter D
    d += 2.0 * sigmoid(cDi, 15, 0.75)                           # interior D (2×)
    d += sigmoid(cA,  5,  2.00) + sigmoid(cA,  5,  3.25)        # athleticism
    d /= 6.0

    return d


# ---------------------------------------------------------------------------
# Step 4 — Rebounding synergy
# ---------------------------------------------------------------------------


def _rebounding_synergy(counts: dict[str, float]) -> float:
    """
    Step 4: Rebounding synergy in [0, 0.5].

    Two sigmoid terms (steep + shallow) capture diminishing returns from
    stacking rebounders.  Normalized by /4.
    """
    cR = counts["R"]
    r  = sigmoid(cR, 15, 0.75) + sigmoid(cR, 5, 1.75)
    r /= 4.0
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_lineup_synergy(players: list[PlayerRatings]) -> dict[str, float]:
    """
    Compute the three synergy multipliers for a five-player on-court lineup.

    Parameters
    ----------
    players : list of exactly 5 player rating dicts or pd.Series.
              Each entry must contain base ratings (keys matching BASE_RATINGS).
              Missing ratings default to 50.

    Returns
    -------
    dict with keys 'off', 'def', 'reb' — each a float.
    Applied in the engine as:  team_composite += synergyFactor * synergy[key]
    """
    if len(players) != 5:
        raise ValueError(f"Exactly 5 players required, got {len(players)}")

    counts = skills_count(players)
    return {
        "off": _offensive_synergy(counts),
        "def": _defensive_synergy(counts),
        "reb": _rebounding_synergy(counts),
    }
