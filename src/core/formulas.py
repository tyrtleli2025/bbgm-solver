"""
BBGM engine formulas: Layer 1 (base ratings), Layer 2 (composite ratings),
and the player OVR regression with piecewise fudge factor.

Source: zengm-games/zengm — constants.basketball.ts, compositeRating.ts,
        ovr.basketball.ts
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Layer 1 — Base ratings
# ---------------------------------------------------------------------------

BASE_RATINGS: list[str] = [
    "hgt", "stre", "spd", "jmp", "endu",
    "ins", "dnk", "ft",  "fg",  "tp",
    "oiq", "diq", "drb", "pss", "reb",
]

# Default value when a rating is missing from a player vector.
# Using 50 (midpoint of 0–100) is neutral for all composites.
_DEFAULT_RATING: float = 50.0

# ---------------------------------------------------------------------------
# Layer 2 — Composite weight table
# ---------------------------------------------------------------------------
# Keys starting with "_c<value>" denote literal constant inputs (not ratings).
# e.g. "_c50" means a constant value of 50 (used as a baseline floor).
# Negative weights are intentional (see shootingMidRange, turnovers, fouling).

COMPOSITE_WEIGHTS: dict[str, dict[str, float]] = {
    # ---- Offensive --------------------------------------------------------
    "usage": {
        "ins": 1.5, "dnk": 1.0, "fg": 1.0, "tp": 1.0,
        "spd": 0.5, "hgt": 0.5, "drb": 0.5, "oiq": 0.5,
    },
    "dribbling": {
        "drb": 1.0, "spd": 1.0,
    },
    "passing": {
        "drb": 0.4, "pss": 1.0, "oiq": 0.5,
    },
    "shootingAtRim": {
        "hgt": 2.0, "stre": 0.3, "dnk": 0.3, "oiq": 0.2,
    },
    "shootingLowPost": {
        "hgt": 1.0, "stre": 0.6, "spd": 0.2, "ins": 1.0, "oiq": 0.4,
    },
    # oiq has a negative weight: higher IQ → fewer ill-advised mid-range attempts
    "shootingMidRange": {
        "oiq": -0.5, "fg": 1.0, "stre": 0.2,
    },
    "shootingThreePointer": {
        "oiq": 0.1, "tp": 1.0,
    },
    "shootingFT": {
        "ft": 1.0,
    },
    "drawingFouls": {
        "hgt": 1.0, "spd": 1.0, "drb": 1.0, "dnk": 1.0, "oiq": 1.0,
    },
    # _c50: literal constant 50 (baseline floor); oiq < 0 → smarter players turn it over less
    "turnovers": {
        "_c50": 0.5, "ins": 1.0, "pss": 1.0, "oiq": -1.0,
    },

    # ---- Defensive --------------------------------------------------------
    "defense": {
        "hgt": 1.0, "stre": 1.0, "spd": 1.0, "jmp": 0.5, "diq": 2.0,
    },
    "defenseInterior": {
        "hgt": 2.5, "stre": 1.0, "spd": 0.5, "jmp": 0.5, "diq": 2.0,
    },
    "defensePerimeter": {
        "hgt": 0.5, "stre": 0.5, "spd": 2.0, "jmp": 0.5, "diq": 1.0,
    },
    "blocking": {
        "hgt": 2.5, "jmp": 1.5, "diq": 0.5,
    },
    # _c50: constant baseline; diq < 0 → smarter defenders steal more (composite = frequency of getting beat)
    "stealing": {
        "_c50": 1.0, "spd": 1.0, "diq": 2.0,
    },
    # _c50: constant baseline; diq/spd < 0 → disciplined, quick players foul less
    "fouling": {
        "_c50": 3.0, "hgt": 1.0, "diq": -1.0, "spd": -1.0,
    },

    # ---- Other ------------------------------------------------------------
    "rebounding": {
        "hgt": 2.0, "stre": 0.1, "jmp": 0.1, "reb": 2.0, "oiq": 0.5, "diq": 0.5,
    },
    "pace": {
        "spd": 1.0, "jmp": 1.0, "dnk": 1.0, "tp": 1.0, "drb": 1.0, "pss": 1.0,
    },
    # _c50: constant floor so endurance is always ≥ 0.25 even at endu=0
    "endurance": {
        "_c50": 1.0, "endu": 1.0,
    },
    "athleticism": {
        "stre": 1.0, "spd": 1.0, "jmp": 1.0, "hgt": 0.75,
    },
    "jumpBall": {
        "hgt": 1.0, "jmp": 0.25,
    },
}

# ---------------------------------------------------------------------------
# Layer 2 — Composite rating computation
# ---------------------------------------------------------------------------

PlayerRatings = Union[dict, pd.Series]


def _get(ratings: PlayerRatings, key: str) -> float:
    """Return a base rating, substituting _DEFAULT_RATING for NaN / missing."""
    val = ratings.get(key) if isinstance(ratings, dict) else ratings.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return _DEFAULT_RATING
    return float(val)


def composite_rating(ratings: PlayerRatings, composite: str) -> float:
    """
    Compute one composite rating for a player.

    Formula (from compositeRating.ts):
        c = clamp( Σ(w_i × x_i) / (100 × Σ w_i), 0, 1 )

    Constants encoded as "_c<value>" keys (e.g. "_c50" → 50.0) are literal
    numeric inputs, not player ratings.  Negative weights are valid.
    """
    weights = COMPOSITE_WEIGHTS[composite]
    numerator = 0.0
    weight_sum = 0.0

    for key, w in weights.items():
        if key.startswith("_c"):
            value = float(key[2:])        # e.g. "_c50" → 50.0
        else:
            value = _get(ratings, key)
        numerator += w * value
        weight_sum += w

    if weight_sum == 0.0:
        return 0.0

    return float(np.clip(numerator / (100.0 * weight_sum), 0.0, 1.0))


def all_composites(ratings: PlayerRatings) -> dict[str, float]:
    """Return every composite rating for a player as a dict keyed by composite name."""
    return {name: composite_rating(ratings, name) for name in COMPOSITE_WEIGHTS}


# ---------------------------------------------------------------------------
# Layer 2 → Player OVR
# ---------------------------------------------------------------------------

# Linear regression coefficients and centering means (ovr.basketball.ts)
_OVR_INTERCEPT: float = 48.5

_OVR_TERMS: list[tuple[str, float, float]] = [
    # (rating_key, coefficient, centering_mean)
    ("hgt",  0.159,   47.5),
    ("diq",  0.159,   46.7),
    ("oiq",  0.133,   46.8),
    ("spd",  0.123,   50.8),
    ("stre", 0.0777,  50.2),
    ("tp",   0.0726,  47.1),
    ("endu", 0.0632,  39.9),
    ("pss",  0.062,   51.3),
    ("drb",  0.059,   54.8),
    ("jmp",  0.051,   48.7),
    ("dnk",  0.0286,  49.5),
    ("ft",   0.0202,  47.0),
    ("ins",  0.0126,  42.4),
    ("fg",   0.01,    47.0),
    ("reb",  0.01,    51.4),
]


def _ovr_fudge(r: float) -> float:
    """
    Piecewise fudge factor applied to the raw regression score to maintain
    the historical 0–100 OVR scale (ovr.basketball.ts).
    """
    if r >= 68:
        return 8.0
    if r >= 50:
        return 4.0 + (r - 50.0) * (4.0 / 18.0)
    if r >= 42:
        return -5.0 + (r - 42.0) * (9.0 / 8.0)
    if r >= 31:
        return -5.0 - (42.0 - r) * (5.0 / 11.0)
    return -10.0


def player_ovr(ratings: PlayerRatings) -> int:
    """
    Compute the BBGM player Overall rating (0–100).

    Missing ratings default to their centering mean so they contribute
    zero to the regression (neutral effect).
    """
    r = _OVR_INTERCEPT
    for key, coeff, mean in _OVR_TERMS:
        value = ratings.get(key) if isinstance(ratings, dict) else ratings.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = mean          # missing → zero contribution
        r += coeff * (float(value) - mean)

    ovr = round(r + _ovr_fudge(r))
    return int(np.clip(ovr, 0, 100))


# ---------------------------------------------------------------------------
# Batch helpers for DataFrames
# ---------------------------------------------------------------------------

def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column for every composite rating to *df* (in-place-style copy).
    Input columns must match the BASE_RATINGS keys.  Missing columns are
    treated as _DEFAULT_RATING (50).
    """
    df = df.copy()
    for name in COMPOSITE_WEIGHTS:
        df[name] = df.apply(lambda row: composite_rating(row, name), axis=1)
    return df


def add_ovr(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'ovr' column to *df* computed from base ratings."""
    df = df.copy()
    df["ovr"] = df.apply(player_ovr, axis=1)
    return df
