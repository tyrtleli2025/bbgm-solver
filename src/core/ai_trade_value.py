"""
ZenGM ValueChangeCalculator — determines whether the CPU accepts a trade.

Source:
  § 1  trade/propose.ts                 — acceptance rule
  § 2  team/ValueChangeCalculator.ts    — evaluate() → dv formula
  § 3  player/value.ts,                 — player_base_value()
       player/valueCombineOvrPot.ts     — age table
  § 4  ValueChangeCalculator.ts         — zscore()
  § 5  ValueChangeCalculator.ts         — sumValues() EXPONENT nonlinearity
  § 6  sumValues()                      — strategy (age) multipliers
  § 7  getContractValue() + sumValues() — contract adjustments
  § 8  getPlayers() / sumValues()       — injury, negative-value dampening
  § 9  getPickInfo() etc.               — pick valuation (TODO stub)
  §10  summary.ts + isUntradable.ts     — hard salary / tradability constraints
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import pandas as pd

from .formulas import player_ovr

# ---------------------------------------------------------------------------
# Public constants (tests reference these directly)
# ---------------------------------------------------------------------------

EXPONENT: int = 7
"""Nonlinearity exponent for basketball: v^7 for stars (v > 1)."""

MIN_VALUE: float = -0.5
"""Floor z-score used in the contract expected-salary formula (basketball)."""

MAX_VALUE: float = 2.0
"""Ceiling z-score used in contract calculations (basketball)."""

CONTRACT_SLOPE: float = 0.13
"""expectedSalary = CONTRACT_SLOPE × (v − MIN_VALUE)   (fraction of cap)."""

SALARY_CAP_DEFAULT: float = 90_000.0
"""Default BBGM salary cap in $K ($90 M)."""

SOFT_CAP_MATCH_PCT: float = 1.25
"""Soft-cap rule: incoming salary ≤ 125 % of outgoing when over the cap."""

# These are the implicit centre and spread of the OVR-normalised scale produced
# by player_base_value().  The normalisation formula maps an average player to
# exactly 47 and one OVR-std above average to 57, so the target spread is 10.
# ZenGM calls these "playerOvrMean / playerOvrStd" and does not compute them
# from a second pass over player.value — they are fixed by the formula.
_VALUE_CENTRE: float = 47.0
_VALUE_SCALE: float  = 10.0

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PlayerRatings = Union[dict, pd.Series]

# League dict keys:
#   ovr_mean, ovr_std   — for normalising OVR/pot (§ 3)
#   value_mean, value_std — for z-scoring player.value (§ 4)
#   salary_cap, current_season, is_offseason
League = dict

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(p: PlayerRatings, key: str, default=None):
    """Safe .get() for both plain dicts and pd.Series rows."""
    val = p.get(key) if isinstance(p, dict) else p.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def _value_combine_ovr_pot(age: float, current: float, pot: float) -> float:
    """
    Blend current performance and potential by age (valueCombineOvrPot.ts, § 3).

    Young players are weighted toward potential; 28+ rely solely on current
    with a small age-decay factor.
    """
    a = int(age)
    if a <= 19:         return 0.700 * pot + 0.300 * current
    if a == 20:         return 0.650 * pot + 0.350 * current
    if a in (21, 22):   return 0.600 * pot + 0.400 * current
    if a == 23:         return 0.550 * pot + 0.450 * current
    if a == 24:         return 0.450 * pot + 0.550 * current
    if a == 25:         return 0.300 * pot + 0.700 * current
    if a == 26:         return 0.150 * pot + 0.850 * current
    if a == 27:         return 0.025 * pot + 0.975 * current
    if a == 28:         return 0.95 * current
    if a == 29:         return 0.94 * current
    if a == 30:         return 0.93 * current
    if 31 <= a <= 33:   return 0.92 * current
    if 34 <= a <= 38:   return 0.91 * current
    return 0.90 * current   # 39+


def _strategy_multiplier(age: float, strategy: str) -> float:
    """
    Per-player age multiplier applied to v before the exponent (§ 6).

    Rebuilding teams boost youth and penalise veterans.
    Contending teams discount youth and are neutral on prime/veteran players.
    Ages not listed in a strategy's table receive multiplier 1.0 (neutral).
    """
    a = int(age)
    if strategy == "rebuilding":
        if a <= 19:   return 1.0750
        if a == 20:   return 1.0500
        if a == 21:   return 1.0375
        if a == 22:   return 1.0250
        if a == 23:   return 1.0125
        # ages 24-26 → neutral
        if a == 27:   return 0.9750
        if a == 28:   return 0.9500
        if a >= 29:   return 0.9000
        return 1.0
    if strategy == "contending":
        if a <= 19:   return 0.8000
        if a == 20:   return 0.8250
        if a == 21:   return 0.8500
        if a == 22:   return 0.8750
        if a == 23:   return 0.9250
        if a == 24:   return 0.9500
        # ages 25+ → neutral
        return 1.0
    return 1.0  # unknown strategy → neutral


def _injury_factor(games_remaining: int) -> float:
    """
    Discount factor for an injured incoming player (§ 8, includeInjuries path).

    > 75 games remaining: v *= 0.25 (severe injury).
    Otherwise: v *= (1 - gamesRemaining/100).
    """
    if games_remaining <= 0:
        return 1.0
    if games_remaining > 75:
        return 0.25
    return 1.0 - games_remaining / 100.0


# ---------------------------------------------------------------------------
# § 3  Player base value
# ---------------------------------------------------------------------------


def player_base_value(player: PlayerRatings, league: League) -> float:
    """
    Compute a player's base value on the ~0–100 BBGM scale (§ 3).

    Steps
    -----
    1. Normalise OVR and pot to a standard league scale
       (mean ≈ 47, std ≈ 10):
           ovr_norm = (ovr − ovr_mean) / ovr_std × 10 + 47
    2. Blend in recent PER when available:
           current = 0.8 × ovr_norm + 0.2 × (31.693 + 1.531 × PER)
       Falls back to ovr_norm when no PER data is present.
    3. Combine current with normalised potential by age (valueCombineOvrPot).

    Parameters
    ----------
    player  : player rating dict or pd.Series row.  Expected keys: base
              ratings, 'age', 'pot' (optional), 'per' (optional PER stat).
    league  : League dict; must contain 'ovr_mean' and 'ovr_std'.
    """
    ovr = float(player_ovr(player))
    age = float(_get(player, "age") or 27.0)
    pot = float(_get(player, "pot") or ovr)

    ovr_mean = float(league["ovr_mean"])
    ovr_std  = float(league["ovr_std"])

    # Step 1: normalise
    def _norm(x: float) -> float:
        return (x - ovr_mean) / ovr_std * 10.0 + 47.0

    ovr_norm = _norm(ovr)
    pot_norm = _norm(pot)

    # Step 2: blend in PER
    per = _get(player, "per")
    if per is not None:
        current = 0.8 * ovr_norm + 0.2 * (31.693 + 1.531 * float(per))
    else:
        current = ovr_norm

    # Step 3: combine by age
    return _value_combine_ovr_pot(age, current, pot_norm)


# ---------------------------------------------------------------------------
# § 2  League statistics (two-pass to avoid circularity)
# ---------------------------------------------------------------------------


def league_value_stats(
    all_rosters: dict[str, pd.DataFrame],
    current_season: int = 0,
    is_offseason: bool = False,
    salary_cap: float = SALARY_CAP_DEFAULT,
    salary_cap_type: str = "soft",
    soft_cap_trade_match: float = SOFT_CAP_MATCH_PCT,
) -> League:
    """
    Compute league-wide OVR statistics needed for z-scoring.

    ZenGM does not compute a separate second pass over player.value — it
    z-scores player.value directly against the OVR distribution (§ 4):
        v = (player.value − ovr_mean) / ovr_std

    Parameters
    ----------
    all_rosters          : {team_name: roster_df} for every team.
    current_season       : season number (for contract-expiry detection).
    is_offseason         : whether it is the offseason (affects expiry).
    salary_cap           : cap in raw salary-column units (default $90 M/$K).
    salary_cap_type      : "soft", "hard", or "none" (default "soft").
    soft_cap_trade_match : fraction for 125 % soft-cap rule (default 1.25).

    Returns
    -------
    League dict with keys: ovr_mean, ovr_std,
    salary_cap, salary_cap_type, soft_cap_trade_match,
    current_season, is_offseason.
    """
    all_players = [
        df.iloc[i]
        for df in all_rosters.values()
        for i in range(len(df))
    ]

    if not all_players:
        return dict(
            ovr_mean=47.0, ovr_std=10.0,
            salary_cap=salary_cap,
            salary_cap_type=salary_cap_type,
            soft_cap_trade_match=soft_cap_trade_match,
            current_season=current_season,
            is_offseason=is_offseason,
        )

    ovrs     = np.array([player_ovr(p) for p in all_players], dtype=float)
    ovr_mean = float(ovrs.mean())
    ovr_std  = float(max(ovrs.std(), 1.0))

    return dict(
        ovr_mean=ovr_mean, ovr_std=ovr_std,
        salary_cap=salary_cap,
        salary_cap_type=salary_cap_type,
        soft_cap_trade_match=soft_cap_trade_match,
        current_season=current_season,
        is_offseason=is_offseason,
    )


# ---------------------------------------------------------------------------
# § 4  Z-score
# ---------------------------------------------------------------------------


def zscore(value: float, mean: float, std: float) -> float:
    """Z-score a player's base value against league statistics (§ 4)."""
    return (value - mean) / std


# ---------------------------------------------------------------------------
# § 6  Strategy inference
# ---------------------------------------------------------------------------


def infer_strategy(roster: pd.DataFrame) -> str:
    """
    Infer 'contending' or 'rebuilding' from a roster (§ 6).

    Heuristic: 'contending' when the top-3 OVR average ≥ 75 AND the roster
    mean age ≥ 25 (win-now talent on a prime-age team).  Can be overridden
    by passing strategy= explicitly to evaluate_dv().
    """
    if len(roster) == 0:
        return "rebuilding"

    ovrs  = sorted([player_ovr(roster.iloc[i]) for i in range(len(roster))],
                   reverse=True)
    ages  = [float(_get(roster.iloc[i], "age") or 27.0) for i in range(len(roster))]

    top_ovr  = float(np.mean(ovrs[: min(3, len(ovrs))]))
    mean_age = float(np.mean(ages))

    if top_ovr >= 75 and mean_age >= 25.0:
        return "contending"
    return "rebuilding"


# ---------------------------------------------------------------------------
# § 7  Contract value
# ---------------------------------------------------------------------------


def contract_value(
    player: PlayerRatings,
    normalized_value: float,
    league: League,
) -> float:
    """
    Contract value adjustment for a player (§ 7, getContractValue).

    Returns 0.0 for expiring contracts (this season, or next if offseason).
    Positive → player is underpaid (bonus ≤ 0.1).
    Negative → player is overpaid (uncapped penalty).

    Parameters
    ----------
    player          : must contain 'salary' ($K) and 'contract_exp' (year).
    normalized_value: the player's RAW z-score (pre-difficulty, pre-strategy)
                      used to estimate what salary the market would offer.
                      sum_values() passes raw_v here, not the adjusted value.
    league          : supplies salary_cap, current_season, is_offseason.
    """
    salary_cap     = float(league["salary_cap"])
    current_season = int(league.get("current_season", 0))
    is_offseason   = bool(league.get("is_offseason", False))

    # Expiration year: look for contract_exp or exp; default = next season
    exp = int(
        _get(player, "contract_exp")
        or _get(player, "exp")
        or (current_season + 1)
    )
    amount = float(_get(player, "salary") or 0.0)

    # Expiring: this season or (next season when it's currently the offseason)
    expiring_cutoff = current_season + (1 if is_offseason else 0)
    if exp <= expiring_cutoff:
        return 0.0

    expected = CONTRACT_SLOPE * (normalized_value - MIN_VALUE)
    actual   = amount / salary_cap if salary_cap > 0 else 0.0

    return min(expected - actual, 0.1)   # bonus capped; penalty uncapped


# ---------------------------------------------------------------------------
# § 5 / §6 / §7 / §8  Sum values (heart of the trade math)
# ---------------------------------------------------------------------------


def sum_values(
    assets: list[PlayerRatings],
    league: League,
    strategy: str,
    include_injuries: bool = False,
    difficulty: float = 0.0,
) -> float:
    """
    Sum the EXPONENT-nonlinear trade values of a list of player assets (§ 5).

    Pipeline per asset
    ------------------
    1. Base value → z-score  (§ 3–4).
    2. Strategy age multiplier  (§ 6).
    3. Contract value: v += contractsFactor × contract_value()  (§ 7).
       contractsFactor = 2.0 for rebuilding, 0.5 for contending.
    4. Injury discount if include_injuries=True  (§ 8).
    5. Difficulty multiplier (only pass nonzero for AI's outgoing side).
    6. Negative-value dampening: v < 0  →  v / 20  (§ 8).
    7. Star exponent: v > 1  →  v^EXPONENT  (§ 5).

    Parameters
    ----------
    assets          : list of player dicts or pd.Series.
    league          : from league_value_stats().
    strategy        : 'contending' or 'rebuilding' for the evaluating team.
    include_injuries: True for the AI's incoming side (what it receives).
    difficulty      : game difficulty scalar; pass for AI's outgoing side only.
                      Positive = harder (AI values its players more).
    """
    contracts_factor = 2.0 if strategy == "rebuilding" else 0.5
    total = 0.0

    for player in assets:
        age = float(_get(player, "age") or 27.0)

        # 1. Base z-score (§ 4).
        #    player_base_value() normalises OVR to mean ≈ 47, std ≈ 10 by design.
        #    ZenGM z-scores directly against those fixed constants (playerOvrMean/Std)
        #    rather than computing a second-pass mean/std from player.value.
        #    raw_v is preserved for contract_value() — that formula must use the
        #    pre-adjustment z-score (§ 7).
        base  = player_base_value(player, league)
        raw_v = zscore(base, _VALUE_CENTRE, _VALUE_SCALE)
        v     = raw_v

        # 2. Difficulty fudge — applied only to positive values, BEFORE strategy
        #    (§ 8: "1 + 0.1 × difficulty applied to outgoing player values").
        if difficulty != 0.0 and v > 0:
            v *= 1.0 + 0.1 * difficulty

        # 3. Strategy age multiplier (§ 6)
        v *= _strategy_multiplier(age, strategy)

        # 4. Injury discount — incoming AI assets only (§ 8)
        if include_injuries:
            games = int(_get(player, "injured_games_remaining") or 0)
            v *= _injury_factor(games)

        # 5. Negative-value dampening (§ 8) — applied BEFORE contract addition
        if v < 0:
            v /= 20.0

        # 6. Contract value — computed from the RAW z-score (§ 7).
        #    contractsFactor = 2 rebuilding / 0.5 contending.
        cv  = contract_value(player, raw_v, league)
        v  += contracts_factor * cv

        # 7. Just-drafted floor: rookies can be cut, so value ≥ 0 (§ 8)
        if _get(player, "just_drafted"):
            v = max(0.0, v)

        # 8. Star exponent (§ 5)
        if v > 1:
            v = v ** EXPONENT

        total += v

    return total


def sum_values_debug(
    assets: list[PlayerRatings],
    league: League,
    strategy: str,
    include_injuries: bool = False,
    difficulty: float = 0.0,
) -> tuple[float, list[dict]]:
    """
    Like sum_values but returns detailed breakdown for each asset.
    Returns (total, [breakdowns]) where each breakdown contains:
      name, base_value, raw_v, strategy_mult, injury_mult, contract_adj,
      v_before_exp, final_v
    """
    contracts_factor = 2.0 if strategy == "rebuilding" else 0.5
    total = 0.0
    breakdowns = []

    for player in assets:
        name = str(_get(player, "name") or f"pid_{_get(player, 'pid')}")
        age = float(_get(player, "age") or 27.0)

        # 1. Base z-score
        base = player_base_value(player, league)
        raw_v = zscore(base, _VALUE_CENTRE, _VALUE_SCALE)
        v = raw_v

        # 2. Difficulty fudge
        difficulty_mult = 1.0
        if difficulty != 0.0 and v > 0:
            difficulty_mult = 1.0 + 0.1 * difficulty
            v *= difficulty_mult

        # 3. Strategy age multiplier
        strategy_mult = _strategy_multiplier(age, strategy)
        v *= strategy_mult

        # 4. Injury discount
        injury_mult = 1.0
        if include_injuries:
            games = int(_get(player, "injured_games_remaining") or 0)
            injury_mult = _injury_factor(games)
            v *= injury_mult

        # 5. Negative-value dampening
        if v < 0:
            v /= 20.0

        # 6. Contract value
        cv = contract_value(player, raw_v, league)
        contract_adj = contracts_factor * cv
        v += contract_adj

        # 7. Just-drafted floor
        if _get(player, "just_drafted"):
            v = max(0.0, v)

        # 8. Star exponent
        v_before_exp = v
        if v > 1:
            v = v ** EXPONENT

        total += v

        breakdowns.append({
            "name": name,
            "age": age,
            "base_value": base,
            "raw_v": raw_v,
            "difficulty_mult": difficulty_mult,
            "strategy_mult": strategy_mult,
            "injury_mult": injury_mult,
            "contract_value": cv,
            "contract_adj": contract_adj,
            "v_before_exp": v_before_exp,
            "final_v": v,
            "exponent_applied": v != v_before_exp,
        })

    return total, breakdowns


# ---------------------------------------------------------------------------
# § 2  Evaluate dv (from the other team's perspective)
# ---------------------------------------------------------------------------


def evaluate_dv(
    other_team_roster: pd.DataFrame,
    league: League,
    incoming: list[PlayerRatings],
    outgoing: list[PlayerRatings],
    strategy: Optional[str] = None,
    difficulty: float = 0.0,
) -> float:
    """
    Compute the value change for the opposing AI team if this trade occurs (§ 2).

    dv > 0  →  AI's value increases  →  trade is realistic (AI will accept).
    dv ≤ 0  →  AI's value decreases  →  AI rejects.

    Parameters
    ----------
    other_team_roster : the AI team's current roster (for strategy inference).
    league            : from league_value_stats().
    incoming          : players WE receive  (= what the AI gives up, assetsRemoved).
    outgoing          : players WE send     (= what the AI receives, assetsAdded).
    strategy          : override AI strategy; inferred from roster if None.
    difficulty        : game difficulty applied to AI's outgoing side (§ 8).
    """
    if strategy is None:
        strategy = infer_strategy(other_team_roster)

    # From the AI's perspective:
    #   assetsAdded   = what we send it (they receive)
    #   assetsRemoved = what it gives us (they lose)
    dv_added   = sum_values(list(outgoing), league, strategy,
                            include_injuries=True,  difficulty=0.0)
    dv_removed = sum_values(list(incoming), league, strategy,
                            include_injuries=False, difficulty=difficulty)

    return dv_added - dv_removed


# ---------------------------------------------------------------------------
# § 1  Acceptance rule
# ---------------------------------------------------------------------------


def ai_accepts(dv: float) -> tuple[bool, str]:
    """
    Apply the AI acceptance rule and return (accepted, message) (§ 1).

    The AI accepts iff dv > 0.  Rejection messages expose the margin.
    """
    if dv > 0:
        return True,  "Accepted."
    if dv > -2:
        return False, "Close, but not quite good enough."
    if dv > -5:
        return False, "That's not a good deal for me."
    return False, "What, are you crazy?!"


# ---------------------------------------------------------------------------
# § 10  Hard constraints
# ---------------------------------------------------------------------------


def salary_match_ok(
    outgoing_salary: float,
    incoming_salary: float,
    our_total_salary: float,
    salary_cap: float,
    salary_cap_type: str = "soft",
    soft_cap_match_pct: float = SOFT_CAP_MATCH_PCT,
) -> bool:
    """
    Check whether the salary-matching constraint passes (§ 10).

    Over-cap status is determined by the **post-trade** payroll (ZenGM rule):
        new_payroll = our_total − outgoing + incoming

    hard — new_payroll must not exceed the cap.
    soft — if new_payroll ≤ cap: allowed;
            elif outgoing ≤ 0: blocked (absorbing salary while going over cap);
            else: incoming ≤ outgoing × soft_cap_match_pct.
    none — no constraint.
    """
    if salary_cap_type == "none":
        return True

    new_payroll = our_total_salary - outgoing_salary + incoming_salary

    if salary_cap_type == "hard":
        return new_payroll <= salary_cap

    # soft cap — use post-trade payroll to determine over-cap status
    if new_payroll <= salary_cap:
        return True   # trade keeps/brings us under the cap
    if outgoing_salary <= 0:
        return False  # absorbing salary while pushing over cap with nothing outgoing
    return incoming_salary <= outgoing_salary * soft_cap_match_pct


def is_untradable(
    player: PlayerRatings,
    current_season: int,
    is_offseason: bool = False,
) -> bool:
    """
    True if a player cannot be traded right now (§ 10, isUntradable.ts).

    Blocks:
    - Expired contract during the post-playoffs / pre-free-agency window.
    - Recently signed / acquired player (gamesUntilTradable > 0).
    """
    if is_offseason:
        exp = int(
            _get(player, "contract_exp")
            or _get(player, "exp")
            or (current_season + 1)
        )
        if exp <= current_season:
            return True

    games_until_tradable = int(_get(player, "gamesUntilTradable") or 0)
    return games_until_tradable > 0


# ---------------------------------------------------------------------------
# § 9  Draft pick stub
# ---------------------------------------------------------------------------

# TODO: Implement pick valuation (§ 9 of bbgm_trade_ai_reference.md).
#
# Required steps when picks are present in the loaded data:
#   1. Estimate pick slot from projected win% (estWinPct formula).
#   2. Regress future picks toward uncertainty target.
#   3. Apply user-pick penalty / AI-pick bonus (user picks pushed later).
#   4. Convert slot → value via the estValues table, then z-score.
#   5. Apply anti-fleece multiplier for 3+ first-rounders.
#
# Until implemented: pass picks as empty lists; they are treated as 0 value.
