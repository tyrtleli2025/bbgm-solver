"""
Trade Market Scanner — Phase 3.

A trade is considered in two independent stages:

  PLAUSIBLE  — the opposing AI team's value-change (dv) is strictly positive,
               the salary-matching constraint holds, and no untradable players
               are involved.  This uses ZenGM's full ValueChangeCalculator
               (ai_trade_value.evaluate_dv) as the acceptance gate.

  DESIRABLE  — the trade raises MY optimal lineup score J, computed via the
               fast _PlayerCache / _fast_optimize path (unchanged from before).

Only trades that pass BOTH gates are returned.  There is no symmetric delta-
window (ASSET_FLOOR / MAX_AI_LOSS) — the AI acceptance formula is the sole
realism filter.

For each (team, target player) pair the scanner assembles the minimal package
of my tradeable assets that satisfies dv > 0.  Package sizes supported: 1-for-1
and 2-for-1 (we send 2, we receive 1).

Performance notes
-----------------
• Per-player sigmoid contributions (_PlayerCache) are precomputed once, making
  each C(n,5) inner loop iteration ~40× faster than calling composite_rating.
• evaluate_dv is cheap (a few dozen Python ops per player pair) so the dv check
  runs before the more expensive _fast_optimize call.
• The old lineup score is precomputed once and reused for every trade.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .formulas import composite_rating, player_ovr
from .optimizer import OFF_WEIGHT, DEF_WEIGHT
from .synergy import _SKILL_A, _SKILL_PARAMS, sigmoid
from .trade_engine import SALARY_SCALE
from .ai_trade_value import (
    evaluate_dv as _evaluate_dv,
    league_value_stats as _league_value_stats,
    infer_strategy as _infer_strategy,
    salary_match_ok as _salary_match_ok,
    is_untradable as _is_untradable,
    SALARY_CAP_DEFAULT as _SALARY_CAP,
    SOFT_CAP_MATCH_PCT as _SOFT_CAP_MATCH_PCT,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

TOP_N_DEFAULT: int = 5
"""Maximum number of results returned by find_best_trades."""

# ---------------------------------------------------------------------------
# Per-player cache  (unchanged — fast lineup scorer)
# ---------------------------------------------------------------------------

_SKILL_TAGS: tuple[str, ...] = tuple(_SKILL_PARAMS.keys())

_I3  = _SKILL_TAGS.index("3")
_IA  = _SKILL_TAGS.index("A")
_IB  = _SKILL_TAGS.index("B")
_IDi = _SKILL_TAGS.index("Di")
_IDp = _SKILL_TAGS.index("Dp")
_IPo = _SKILL_TAGS.index("Po")
_IPs = _SKILL_TAGS.index("Ps")


class _PlayerCache:
    """Precomputed per-player values for fast lineup scoring."""
    __slots__ = ("ovr", "sigs")

    def __init__(self, ovr: int, sigs: np.ndarray) -> None:
        self.ovr  = ovr
        self.sigs = sigs


def _build_cache(player) -> _PlayerCache:
    return _PlayerCache(
        ovr=player_ovr(player),
        sigs=np.array([
            sigmoid(composite_rating(player, comp), _SKILL_A, cutoff)
            for comp, cutoff in _SKILL_PARAMS.values()
        ], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Fast lineup scorer  (unchanged)
# ---------------------------------------------------------------------------


def _fast_score(c0: _PlayerCache, c1: _PlayerCache, c2: _PlayerCache,
                c3: _PlayerCache, c4: _PlayerCache) -> float:
    ovr_sum = c0.ovr + c1.ovr + c2.ovr + c3.ovr + c4.ovr
    s = c0.sigs + c1.sigs + c2.sigs + c3.sigs + c4.sigs

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

    sDp = float(s[_IDp])
    sDi = float(s[_IDi])
    def_  = sigmoid(sDp, 15, 0.75)
    def_ += 2.0 * sigmoid(sDi, 15, 0.75)
    def_ += sigmoid(sA, 5, 2.00) + sigmoid(sA, 5, 3.25)
    def_ /= 6.0

    return ovr_sum + OFF_WEIGHT * off + DEF_WEIGHT * def_


def _fast_optimize(caches: list[_PlayerCache]) -> float:
    """Best lineup score over all C(n, 5) combinations."""
    best = -math.inf
    for combo in itertools.combinations(caches, 5):
        score = _fast_score(*combo)
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Full-roster objective J  (engine_reference.md §  Team Overall Rating)
# ---------------------------------------------------------------------------

# Regular-season exponential-decay parameters (engine_reference.md)
_TEAM_A: float = 0.3334
_TEAM_B: float = -0.1609

# Precomputed weights w_i = a × e^(b × i) for rotation ranks 0-9.
# Rank 0 (best player) gets the highest weight.
_TEAM_WEIGHTS: tuple[float, ...] = tuple(
    _TEAM_A * math.exp(_TEAM_B * i) for i in range(10)
)


def _compute_team_j(caches: list[_PlayerCache]) -> float:
    """
    Full team objective J (lineup synergy + depth bonus).

    J = _fast_optimize(top-5 synergy lineup score)
      + Σ_{i=5}^{min(9, n-1)}  w_i × sorted_OVR_i

    The depth term uses the regular-season exponential-decay weights from
    engine_reference.md (a=0.3334, b=-0.1609) for rotation ranks 6-10.
    This makes J sensitive to depth: trading away a bench player has a cost
    even when the starting lineup is unchanged, so the search explores
    diverse packages rather than fixating on the single best top-5 trade.

    The weights for ranks 6-10 are ≈ 0.149, 0.127, 0.108, 0.092, 0.079,
    so a typical bench player (OVR 65) contributes ≈ 5-10 points to J.
    """
    lineup_score = _fast_optimize(caches)

    ovrs = sorted((c.ovr for c in caches), reverse=True)
    depth_bonus = sum(
        _TEAM_WEIGHTS[i] * ovrs[i]
        for i in range(5, min(10, len(ovrs)))
    )
    return lineup_score + depth_bonus


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_salary(player, salary_scale: float) -> float:
    """Extract raw salary from a player dict or pd.Series."""
    raw = player.get("salary") if isinstance(player, dict) else player.get("salary")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return 0.0
    return float(raw)


def _total_salary(roster_df: pd.DataFrame) -> float:
    total = 0.0
    for i in range(len(roster_df)):
        raw = roster_df.iloc[i].get("salary")
        if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
            total += float(raw)
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _auto_v_context(
    my_roster_df: pd.DataFrame,
    league_rosters_dict: dict[str, pd.DataFrame],
    current_season: int,
    salary_cap: float,
):
    """
    Build a LeagueVContext from roster DataFrames when use_v_function=True
    but no pre-built v_context is provided.

    Uses contract_exp=current_season+3 for all players (DataFrames don't
    carry contract expiry).  This is sufficient to capture the age trajectory
    signal — young players project as improving, old players as declining.
    """
    from src.value import LeagueState, LeagueVContext  # lazy import avoids circularity

    players: list[dict] = []

    my_tid: int = 0
    if "tid" in my_roster_df.columns and len(my_roster_df) > 0:
        raw = my_roster_df.iloc[0].get("tid")
        if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
            my_tid = int(raw)

    for i in range(len(my_roster_df)):
        row = my_roster_df.iloc[i].to_dict()
        row.setdefault("contract_exp", current_season + 3)
        players.append(row)

    for _, df in league_rosters_dict.items():
        if df is None or len(df) == 0:
            continue
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            row.setdefault("contract_exp", current_season + 3)
            players.append(row)

    all_tids = sorted({
        int(p["tid"]) for p in players
        if p.get("tid") is not None
        and not (isinstance(p["tid"], float) and math.isnan(p["tid"]))
    })
    teams = [{"tid": t} for t in all_tids]

    ls = LeagueState(
        players=players,
        teams=teams,
        picks=[],
        current_season=current_season,
        salary_cap=salary_cap,
        my_tid=my_tid,
        num_teams=max(30, len(teams)),
    )
    return LeagueVContext(ls)


def find_best_trades(
    my_roster_df: pd.DataFrame,
    league_rosters_dict: dict[str, pd.DataFrame],
    league: Optional[dict] = None,
    salary_cap: float = _SALARY_CAP,
    salary_scale: float = SALARY_SCALE,
    top_n: int = TOP_N_DEFAULT,
    progress: Optional[Callable[[str, int, int], None]] = None,
    v_context=None,        # Optional[LeagueVContext] — pre-built context
    use_v_function: bool = False,  # auto-build LeagueVContext when no v_context supplied
) -> list[dict]:
    """
    Scan the league for the best available trades for my team.

    Filters
    -------
    1. Untradable players are excluded (gamesUntilTradable > 0 or
       expired contract during offseason).
    2. Salary match: the other team's post-trade salary must satisfy the
       soft-cap 125 % rule (or hard-cap, if configured).
    3. AI acceptance: evaluate_dv(other_team, ...) > 0 using the other team's
       inferred strategy and ZenGM's full ValueChangeCalculator formula.
    4. Lineup improvement: my optimal lineup score J must increase (> 0).

    Parameters
    ----------
    my_roster_df        : my team's roster DataFrame.
    league_rosters_dict : {team_name: roster_df} for every opponent.
    league              : League dict from ai_trade_value.league_value_stats().
                          If None, computed from all rosters automatically.
    salary_cap          : cap in raw salary units (default $90 M in $K).
    salary_scale        : passed through for salary context (default 1 000).
    top_n               : maximum results to return (default 5).
    progress            : optional callable(team_name, n_done, n_total).
    v_context           : pre-built LeagueVContext; takes priority over use_v_function.
    use_v_function      : when True and v_context is None, auto-build a LeagueVContext
                          from the supplied DataFrames and use ΔV as the improvement gate.

    Returns
    -------
    list of up to top_n dicts sorted by net_lineup_score descending:
        'team'             : str   — trade-partner key
        'trade_type'       : str   — '1-for-1' or '2-for-1'
        'incoming'         : list  — player dicts arriving on my roster
        'outgoing'         : list  — player dicts leaving my roster
        'net_lineup_score' : float — improvement to my optimal lineup score (ΔV or ΔJ)
        'dv'               : float — the AI's acceptance margin (> 0)
        'new_score'        : float — my new optimal lineup score
    """
    if len(my_roster_df) < 5:
        raise ValueError(
            f"My roster has only {len(my_roster_df)} players; at least 5 required."
        )

    # Build league stats if not supplied
    if league is None:
        all_rosters = {"__mine__": my_roster_df, **league_rosters_dict}
        league = _league_value_stats(
            all_rosters,
            salary_cap=salary_cap,
            current_season=0,
            is_offseason=False,
        )

    current_season   = int(league.get("current_season", 0))
    is_offseason     = bool(league.get("is_offseason", False))
    sal_cap_type     = str(league.get("salary_cap_type", "soft"))
    sal_match_pct    = float(league.get("soft_cap_trade_match", _SOFT_CAP_MATCH_PCT))
    n_mine = len(my_roster_df)

    # Auto-build LeagueVContext when use_v_function=True and none was supplied.
    if use_v_function and v_context is None:
        v_context = _auto_v_context(
            my_roster_df, league_rosters_dict, current_season, salary_cap
        )

    # --- One-time precomputation for my roster --------------------------------
    my_caches = [_build_cache(my_roster_df.iloc[i]) for i in range(n_mine)]
    # When v_context is provided, use V-based scoring (ΔV) instead of ΔJ.
    # J is still computed here for the case v_context is None.
    old_score = _compute_team_j(my_caches)   # full-roster J (lineup + depth)

    # Identify my tradeable players up front
    my_tradeable = [
        i for i in range(n_mine)
        if not _is_untradable(my_roster_df.iloc[i], current_season, is_offseason)
    ]
    my_total_salary = _total_salary(my_roster_df)

    passing: list[dict] = []
    n_total = len(league_rosters_dict)
    n_done  = 0

    for team_name, their_df in league_rosters_dict.items():
        if their_df is None or len(their_df) < 1:
            n_done += 1
            if progress:
                progress(team_name, n_done, n_total)
            continue

        n_theirs = len(their_df)
        their_strategy    = _infer_strategy(their_df)
        their_caches      = [_build_cache(their_df.iloc[j]) for j in range(n_theirs)]
        their_total_sal   = _total_salary(their_df)

        their_tradeable = [
            j for j in range(n_theirs)
            if not _is_untradable(their_df.iloc[j], current_season, is_offseason)
        ]

        # ── 1-for-1 ────────────────────────────────────────────────────────
        for j in their_tradeable:
            their_p   = their_df.iloc[j]
            their_sal = _get_salary(their_p, salary_scale)

            for i in my_tradeable:
                my_p   = my_roster_df.iloc[i]
                my_sal = _get_salary(my_p, salary_scale)

                # Gate 1a: salary match — THEIR team absorbing my player
                if not _salary_match_ok(
                    their_sal, my_sal, their_total_sal, salary_cap,
                    sal_cap_type, sal_match_pct
                ):
                    continue

                # Gate 1b: salary match — MY team absorbing their player
                if not _salary_match_ok(
                    my_sal, their_sal, my_total_salary, salary_cap,
                    sal_cap_type, sal_match_pct
                ):
                    continue

                # Gate 2: AI acceptance (their dv from their perspective)
                dv = _evaluate_dv(
                    their_df, league,
                    incoming=[their_p],   # what they give up
                    outgoing=[my_p],      # what they receive
                    strategy=their_strategy,
                )
                if dv <= 0:
                    continue

                # Gate 3: improvement in the objective (J or V)
                if v_context is not None:
                    their_d = their_p.to_dict()
                    my_d    = my_p.to_dict()
                    net_score = v_context.delta_v(
                        add_players=[their_d], remove_players=[my_d]
                    )
                    new_score = v_context.v_current + net_score
                else:
                    post_caches = (
                        [c for k, c in enumerate(my_caches) if k != i]
                        + [their_caches[j]]
                    )
                    new_score  = _compute_team_j(post_caches)
                    net_score  = new_score - old_score
                    their_d    = their_p.to_dict()
                    my_d       = my_p.to_dict()

                if net_score <= 0:
                    continue

                passing.append({
                    "team":             team_name,
                    "trade_type":       "1-for-1",
                    "incoming":         [their_d],
                    "outgoing":         [my_d],
                    "net_lineup_score": net_score,
                    "dv":               dv,
                    "new_score":        new_score,
                })

        # ── 2-for-1  (we send 2, receive 1) ───────────────────────────────
        if n_mine >= 6 and len(my_tradeable) >= 2:
            for j in their_tradeable:
                their_p   = their_df.iloc[j]
                their_sal = _get_salary(their_p, salary_scale)

                for i1, i2 in itertools.combinations(my_tradeable, 2):
                    my_p1  = my_roster_df.iloc[i1]
                    my_p2  = my_roster_df.iloc[i2]
                    my_sal = _get_salary(my_p1, salary_scale) + _get_salary(my_p2, salary_scale)

                    # Gate 1a: their salary match
                    if not _salary_match_ok(
                        their_sal, my_sal, their_total_sal, salary_cap
                    ):
                        continue

                    # Gate 1b: my salary match
                    if not _salary_match_ok(
                        my_sal, their_sal, my_total_salary, salary_cap
                    ):
                        continue

                    dv = _evaluate_dv(
                        their_df, league,
                        incoming=[their_p],
                        outgoing=[my_p1, my_p2],
                        strategy=their_strategy,
                    )
                    if dv <= 0:
                        continue

                    drop = {i1, i2}
                    their_d = their_p.to_dict()
                    my_d1   = my_p1.to_dict()
                    my_d2   = my_p2.to_dict()

                    if v_context is not None:
                        net_score = v_context.delta_v(
                            add_players=[their_d],
                            remove_players=[my_d1, my_d2],
                        )
                        new_score = v_context.v_current + net_score
                    else:
                        post_caches = (
                            [c for k, c in enumerate(my_caches) if k not in drop]
                            + [their_caches[j]]
                        )
                        if len(post_caches) < 5:
                            continue
                        new_score = _compute_team_j(post_caches)
                        net_score = new_score - old_score

                    if net_score <= 0:
                        continue

                    passing.append({
                        "team":             team_name,
                        "trade_type":       "2-for-1",
                        "incoming":         [their_d],
                        "outgoing":         [my_d1, my_d2],
                        "net_lineup_score": net_score,
                        "dv":               dv,
                        "new_score":        new_score,
                    })

        n_done += 1
        if progress:
            progress(team_name, n_done, n_total)

    passing.sort(key=lambda t: t["net_lineup_score"], reverse=True)
    return passing[:top_n]
