"""
Depth-limited beam search over sequences of AI-accepted trades.

Architecture
------------
node  = _BeamNode(roster, league_dict, sequence, J, sig)
edge  = one AI-accepted trade produced by find_best_trades
value = J(roster) = _fast_optimize(precomputed caches)

Search:
  1. Expand top-K candidate trades per beam node (via find_best_trades).
  2. Apply each trade: update my roster, lock acquired players, update
     opponent's roster so the same team cannot be traded with twice in
     sequence without reflecting the previous deal.
  3. Prune: keep top-beam_width nodes by J; skip already-visited roster states.
  4. Collect every reachable node as a potential SearchResult.
  5. Return top-N sequences by j_final.

gamesUntilTradable constraint:
  Players acquired in a prior step are stamped gamesUntilTradable=GAMES_LOCKOUT.
  is_untradable() inside find_best_trades naturally excludes them from the
  next step's outgoing pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .formulas import BASE_RATINGS
from .market_scanner import find_best_trades, _fast_optimize, _build_cache, _compute_team_j
from .ai_trade_value import league_value_stats, player_base_value, SALARY_CAP_DEFAULT
from .trade_engine import SALARY_SCALE

# ---------------------------------------------------------------------------
# Tuning constants (exposed so tests can reference them)
# ---------------------------------------------------------------------------

BEAM_WIDTH_DEFAULT: int = 5
"""Candidate trades expanded per node per depth step (also the beam size)."""

DEPTH_DEFAULT: int = 3
"""Maximum number of sequential trades in a returned sequence."""

TOP_N_DEFAULT: int = 5
"""Maximum number of SearchResult objects returned."""

GAMES_LOCKOUT: int = 82
"""gamesUntilTradable stamped on newly acquired players to prevent immediate re-trade."""

MIN_STEP_J_GAIN: float = 1.0
"""Minimum improvement in J required for a single trade step to be kept.
Filters out near-zero moves that appear favourable only due to floating-point
noise in the lineup scorer."""

MIN_TOTAL_J_GAIN: float = 1.0
"""Minimum total J improvement across an entire sequence for it to be returned."""


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class TradeStep:
    """One accepted trade edge in a sequence."""
    team:       str
    trade_type: str
    incoming:   list   # list[dict] — player dicts arriving on my roster
    outgoing:   list   # list[dict] — player dicts leaving my roster
    dv:         float  # AI acceptance margin (> 0)
    j_before:   float  # lineup score immediately before this trade
    j_after:    float  # lineup score immediately after this trade


@dataclass
class SearchResult:
    """A complete sequence of trades with J values at every step."""
    sequence: list[TradeStep]
    j_start:  float   # J of the initial roster
    j_final:  float   # J of the final roster after all trades

    @property
    def j_trajectory(self) -> list[float]:
        """[j_start, after_step_1, after_step_2, …]"""
        return [self.j_start] + [s.j_after for s in self.sequence]

    @property
    def total_j_gain(self) -> float:
        return self.j_final - self.j_start


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_j(roster_df: pd.DataFrame) -> float:
    """
    Full team objective J = lineup synergy (top-5) + depth bonus (ranks 6-10).
    Uses _compute_team_j from market_scanner so the search objective matches
    the trade scanner's gate objective exactly.
    """
    caches = [_build_cache(roster_df.iloc[i]) for i in range(len(roster_df))]
    return _compute_team_j(caches)


def _player_key(player) -> object:
    """
    Stable, hashable identity for a player.  Prefers 'pid'; falls back to a
    tuple of base-rating values (handles test players without pids).
    """
    pid = player.get("pid") if isinstance(player, dict) else player.get("pid")
    if pid is not None and not (isinstance(pid, float) and math.isnan(float(pid))):
        return ("pid", int(float(pid)))
    getter = player.get if isinstance(player, dict) else player.get
    return ("r", tuple(int(float(getter(r) or 50)) for r in BASE_RATINGS))


def _roster_signature(roster_df: pd.DataFrame) -> frozenset:
    """Canonical fingerprint for a roster — used to detect revisited states."""
    return frozenset(
        _player_key(roster_df.iloc[i].to_dict()) for i in range(len(roster_df))
    )


def _resolve_outgoing_indices(
    roster_df: pd.DataFrame,
    outgoing_dicts: list[dict],
) -> list[int]:
    """
    Map each outgoing player dict (from find_best_trades) back to its row
    index in roster_df.  Matches by pid first, then by base-rating tuple.
    """
    indices: list[int] = []
    for out_d in outgoing_dicts:
        target = _player_key(out_d)
        for idx in range(len(roster_df)):
            if _player_key(roster_df.iloc[idx].to_dict()) == target:
                indices.append(idx)
                break
    return indices


def _apply_trade_to_my_roster(
    roster_df: pd.DataFrame,
    out_indices: list[int],
    incoming_dicts: list[dict],
    lockout_games: int,
) -> pd.DataFrame:
    """
    Return a new roster with outgoing rows removed and incoming players appended,
    stamped with gamesUntilTradable=lockout_games so they cannot be immediately
    re-traded in subsequent search steps.
    """
    drop = set(out_indices)
    kept = [roster_df.iloc[i] for i in range(len(roster_df)) if i not in drop]
    kept_df = pd.DataFrame(kept).reset_index(drop=True) if kept else pd.DataFrame()

    if not incoming_dicts:
        return kept_df

    locked_rows = []
    for p in incoming_dicts:
        row = dict(p)
        row["gamesUntilTradable"] = lockout_games
        locked_rows.append(row)

    return pd.concat([kept_df, pd.DataFrame(locked_rows)], ignore_index=True)


def _apply_trade_to_league(
    league_dict: dict[str, pd.DataFrame],
    trade: dict,
) -> dict[str, pd.DataFrame]:
    """
    Return an updated league_rosters_dict reflecting a completed trade:
    the opponent team loses the players they gave us and gains ours.
    This prevents the search from re-proposing the same player across steps.
    """
    team = trade["team"]
    if team not in league_dict:
        return league_dict

    their_df = league_dict[team]
    incoming_keys = {_player_key(p) for p in trade["incoming"]}

    # Remove what they traded away (convert to dicts so concat with gained rows works)
    their_kept = [
        their_df.iloc[j].to_dict()
        for j in range(len(their_df))
        if _player_key(their_df.iloc[j].to_dict()) not in incoming_keys
    ]

    # Add what they received from us
    their_gained = [dict(p) for p in trade["outgoing"]]

    new_their = (
        pd.DataFrame(their_kept + their_gained).reset_index(drop=True)
        if (their_kept or their_gained) else pd.DataFrame()
    )

    new_dict = dict(league_dict)
    new_dict[team] = new_their
    return new_dict


def _step_roster_value_gain(
    incoming: list[dict],
    outgoing: list[dict],
    league: dict,
) -> float:
    """
    Net change in age-adjusted roster value (sum of player_base_value) for one
    trade step.  Positive → we gain value; negative → we lose value.

    player_base_value blends current OVR with potential weighted by age, so
    trading a young high-pot player for an older equal-OVR player is penalised
    even if the immediate lineup score is unchanged.
    """
    in_val  = sum(player_base_value(p, league) for p in incoming)
    out_val = sum(player_base_value(p, league) for p in outgoing)
    return in_val - out_val


# ---------------------------------------------------------------------------
# Beam node (internal)
# ---------------------------------------------------------------------------


class _BeamNode:
    __slots__ = ("roster_df", "league_dict", "sequence", "j_score", "sig")

    def __init__(
        self,
        roster_df:   pd.DataFrame,
        league_dict: dict[str, pd.DataFrame],
        sequence:    list[TradeStep],
        j_score:     float,
        sig:         frozenset,
    ) -> None:
        self.roster_df   = roster_df
        self.league_dict = league_dict
        self.sequence    = sequence
        self.j_score     = j_score
        self.sig         = sig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def beam_search(
    initial_roster_df:    pd.DataFrame,
    league_rosters_dict:  dict[str, pd.DataFrame],
    league:               Optional[dict] = None,
    depth:                int   = DEPTH_DEFAULT,
    beam_width:           int   = BEAM_WIDTH_DEFAULT,
    top_n:                int   = TOP_N_DEFAULT,
    salary_cap:           float = SALARY_CAP_DEFAULT,
    salary_scale:         float = SALARY_SCALE,
    lockout_games:        int   = GAMES_LOCKOUT,
    min_step_j_gain:      float = MIN_STEP_J_GAIN,
    min_total_j_gain:     float = MIN_TOTAL_J_GAIN,
) -> list[SearchResult]:
    """
    Depth-limited beam search over trade sequences.

    At each depth step:
      • Each beam node generates up to beam_width candidate trades via
        find_best_trades (all gates: dv > 0, salary match, untradable).
      • Each candidate is applied: my roster and the opponent's roster are
        updated; acquired players are locked for subsequent steps.
      • Beam is pruned to the top-beam_width nodes by J; previously visited
        roster states are skipped.
      • Every newly reached node is recorded as a potential result.

    Parameters
    ----------
    initial_roster_df   : starting roster DataFrame.
    league_rosters_dict : {team_name: roster_df} for every opponent.
    league              : League dict from league_value_stats(); auto-computed if None.
    depth               : maximum trade-sequence length (default 3).
    beam_width          : trades expanded per node AND beam size (default 5).
    top_n               : number of results to return (default 5).
    salary_cap          : cap in raw salary units (default $90 M in $K).
    salary_scale        : salary divisor passed through (default 1 000).
    lockout_games       : gamesUntilTradable for newly acquired players (default 82).

    Returns
    -------
    Up to top_n SearchResult objects, sorted by j_final descending.
    Each sequence has strictly non-decreasing J (every edge has net_lineup_score > 0).
    """
    if len(initial_roster_df) < 5:
        raise ValueError(
            f"Initial roster has only {len(initial_roster_df)} players; need ≥ 5."
        )

    if league is None:
        all_r = {"__mine__": initial_roster_df, **league_rosters_dict}
        league = league_value_stats(all_r, salary_cap=salary_cap)

    j_start  = _compute_j(initial_roster_df)
    init_sig = _roster_signature(initial_roster_df)

    beam: list[_BeamNode] = [
        _BeamNode(
            roster_df=initial_roster_df,
            league_dict=dict(league_rosters_dict),
            sequence=[],
            j_score=j_start,
            sig=init_sig,
        )
    ]
    visited: set[frozenset] = {init_sig}
    all_results: list[SearchResult] = []

    for _depth_step in range(depth):
        next_beam: list[_BeamNode] = []

        for node in beam:
            candidates = find_best_trades(
                node.roster_df,
                node.league_dict,
                league=league,
                salary_cap=salary_cap,
                salary_scale=salary_scale,
                top_n=beam_width,
            )

            for trade in candidates:
                out_indices = _resolve_outgoing_indices(
                    node.roster_df, trade["outgoing"]
                )

                new_roster = _apply_trade_to_my_roster(
                    node.roster_df, out_indices, trade["incoming"], lockout_games
                )
                if len(new_roster) < 5:
                    continue

                new_sig = _roster_signature(new_roster)
                if new_sig in visited:
                    continue
                visited.add(new_sig)

                new_j = _compute_j(new_roster)

                # Guardrail A: require a meaningful per-step J gain
                if new_j - node.j_score < min_step_j_gain:
                    continue

                # Guardrail B: reject steps that decrease age-adjusted roster value.
                # player_base_value blends OVR with potential weighted by age, so
                # trading a young high-pot player for an older same-OVR player
                # produces a negative gain and is blocked here.
                rv_gain = _step_roster_value_gain(
                    trade["incoming"], trade["outgoing"], league
                )
                if rv_gain < 0:
                    continue

                step = TradeStep(
                    team=trade["team"],
                    trade_type=trade["trade_type"],
                    incoming=trade["incoming"],
                    outgoing=trade["outgoing"],
                    dv=trade["dv"],
                    j_before=node.j_score,
                    j_after=new_j,
                )
                new_seq  = node.sequence + [step]
                new_ld   = _apply_trade_to_league(node.league_dict, trade)

                new_node = _BeamNode(
                    roster_df=new_roster,
                    league_dict=new_ld,
                    sequence=new_seq,
                    j_score=new_j,
                    sig=new_sig,
                )
                next_beam.append(new_node)

                all_results.append(SearchResult(
                    sequence=new_seq,
                    j_start=j_start,
                    j_final=new_j,
                ))

        if not next_beam:
            break

        # Diversity-preserving beam selection.
        # Group nodes by the identity of the players they acquired in their
        # most recent trade step.  Keep only the highest-J node per group,
        # then take the top-beam_width across groups.
        # This ensures the beam explores different acquisitions rather than
        # filling with near-identical rosters that all received the same star.
        _groups: dict[frozenset, _BeamNode] = {}
        for node in next_beam:
            if node.sequence:
                last_step = node.sequence[-1]
                group_key = frozenset(
                    _player_key(p) for p in last_step.incoming
                )
            else:
                group_key = frozenset()
            prev = _groups.get(group_key)
            if prev is None or node.j_score > prev.j_score:
                _groups[group_key] = node

        diverse = sorted(_groups.values(), key=lambda n: n.j_score, reverse=True)
        beam = diverse[:beam_width]

    # Drop sequences whose total gain is below the minimum threshold
    all_results = [r for r in all_results if r.total_j_gain >= min_total_j_gain]
    all_results.sort(key=lambda r: r.j_final, reverse=True)
    return all_results[:top_n]
