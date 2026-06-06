"""
Unified offseason decision optimizer.

Treats re-signings and trades as a single action space, all scored by ΔV,
all drawing on the same cap pool. Uses a greedy search: at each step apply
the highest-ΔV feasible action, recompute V for the updated roster, repeat
until no action improves V.

Action types
------------
- "resign"  : re-sign an expiring player at estimated demand
- "trade"   : execute a trade found by find_best_trades()

Players with expiring contracts that are never assigned a "resign" action
are collected in the "let_walk" output.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from src.core.formulas import BASE_RATINGS, player_ovr
from src.core.market_scanner import _auto_v_context, find_best_trades
from src.core.resigning import (
    MIN_CONTRACT,
    REPLACEMENT_OVR,
    _player_dict_from_row,
    _replacement_player,
)
from src.project import project_ratings, project_ovr, estimate_next_contract

log = logging.getLogger(__name__)

# How many trade candidates to seed the action pool with
_TRADE_POOL_SIZE = 20

# Only apply actions with ΔV above this floor (avoids noise)
_MIN_DV = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pid_of(p: dict) -> Optional[int]:
    raw = p.get("pid")
    if raw is None:
        return None
    try:
        v = int(float(raw))
        return None if math.isnan(float(raw)) else v
    except (ValueError, TypeError):
        return None


def _apply_action_to_players(
    current: list[dict],
    remove_pids: set[int],
    add_players: list[dict],
) -> list[dict]:
    """Return a new player list with removals and additions applied."""
    kept = [p for p in current if _pid_of(p) not in remove_pids]
    return kept + add_players


def _current_salary(players: list[dict]) -> float:
    return sum(float(p.get("salary", 0) or 0) for p in players)


def _build_resign_action(
    row: pd.Series,
    league: dict,
    v_context,
    my_tid: int,
) -> Optional[dict]:
    """
    Build a resign action dict for one expiring player.
    Returns None if the player has no valid pid.
    """
    current_season = int(league.get("current_season", 0))
    ovr_mean = float(league.get("ovr_mean", 65.0))
    ovr_std = float(league.get("ovr_std", 10.0))
    salary_cap = float(league.get("salary_cap", 90_000.0))

    name = str(row.get("name") or f"pid_{row.get('pid')}")
    age = float(row.get("age") or 27.0)
    pid = _pid_of(row.to_dict() if hasattr(row, "to_dict") else row)
    current_ovr = float(player_ovr(row))

    current_ratings = {r: float(row.get(r, 50.0)) for r in BASE_RATINGS}
    proj_ratings = project_ratings(current_ratings, age, years_forward=1)
    proj_ovr = project_ovr(proj_ratings)

    demand = max(MIN_CONTRACT, estimate_next_contract(
        projected_ovr=proj_ovr,
        age=age + 1.0,
        is_my_team=True,
        ovr_mean=ovr_mean,
        ovr_std=ovr_std,
        max_contract=salary_cap * 0.35,
        min_contract=MIN_CONTRACT,
    ))

    player_d = _player_dict_from_row(row, current_season)
    resigned = dict(player_d)
    resigned["salary"] = demand
    resigned["contract_exp"] = current_season + 3

    replacement = _replacement_player(my_tid, current_season)

    # ΔV of resigning relative to letting walk
    dv_resign = v_context.delta_v(add_players=[resigned], remove_players=[player_d])
    dv_walk = v_context.delta_v(add_players=[replacement], remove_players=[player_d])
    delta_v = dv_resign - dv_walk

    return dict(
        type="resign",
        name=name,
        age=int(age),
        current_ovr=round(current_ovr, 1),
        projected_ovr=round(proj_ovr, 1),
        demand_k=round(demand),
        delta_v=delta_v,
        # Internal fields (prefixed _) used during search
        _remove_pids={pid} if pid is not None else set(),
        _add_players=[resigned],
        # For resign actions ΔV is computed vs the walk scenario (replacement fills slot),
        # not vs the baseline state (which V already treats as "player re-signed").
        _walk_players=[replacement],   # used by _recompute_dv for resign actions
        _salary_delta=demand - float(row.get("salary", 0) or 0),
        _expiring_pid=pid,
    )


def _build_trade_action(trade: dict, v_context) -> dict:
    """
    Build a trade action dict from a find_best_trades result.
    Resolves incoming players through the pid map for full contract info.
    """
    incoming_full = []
    for p in trade.get("incoming", []):
        pid = _pid_of(p)
        full = v_context._pid_map.get(pid) if pid is not None else None
        if full is not None:
            d = dict(full)
        else:
            d = dict(p)
            d.setdefault("contract_exp", v_context.ls.current_season + 3)
        d["tid"] = v_context.my_tid
        incoming_full.append(d)

    outgoing_pids = {
        _pid_of(p)
        for p in trade.get("outgoing", [])
        if _pid_of(p) is not None
    }

    in_names  = [p.get("name") or f"pid_{p.get('pid')}" for p in trade.get("incoming", [])]
    out_names = [p.get("name") or f"pid_{p.get('pid')}" for p in trade.get("outgoing", [])]
    description = (
        f"Get {', '.join(in_names)} from {trade['team']}, "
        f"send {', '.join(out_names)}"
    )

    salary_in  = sum(float(p.get("salary", 0) or 0) for p in trade.get("incoming", []))
    salary_out = sum(float(p.get("salary", 0) or 0) for p in trade.get("outgoing", []))

    return dict(
        type="trade",
        team=trade["team"],
        description=description,
        dv_ai=trade.get("dv", 0.0),
        delta_v=trade.get("net_lineup_score", 0.0),  # initial estimate; recomputed in search
        incoming=[dict(p) for p in trade.get("incoming", [])],
        outgoing=[dict(p) for p in trade.get("outgoing", [])],
        demand_k=None,
        _remove_pids=outgoing_pids,
        _add_players=incoming_full,
        _salary_delta=salary_in - salary_out,
        _expiring_pid=None,
    )


def _recompute_dv(action: dict, current_players: list[dict], v_context) -> float:
    """
    Compute ΔV of applying *action* on top of *current_players*.

    For trade actions: ΔV = V(roster with incoming) − V(current roster).

    For resign actions: ΔV = V(resigned player) − V(walk + replacement).
    This comparison is correct because the V function already assumes valuable
    players get re-signed when their contract expires, so the naive
    "vs current state" yields ≈ 0. The meaningful signal is resign-vs-walk.
    """
    remove_pids = action["_remove_pids"]
    add_players = action["_add_players"]
    new_players = _apply_action_to_players(current_players, remove_pids, add_players)
    v_after = v_context._compute_my_v(new_players)

    if "_walk_players" in action:
        # Resign action: compare against the let-walk scenario
        walk_players = _apply_action_to_players(
            current_players, remove_pids, action["_walk_players"]
        )
        v_before = v_context._compute_my_v(walk_players)
    else:
        v_before = v_context._compute_my_v(current_players)

    return v_after - v_before


def _is_feasible(
    action: dict,
    current_player_pids: set[int],
    consumed_pids: set[int],
) -> bool:
    """
    Return True only when all four feasibility guards pass.

    Guard 1 — don't send a player already committed (resigned or traded away).
    Guard 2 — don't act on an expiring player whose disposition is already set.
    Guard 3 — don't send a player no longer on the current roster.
    Guard 4 — don't acquire a player already on the current roster.
              (For resign actions the same pid is both removed and added,
               so we check net-new pids: add_pids minus remove_pids.)
    """
    remove_pids   = action["_remove_pids"]
    expiring_pid  = action.get("_expiring_pid")
    add_pids      = {
        _pid_of(p)
        for p in action["_add_players"]
        if _pid_of(p) is not None
    }

    # Guard 1: no outgoing player is already committed
    if remove_pids & consumed_pids:
        return False

    # Guard 2: expiring player not already acted on
    if expiring_pid is not None and expiring_pid in consumed_pids:
        return False

    # Guard 3: all outgoing players are actually on my roster right now
    if not remove_pids.issubset(current_player_pids):
        return False

    # Guard 4: no net-new incoming player is already on my roster
    genuinely_new = add_pids - remove_pids
    if genuinely_new & current_player_pids:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_decisions(
    my_roster_df: pd.DataFrame,
    league_rosters_dict: dict[str, pd.DataFrame],
    league: dict,
    cap_info: dict,
) -> dict:
    """
    Unified offseason decision optimizer.

    Generates all candidate resign and trade actions, then applies a greedy
    search: at each step pick the highest-ΔV feasible action, update the
    roster state, and continue until no action improves V.

    Parameters
    ----------
    my_roster_df        : current roster DataFrame.
    league_rosters_dict : opponent rosters.
    league              : League dict from league_value_stats().
    cap_info            : dict from parse_league_data() (salary_cap etc.).

    Returns
    -------
    dict with keys:
      actions        — applied actions in order, each with type/description/delta_v
      let_walk       — expiring players not in applied actions
      projected_v    — V after all applied actions
      initial_v      — V before any actions
      remaining_cap_k — estimated remaining cap space after all actions
    """
    current_season = int(league.get("current_season", 0))
    salary_cap = float(cap_info.get("salary_cap", 90_000.0))

    # --- Build LeagueVContext ---
    v_context = _auto_v_context(
        my_roster_df, league_rosters_dict, current_season, salary_cap
    )
    my_tid = v_context.my_tid
    initial_v = float(v_context.v_current)

    # --- Generate resign actions (one per expiring player) ---
    resign_actions: list[dict] = []
    expiring_pids: set[int] = set()

    for i in range(len(my_roster_df)):
        row = my_roster_df.iloc[i]
        contract_exp = int(row.get("contract_exp") or 0)
        if contract_exp != current_season:
            continue
        pid = _pid_of(row.to_dict())
        if pid is not None:
            expiring_pids.add(pid)
        action = _build_resign_action(row, league, v_context, my_tid)
        if action is not None:
            resign_actions.append(action)

    # --- Generate trade actions ---
    raw_trades = find_best_trades(
        my_roster_df,
        league_rosters_dict,
        league=league,
        salary_cap=salary_cap,
        current_season=current_season,
        v_context=v_context,
        use_v_function=True,
        top_n=_TRADE_POOL_SIZE,
    )
    trade_actions = [_build_trade_action(t, v_context) for t in raw_trades]

    action_pool: list[dict] = resign_actions + trade_actions

    log.info(
        "optimize_decisions: %d resign + %d trade candidates",
        len(resign_actions), len(trade_actions),
    )

    # --- Greedy search ---
    current_players: list[dict] = list(v_context._my_players)
    current_player_pids: set[int] = {
        _pid_of(p) for p in current_players if _pid_of(p) is not None
    }
    current_v = initial_v
    consumed_pids: set[int] = set()
    applied: list[dict] = []

    while action_pool:
        best_idx   = None
        best_dv    = _MIN_DV
        best_players:     Optional[list[dict]] = None
        best_player_pids: Optional[set[int]]   = None

        for idx, action in enumerate(action_pool):
            if not _is_feasible(action, current_player_pids, consumed_pids):
                continue

            new_players = _apply_action_to_players(
                current_players, action["_remove_pids"], action["_add_players"]
            )

            # Roster size guard
            if len(new_players) < 5 or len(new_players) > 15:
                continue

            dv = _recompute_dv(action, current_players, v_context)
            if dv > best_dv:
                best_dv  = dv
                best_idx = idx
                best_players = new_players
                # Pre-compute the new pid set so we don't repeat work on commit
                add_pids = {
                    _pid_of(p) for p in action["_add_players"]
                    if _pid_of(p) is not None
                }
                best_player_pids = (
                    (current_player_pids - action["_remove_pids"]) | add_pids
                )

        if best_idx is None:
            break

        chosen = action_pool.pop(best_idx)
        chosen["delta_v"] = best_dv

        # Commit: mark outgoing and expiring pids so they can't be re-used
        consumed_pids |= chosen["_remove_pids"]
        if chosen.get("_expiring_pid") is not None:
            consumed_pids.add(chosen["_expiring_pid"])

        # --- Update the three pieces of state ---
        # 1. Roster: current_players already computed as best_players above
        current_players    = best_players
        # 2. Opponent pool guard: current_player_pids tracks who is on my roster;
        #    any future action that tries to add a pid in this set is blocked by Guard 4.
        current_player_pids = best_player_pids
        # 3. Cap: recompute directly from the new roster to avoid delta drift.
        #    Let-walk players are still in current_players at this point; we adjust
        #    for them after the loop.
        current_v = v_context._compute_my_v(current_players)

        # Build clean output action (no private _ fields)
        out = {k: v for k, v in chosen.items() if not k.startswith("_")}
        applied.append(out)

    # --- Collect let-walk players ---
    # Expiring players whose pid is NOT in consumed_pids were not acted on → let walk.
    let_walk = []
    for i in range(len(my_roster_df)):
        row = my_roster_df.iloc[i]
        contract_exp = int(row.get("contract_exp") or 0)
        if contract_exp != current_season:
            continue
        pid = _pid_of(row.to_dict())
        if pid not in consumed_pids:
            let_walk.append(dict(
                name=str(row.get("name") or f"pid_{pid}"),
                age=int(float(row.get("age") or 27)),
                current_ovr=round(float(player_ovr(row)), 1),
            ))

    # --- Cap: recompute from final roster, then adjust for let-walk departures ---
    # current_players still contains the expiring-but-let-walk players at their old
    # salary.  After the season they depart and each slot costs MIN_CONTRACT instead.
    current_payroll = _current_salary(current_players)
    for p in current_players:
        pid = _pid_of(p)
        if pid in expiring_pids and pid not in consumed_pids:
            current_payroll -= float(p.get("salary", 0) or 0)
            current_payroll += MIN_CONTRACT

    remaining_cap_k = max(0.0, salary_cap - current_payroll)

    log.info(
        "optimize_decisions: applied %d actions  projected_V=%.4f  remaining_cap=$%.0fK",
        len(applied), current_v, remaining_cap_k,
    )

    return dict(
        actions=applied,
        let_walk=let_walk,
        projected_v=round(current_v, 6),
        initial_v=round(initial_v, 6),
        remaining_cap_k=round(remaining_cap_k),
    )
