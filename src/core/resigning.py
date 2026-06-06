"""
Re-signing advisor.

For each player whose contract expires next season, project their ratings
forward one year, estimate what they'll demand in free agency, and compute
the ΔV of keeping them (at that salary) versus letting them walk (replaced
by a replacement-level player at the league minimum).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.core.formulas import player_ovr
from src.project import project_ratings, project_ovr, estimate_next_contract

if TYPE_CHECKING:
    from src.value import LeagueVContext

REPLACEMENT_OVR: float = 40.0
MIN_CONTRACT: float = 1_200.0   # $K

_THRESHOLDS = dict(resign=0.005, let_walk=-0.005)


def _player_dict_from_row(row: pd.Series | dict, current_season: int) -> dict:
    """Convert a DataFrame row to a full player dict suitable for LeagueVContext."""
    d = dict(row) if isinstance(row, dict) else row.to_dict()
    # Ensure contract_exp is present
    d.setdefault("contract_exp", current_season + 1)
    return d


def _replacement_player(tid: int, current_season: int) -> dict:
    """Synthetic replacement-level player (OVR ≈ 40) on a min contract."""
    from src.core.formulas import BASE_RATINGS
    p = {r: REPLACEMENT_OVR for r in BASE_RATINGS}
    p.update(
        pid=-1,
        tid=tid,
        age=25.0,
        salary=MIN_CONTRACT,
        contract_exp=current_season + 2,
        pot=REPLACEMENT_OVR,
    )
    return p


def recommend_resigning(
    my_roster_df: pd.DataFrame,
    league_rosters_dict: dict[str, pd.DataFrame],
    league: dict,
    v_context: "LeagueVContext",
) -> list[dict]:
    """
    Recommend whether to re-sign each expiring player on my roster.

    For each player whose contract_exp == current_season + 1 (i.e. expires
    after this season, meaning they're a free agent next offseason):

      1. Project ratings 1 year forward.
      2. Estimate the salary they'll demand (with loyalty discount).
      3. Compute ΔV: keeping them at that salary vs. replacing them with a
         replacement-level player at the league minimum.
      4. Tag: "resign" (ΔV > 0.005), "let walk" (ΔV < -0.005), "borderline".

    Returns a list of dicts sorted by delta_v descending.
    """
    current_season = int(league.get("current_season", 0))
    my_tid = v_context.my_tid
    ovr_mean = float(league.get("ovr_mean", 65.0))
    ovr_std = float(league.get("ovr_std", 10.0))

    results = []

    for i in range(len(my_roster_df)):
        row = my_roster_df.iloc[i]
        contract_exp = int(row.get("contract_exp") or 0)

        if contract_exp != current_season:
            continue

        name = str(row.get("name") or f"pid_{row.get('pid')}")
        age = float(row.get("age") or 27.0)
        current_ovr = float(player_ovr(row))

        # --- 1. Project ratings 1 year forward ---
        from src.core.formulas import BASE_RATINGS
        current_ratings = {r: float(row.get(r, 50.0)) for r in BASE_RATINGS}
        proj_ratings = project_ratings(current_ratings, age, years_forward=1)
        proj_ovr = project_ovr(proj_ratings)

        # --- 2. Estimate demand (with loyalty discount) ---
        demand = max(MIN_CONTRACT, estimate_next_contract(
            projected_ovr=proj_ovr,
            age=age + 1.0,
            is_my_team=True,
            ovr_mean=ovr_mean,
            ovr_std=ovr_std,
            max_contract=float(league.get("salary_cap", 90_000.0)) * 0.35,
            min_contract=MIN_CONTRACT,
        ))

        # --- 3. Compute ΔV: resign vs. let walk ---
        player_d = _player_dict_from_row(row, current_season)

        # Build the "resigned" version: update salary and extend contract
        resigned = dict(player_d)
        resigned["salary"] = demand
        resigned["contract_exp"] = current_season + 3   # typical 2-year deal

        # Replacement player that fills the roster slot
        replacement = _replacement_player(my_tid, current_season)

        # ΔV(resign)  = V(keep them) − V(baseline)
        # ΔV(let walk)= V(replacement) − V(baseline)
        # Net ΔV of resigning over letting walk = ΔV(resign) − ΔV(let walk)
        dv_resign = v_context.delta_v(
            add_players=[resigned],
            remove_players=[player_d],
        )
        dv_walk = v_context.delta_v(
            add_players=[replacement],
            remove_players=[player_d],
        )
        delta_v = dv_resign - dv_walk

        if delta_v > _THRESHOLDS["resign"]:
            rec = "resign"
        elif delta_v < _THRESHOLDS["let_walk"]:
            rec = "let walk"
        else:
            rec = "borderline"

        results.append(dict(
            name=name,
            age=int(age),
            current_ovr=round(current_ovr, 1),
            projected_ovr=round(proj_ovr, 1),
            demand_k=round(demand),
            delta_v=round(delta_v, 6),
            recommendation=rec,
        ))

    results.sort(key=lambda r: r["delta_v"], reverse=True)
    return results
