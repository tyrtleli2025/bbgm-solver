"""
Horizon-aware team value function V.

V(state) = Σ_{t=1}^{H}  γ^t  ×  P(my_team wins title in season t)

Title probability is computed via softmax over playoff-team MOVs:
    P(title | team_i) = exp(β × MOV_i) / Σ_{j ∈ playoffs} exp(β × MOV_j)

Team MOV is the engine's closed-form model (engine_reference.md §Team OVR):
    MOV = -k + a × Σ_{i=0}^{9} exp(b×i) × sorted_OVR_i   (top-10 players only)

For each future season, player ratings are projected forward using the aging
model (src/project.py), rosters are pruned by contract status, gaps are filled
with replacement-level players, and draft picks add phantom rookies.

Performance
-----------
LeagueVContext precomputes all 30 opponent teams' MOV trajectories once (~30ms).
After that, delta_v() for a single trade candidate runs in ~1ms by projecting
only my team's roster and rerunning the softmax.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.formulas import BASE_RATINGS
from src.project import (
    project_ratings,
    project_ovr,
    project_contract_status,
    estimate_next_contract,
)

# ---------------------------------------------------------------------------
# Team MOV constants (engine_reference.md §Team OVR)
# ---------------------------------------------------------------------------

_RS_A, _RS_B, _RS_K   = 0.3334, -0.1609, 102.98    # regular season
_PO_A, _PO_B, _PO_K   = 0.6388, -0.2245, 157.43    # playoffs

_RS_WEIGHTS = tuple(_RS_A * math.exp(_RS_B * i) for i in range(10))
_PO_WEIGHTS = tuple(_PO_A * math.exp(_PO_B * i) for i in range(10))


def _mov(sorted_top10_ovrs: list[float], playoff: bool = False) -> float:
    """Predicted margin of victory from the top-10 sorted OVRs."""
    w = _PO_WEIGHTS if playoff else _RS_WEIGHTS
    k = _PO_K       if playoff else _RS_K
    ovrs = list(sorted_top10_ovrs[:10])
    while len(ovrs) < 10:
        ovrs.append(0.0)
    return -k + sum(w[i] * ovrs[i] for i in range(10))


# ---------------------------------------------------------------------------
# Draft pick expected OVR / salary tables (§4.2 and §3.5)
# ---------------------------------------------------------------------------

# Interpolated from anchors: pick 1→64.3, pick 10→55.1, pick 30→49.3
def _build_pick_ovr_table() -> dict[int, float]:
    table: dict[int, float] = {}
    for s in range(1, 11):
        t = (s - 1) / 9.0
        table[s] = 64.3 + t * (55.1 - 64.3)
    for s in range(11, 31):
        t = (s - 10) / 20.0
        table[s] = 55.1 + t * (49.3 - 55.1)
    return table

VALUE_BY_PICK: dict[int, float] = _build_pick_ovr_table()
UNDRAFTED_VALUE: float = 36.6

# Rookie salary anchors from §3.5 (in $K)
_SALARY_ANCHORS = [(1, 12500), (5, 9850), (10, 6620), (15, 5240), (20, 3860), (30, 1360)]

def _rookie_salary(pick: int) -> float:
    """Approximate rookie salary for a given pick number (in $K)."""
    if pick <= 0 or pick > 30:
        return 1200.0   # round 2 / undrafted → min contract
    # Linear interpolation between anchors
    for i in range(len(_SALARY_ANCHORS) - 1):
        lo_pick, lo_sal = _SALARY_ANCHORS[i]
        hi_pick, hi_sal = _SALARY_ANCHORS[i + 1]
        if lo_pick <= pick <= hi_pick:
            t = (pick - lo_pick) / (hi_pick - lo_pick)
            return lo_sal + t * (hi_sal - lo_sal)
    return 1360.0   # beyond pick 30


def _pick_ovr(slot: int, round_num: int = 1,
              undrafted_pool: list[dict] | None = None) -> float:
    """
    Expected OVR of a drafted player from a given slot.

    If undrafted_pool (list of actual prospects, sorted by OVR descending) is provided,
    use the Nth prospect's OVR for slot N. Otherwise fall back to VALUE_BY_PICK average.
    """
    if round_num > 1:
        return UNDRAFTED_VALUE + 5.0    # late 2nd ≈ slightly above replacement

    if undrafted_pool and slot > 0 and slot <= len(undrafted_pool):
        from src.core.formulas import player_ovr as _player_ovr
        return float(_player_ovr(undrafted_pool[slot - 1]))

    return VALUE_BY_PICK.get(min(max(slot, 1), 30), 49.3)


# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------

H_DEFAULT              = 5      # horizon (seasons)
GAMMA_DEFAULT          = 0.95   # discount factor
BETA_DEFAULT           = 0.15   # softmax temperature
REPLACEMENT_OVR        = 40.0   # OVR below which a player isn't re-signed
MIN_ROSTER             = 10
NUM_PLAYOFF_TEAMS      = 16


# ---------------------------------------------------------------------------
# LeagueState — raw data container
# ---------------------------------------------------------------------------

@dataclass
class LeagueState:
    """
    Holds the full league snapshot needed for V computation.

    All player dicts contain the 15 base ratings as top-level keys plus:
      pid, tid, age, salary, contract_exp, pot (optional)

    These are extracted directly from the ZenGM JSON export so that
    contract_exp is available (the DataFrame parser drops it).

    undrafted_by_year: dict[int, list[dict]] — undrafted prospects (tid=-2)
      grouped by draft year, sorted by player_ovr descending. Used to value
      draft picks based on actual prospect ratings instead of averages.
    """
    players:         list[dict]
    teams:           list[dict]
    picks:           list[dict]
    current_season:  int
    salary_cap:      float
    my_tid:          int
    num_teams:       int = 30
    undrafted_by_year: dict[int, list[dict]] = field(default_factory=dict)

    # ----------- factory ---------------------------------------------------

    @classmethod
    def from_data(cls, data: dict, my_tid: int = 0) -> "LeagueState":
        """Build a LeagueState from a raw ZenGM export dict."""
        from src.core.parser import _read_season, _read_cap_info
        from src.core.formulas import player_ovr as _player_ovr

        current_season = _read_season(data)
        cap_info       = _read_cap_info(data)

        players = []
        undrafted_by_year: dict[int, list[dict]] = {}

        for raw in data.get("players", []):
            tid = int(raw.get("tid", -1))
            if tid == -2:  # undrafted prospect
                p = _extract_undrafted(raw, current_season)
                if p is not None:
                    # Infer draft year from ratings array (the era when they'd be drafted)
                    # If draftYear is set, use it; else use most recent rating season + 1
                    draft_year = int(raw.get("draftYear", 0))
                    if not draft_year:
                        ratings = raw.get("ratings", [])
                        if ratings:
                            latest_season = int(ratings[-1].get("season", 0))
                            # Prospect should be drafted the season after their last rating
                            draft_year = latest_season + 1
                        else:
                            draft_year = current_season + 1
                    undrafted_by_year.setdefault(draft_year, []).append(p)
            else:
                p = _extract_player(raw, current_season)
                if p is not None:
                    players.append(p)

        # Sort undrafted by OVR descending (best prospects first)
        for year in undrafted_by_year:
            undrafted_by_year[year].sort(key=lambda p: _player_ovr(p), reverse=True)

        teams = data.get("teams", [])
        picks = data.get("draftPicks", [])
        return cls(
            players=players,
            teams=teams,
            picks=picks,
            current_season=current_season,
            salary_cap=cap_info["salary_cap"],
            my_tid=my_tid,
            num_teams=max(len(teams), 30),
            undrafted_by_year=undrafted_by_year,
        )

    @classmethod
    def from_file(cls, filepath: str | Path, my_tid: int = 0) -> "LeagueState":
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_data(data, my_tid=my_tid)


def _extract_undrafted(raw: dict, current_season: int) -> Optional[dict]:
    """Convert a raw undrafted prospect (tid=-2) to value.py format."""
    ratings_list = raw.get("ratings", [])
    if not ratings_list:
        return None
    r = ratings_list[-1]

    born_year = raw.get("born", {}).get("year", 0)
    age = float(current_season - born_year) if (current_season and born_year) else 20.0

    p: dict = {k: float(r.get(k, 50.0)) for k in BASE_RATINGS}
    p.update(
        pid=int(raw.get("pid", -1)),
        tid=-2,
        age=age,
        pot=float(r.get("pot", 0.0)),
        salary=1200.0,  # league minimum for undrafted
        contract_exp=current_season + 1,  # default contract year
        name=(raw.get("firstName", "") + " " + raw.get("lastName", "")).strip(),
    )
    return p


def _extract_player(raw: dict, current_season: int) -> Optional[dict]:
    """Convert a raw ZenGM player dict to the flat format used by value.py."""
    tid = int(raw.get("tid", -3))
    if tid == -3:          # retired
        return None
    if tid == -2:          # draft prospect — skip in main players list
        return None
    if raw.get("retiredYear") is not None:
        return None

    ratings_list = raw.get("ratings", [])
    if not ratings_list:
        return None
    r = ratings_list[-1]         # most recent season ratings

    born_year = raw.get("born", {}).get("year", 0)
    age       = float(current_season - born_year) if (current_season and born_year) else 27.0

    contract = raw.get("contract", {}) or {}
    salary      = float(contract.get("amount", 1200.0))
    contract_exp = int(contract.get("exp", current_season + 1))

    p: dict = {k: float(r.get(k, 50.0)) for k in BASE_RATINGS}
    p.update(
        pid=int(raw.get("pid", -1)),
        tid=tid,
        age=age,
        pot=float(r.get("pot", 0.0)),
        salary=salary,
        contract_exp=contract_exp,
        name=(raw.get("firstName", "") + " " + raw.get("lastName", "")).strip(),
    )
    return p


# ---------------------------------------------------------------------------
# Standalone V computation (no precomputation, slower)
# ---------------------------------------------------------------------------

def compute_V(
    my_tid: int,
    league_state: LeagueState,
    H: int              = H_DEFAULT,
    gamma: float        = GAMMA_DEFAULT,
    beta: float         = BETA_DEFAULT,
    replacement_ovr: float = REPLACEMENT_OVR,
) -> float:
    """
    Compute V for my_tid from scratch over the H-season horizon.

    This is the reference implementation — correct but not optimised for
    repeated calls.  For beam search, use LeagueVContext.delta_v() instead.
    """
    ctx = LeagueVContext(league_state, H=H, gamma=gamma, beta=beta,
                         replacement_ovr=replacement_ovr)
    return ctx.v_current


def asset_marginal_value(
    my_tid: int,
    league_state: LeagueState,
    add_players:    Optional[list[dict]] = None,
    remove_players: Optional[list[dict]] = None,
    add_picks:      Optional[list[dict]] = None,
    remove_picks:   Optional[list[dict]] = None,
    H: int              = H_DEFAULT,
    gamma: float        = GAMMA_DEFAULT,
    beta: float         = BETA_DEFAULT,
    replacement_ovr: float = REPLACEMENT_OVR,
) -> float:
    """ΔV = V(state after modification) − V(current state)."""
    ctx = LeagueVContext(league_state, H=H, gamma=gamma, beta=beta,
                         replacement_ovr=replacement_ovr)
    return ctx.delta_v(
        add_players    = add_players    or [],
        remove_players = remove_players or [],
        add_picks      = add_picks      or [],
        remove_picks   = remove_picks   or [],
    )


# ---------------------------------------------------------------------------
# LeagueVContext — fast repeated evaluations
# ---------------------------------------------------------------------------

class LeagueVContext:
    """
    Precomputes all opponent team MOVs across the H-season horizon once,
    then provides fast delta_v() evaluation for each trade candidate.

    Usage
    -----
    ctx = LeagueVContext(league_state)        # ~30 ms
    dv  = ctx.delta_v(incoming, outgoing)     # ~1 ms per call
    """

    def __init__(
        self,
        league_state: LeagueState,
        H: int              = H_DEFAULT,
        gamma: float        = GAMMA_DEFAULT,
        beta: float         = BETA_DEFAULT,
        replacement_ovr: float = REPLACEMENT_OVR,
    ) -> None:
        self.ls              = league_state
        self.H               = H
        self.gamma           = gamma
        self.beta            = beta
        self.replacement_ovr = replacement_ovr
        self.my_tid          = league_state.my_tid

        # PID → full player dict (for looking up incoming players by pid)
        self._pid_map: dict[int, dict] = {
            int(float(p["pid"])): p
            for p in league_state.players
            if p.get("pid") is not None
        }

        # My team's current player list
        self._my_players: list[dict] = [
            p for p in league_state.players
            if int(p.get("tid", -1)) == self.my_tid
        ]

        # Precompute opponent season MOVs
        # _opp_movs[t_idx] = {tid: regular_season_mov}  for t = t_idx + 1
        self._opp_movs: list[dict[int, float]] = []
        self._precompute_opponents()

        # V for current state
        self.v_current: float = self._compute_my_v(self._my_players)

    # -----------------------------------------------------------------------
    # Precomputation
    # -----------------------------------------------------------------------

    def _precompute_opponents(self) -> None:
        """Compute regular-season MOV for each opponent team for t=1..H."""
        my_tid = self.my_tid
        ls     = self.ls

        for t in range(1, self.H + 1):
            team_ovrs: dict[int, list[float]] = {}

            for player in ls.players:
                tid = int(player.get("tid", -1))
                if tid < 0 or tid == my_tid:
                    continue
                proj_ovr = self._project_player(player, t)
                if proj_ovr is not None:
                    team_ovrs.setdefault(tid, []).append(proj_ovr)

            # Add draft-pick phantoms for this season
            for pick in ls.picks:
                pick_tid    = int(pick.get("tid",    -1))
                pick_season = int(pick.get("season", ls.current_season + 1))
                round_num   = int(pick.get("round",  1))
                if pick_tid < 0 or pick_tid == my_tid:
                    continue
                if pick_season != ls.current_season + t:
                    continue
                slot = _estimate_slot(pick_tid, t, team_ovrs)
                undraft = ls.undrafted_by_year.get(pick_season, [])
                team_ovrs.setdefault(pick_tid, []).append(
                    _pick_ovr(slot, round_num, undrafted_pool=undraft)
                )

            # Fill rosters to minimum size
            all_opp_tids = {
                int(p.get("tid", -1))
                for p in ls.players
                if int(p.get("tid", -1)) >= 0 and int(p.get("tid", -1)) != my_tid
            }
            for tid in all_opp_tids:
                ovrs = team_ovrs.get(tid, [])
                while len(ovrs) < MIN_ROSTER:
                    ovrs.append(self.replacement_ovr)
                team_ovrs[tid] = ovrs

            self._opp_movs.append({
                tid: _mov(sorted(ovrs, reverse=True))
                for tid, ovrs in team_ovrs.items()
            })

    def _project_player(self, player: dict, t: int) -> Optional[float]:
        """
        Project a player's OVR t seasons forward.
        Returns None if the player departs and won't be re-signed.
        """
        age  = float(player.get("age",  27.0))
        ratings = {r: float(player.get(r, 50.0)) for r in BASE_RATINGS}
        contract_exp = int(player.get("contract_exp", self.ls.current_season + 1))
        current_season = self.ls.current_season

        proj_r   = project_ratings(ratings, age, t)
        proj_ovr = project_ovr(proj_r)

        under_contract = project_contract_status(
            contract_exp, current_season, current_season + t
        )
        if under_contract:
            return proj_ovr

        # Contract expired: re-sign if valuable enough
        if proj_ovr >= self.replacement_ovr:
            return proj_ovr   # assume re-signed (simplified)
        return None            # departed

    # -----------------------------------------------------------------------
    # My-team V computation (fast — opponent MOVs already cached)
    # -----------------------------------------------------------------------

    def _compute_my_v(self, my_players: list[dict]) -> float:
        v = 0.0
        my_tid = self.my_tid
        ls     = self.ls

        for t_idx in range(self.H):
            t         = t_idx + 1
            opp_movs  = self._opp_movs[t_idx]

            # Project my team
            my_ovrs: list[float] = []
            for player in my_players:
                proj_ovr = self._project_player(player, t)
                if proj_ovr is not None:
                    my_ovrs.append(proj_ovr)

            # Add my picks that resolve this season
            for pick in ls.picks:
                if int(pick.get("tid", -1)) != my_tid:
                    continue
                pick_season = int(pick.get("season", ls.current_season + 1))
                if pick_season != ls.current_season + t:
                    continue
                round_num = int(pick.get("round", 1))
                slot      = _estimate_slot_for_my_team(my_ovrs, opp_movs)
                undraft = ls.undrafted_by_year.get(pick_season, [])
                my_ovrs.append(_pick_ovr(slot, round_num, undrafted_pool=undraft))

            while len(my_ovrs) < MIN_ROSTER:
                my_ovrs.append(self.replacement_ovr)

            my_mov = _mov(sorted(my_ovrs, reverse=True))

            # All team MOVs this season
            all_movs = {my_tid: my_mov, **opp_movs}

            # Playoff teams: top 16 by regular-season MOV
            playoff_tids = [
                tid for tid, _ in
                sorted(all_movs.items(), key=lambda x: x[1], reverse=True)[
                    :NUM_PLAYOFF_TEAMS
                ]
            ]

            if my_tid not in playoff_tids:
                # Miss playoffs → zero title equity this season
                v += (self.gamma ** t) * 0.0
                continue

            # Softmax title probability over playoff teams
            movs_po = {tid: all_movs[tid] for tid in playoff_tids}
            p_title  = _title_prob(movs_po, my_tid, self.beta)
            v += (self.gamma ** t) * p_title

        return v

    # -----------------------------------------------------------------------
    # delta_v: the public interface for the trade search
    # -----------------------------------------------------------------------

    def delta_v(
        self,
        add_players:    list[dict],
        remove_players: list[dict],
        add_picks:      list[dict] | None = None,
        remove_picks:   list[dict] | None = None,
    ) -> float:
        """
        ΔV = V(my team after trade) − V(my team before trade).

        add_players / remove_players are player dicts as returned by
        DataFrame.to_dict(); we look up the full player info (including
        contract_exp) via pid from the precomputed pid map.
        """
        remove_pids = {
            int(float(p["pid"]))
            for p in remove_players
            if p.get("pid") is not None
        }

        # Start from my current roster, drop outgoing
        new_players = [
            p for p in self._my_players
            if int(float(p.get("pid", -1))) not in remove_pids
        ]

        # Add incoming (look up full player info if available)
        for p in add_players:
            pid = int(float(p.get("pid", -1))) if p.get("pid") is not None else -1
            full = self._pid_map.get(pid)
            if full is not None:
                player_to_add = dict(full)
            else:
                # Player not in our map (e.g. a newly projected phantom):
                # use the dict directly, estimating contract_exp if missing
                player_to_add = dict(p)
                if "contract_exp" not in player_to_add:
                    player_to_add["contract_exp"] = (
                        self.ls.current_season + 3
                    )
            player_to_add["tid"] = self.my_tid
            new_players.append(player_to_add)

        return self._compute_my_v(new_players) - self.v_current

    def compute_v_for_roster_df(self, roster_df) -> float:
        """
        Compute V for my team given a roster DataFrame (used in beam_search).
        Looks up each player by pid to get full contract info.
        """
        players: list[dict] = []
        for i in range(len(roster_df)):
            row = roster_df.iloc[i].to_dict()
            pid = int(float(row.get("pid", -1)))
            if pid in self._pid_map:
                players.append(self._pid_map[pid])
            else:
                p = {r: float(row.get(r, 50.0)) for r in BASE_RATINGS}
                p.update(
                    pid          = pid,
                    tid          = self.my_tid,
                    age          = float(row.get("age", 27.0)),
                    salary       = float(row.get("salary", 1200.0)),
                    contract_exp = int(row.get("contract_exp", self.ls.current_season + 3)),
                )
                players.append(p)
        return self._compute_my_v(players)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_prob(movs: dict[int, float], my_tid: int, beta: float) -> float:
    """Softmax title probability for my_tid among the given teams."""
    exp_vals = {tid: math.exp(beta * m) for tid, m in movs.items()}
    total    = sum(exp_vals.values())
    return exp_vals.get(my_tid, 0.0) / total if total > 0.0 else 0.0


def _estimate_slot(tid: int, t: int, current_team_ovrs: dict[int, list[float]]) -> int:
    """
    Rough pick-slot estimate based on how a team's projected OVR compares to others.
    Worse projected teams get better (lower number) slots.
    """
    team_movs = {
        other_tid: _mov(sorted(ovrs, reverse=True))
        for other_tid, ovrs in current_team_ovrs.items()
    }
    sorted_by_mov = sorted(team_movs, key=lambda x: team_movs.get(x, 0.0))
    try:
        rank = sorted_by_mov.index(tid) + 1
    except ValueError:
        rank = 15
    # Roughly: rank among bottom half → lottery pick, otherwise mid-to-late
    return max(1, min(30, rank))


def _estimate_slot_for_my_team(
    my_ovrs: list[float], opp_movs: dict[int, float]
) -> int:
    """Estimate my pick slot relative to opponents."""
    if not my_ovrs:
        return 1
    my_mov = _mov(sorted(my_ovrs, reverse=True))
    rank   = sum(1 for m in opp_movs.values() if m < my_mov) + 1
    # Better teams pick later
    n_teams = len(opp_movs) + 1
    slot    = n_teams - rank + 1
    return max(1, min(30, slot))
