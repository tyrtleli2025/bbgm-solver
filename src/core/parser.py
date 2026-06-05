"""
Data I/O Parser — Phase 0.

Reads a ZenGM Basketball GM league export JSON and converts it into the
DataFrame format expected by the rest of the solver.

ZenGM export schema (relevant fields)
--------------------------------------
player
    pid          int      player ID
    firstName    str
    lastName     str
    tid          int      team ID:
                           -3 = retired
                           -2 = draft prospect (undrafted)
                           -1 = free agent
                           ≥0 = rostered
    retiredYear  int|null if set → player has retired
    born
        year     int      birth year  (age = current_season − born.year)
    contract
        amount   float    annual salary in $thousands  (5 000 = $5 M/yr)
        exp      int      expiration season
    ratings      list     one entry per season; LAST entry = current season
        season   int
        hgt, stre, spd, jmp, endu, ins, dnk, ft, fg, tp,
        oiq, diq, drb, pss, reb   (base ratings 0–100)
        pot      int      potential (0–100); auto-computed but often present
        ovr      int      overall;  auto-computed — we recompute with formulas.py
        pos      str      position label

Sources
-------
https://basketball-gm.com/manual/customization/players/
https://basketball-gm.com/manual/customization/json-schema/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .formulas import BASE_RATINGS

# ---------------------------------------------------------------------------
# ZenGM sentinel team IDs
# ---------------------------------------------------------------------------

_TID_RETIRED         = -3
_TID_DRAFT_PROSPECT  = -2
_TID_FREE_AGENT      = -1

# ---------------------------------------------------------------------------
# Column layout of the output DataFrames
# ---------------------------------------------------------------------------

#: Non-rating metadata columns appended alongside the 15 base ratings.
META_COLUMNS: list[str] = ["pid", "name", "age", "pot", "salary", "tid", "pos"]

#: Full column order guaranteed in every output DataFrame.
OUTPUT_COLUMNS: list[str] = BASE_RATINGS + META_COLUMNS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_game_attr(data: dict, key: str, default):
    """
    Read a single key from gameAttributes, handling both export formats:
    - List of ``{"key": ..., "value": ...}`` objects (most exports).
    - Plain dict ``{"season": 2024, ...}`` (some simplified exports).
    """
    ga = data.get("gameAttributes", [])
    if isinstance(ga, list):
        for item in ga:
            if isinstance(item, dict) and item.get("key") == key:
                return item["value"]
    elif isinstance(ga, dict):
        if key in ga:
            return ga[key]
    return default


def _read_season(data: dict) -> int:
    """Return the current season integer, or 0 if absent."""
    val = _read_game_attr(data, "season", 0)
    return int(val) if val else 0


def _read_cap_info(data: dict) -> dict:
    """
    Extract salary-cap settings from a ZenGM export.

    Returns a dict with:
        salary_cap          : cap in $K (default 90 000 = $90 M)
        salary_cap_type     : "soft", "hard", or "none" (default "soft")
        soft_cap_trade_match: fraction for the 125 % rule (default 1.25)
    """
    salary_cap = float(_read_game_attr(data, "salaryCap", 90_000))
    salary_cap_type = str(_read_game_attr(data, "salaryCapType", "soft"))
    # ZenGM stores softCapTradeSalaryMatch as a percentage (e.g. 125 → 1.25)
    pct_raw = float(_read_game_attr(data, "softCapTradeSalaryMatch", 125))
    soft_cap_match = pct_raw / 100.0 if pct_raw > 1.0 else pct_raw
    return {
        "salary_cap":          salary_cap,
        "salary_cap_type":     salary_cap_type,
        "soft_cap_trade_match": soft_cap_match,
    }


def _parse_player(player: dict, current_season: int) -> Optional[dict]:
    """
    Convert one ZenGM player dict to a solver row dict.

    Returns ``None`` for retired players and undrafted draft prospects so the
    caller can skip them.  Free agents (tid=-1) are kept — they are active,
    just unsigned.
    """
    tid = int(player.get("tid", _TID_FREE_AGENT))

    # --- active-player filter ---
    if tid in (_TID_RETIRED, _TID_DRAFT_PROSPECT):
        return None
    if player.get("retiredYear") is not None:   # retired mid-season edge case
        return None

    # --- ratings: always use the most-recent season entry ---
    ratings_history = player.get("ratings", [])
    if not ratings_history:
        return None
    r = ratings_history[-1]

    # Base ratings — default to 50 if the key is absent in this export
    row: dict = {stat: float(r.get(stat, 50)) for stat in BASE_RATINGS}

    # --- age ---
    born_year = player.get("born", {}).get("year", 0)
    if current_season and born_year:
        age: float = float(current_season - born_year)
    else:
        age = 27.0   # neutral prime-age default

    # --- contract / salary ---
    contract = player.get("contract", {}) or {}
    salary = float(contract.get("amount", 0.0))   # already in $K

    # --- name ---
    first = player.get("firstName", "")
    last  = player.get("lastName",  "")
    name  = f"{first} {last}".strip() or f"pid_{player.get('pid', '?')}"

    row.update({
        "pid":    int(player.get("pid", -1)),
        "name":   name,
        "age":    age,
        "pot":    float(r.get("pot", 0.0)),
        "salary": salary,
        "tid":    tid,
        "pos":    str(r.get("pos", "")),
    })
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_league_data(
    data: dict,
    my_tid: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """
    Parse a ZenGM league export from an already-loaded dict.

    This is the core parsing logic reused by both ``parse_league_json``
    (file-based) and the HTTP server endpoint (payload-based).

    Parameters
    ----------
    data   : the top-level ZenGM export dict (keys: players, gameAttributes,
             teams, …).
    my_tid : team ID of the user's team (default 0).

    Returns
    -------
    (my_roster_df, league_rosters_dict, cap_info)
    """
    current_season = _read_season(data)
    cap_info       = _read_cap_info(data)

    # Build tid → abbreviation lookup from the teams list
    team_label: dict[int, str] = {}
    for team in data.get("teams", []):
        t = int(team.get("tid", -99))
        abbrev = (
            team.get("abbrev")
            or team.get("name")
            or f"team_{t}"
        )
        team_label[t] = str(abbrev)

    # Parse every player; bucket by tid
    rows_by_tid: dict[int, list[dict]] = {}
    for player in data.get("players", []):
        row = _parse_player(player, current_season)
        if row is None:
            continue
        rows_by_tid.setdefault(row["tid"], []).append(row)

    # My roster
    my_rows = rows_by_tid.get(my_tid, [])
    if not my_rows:
        available = sorted(rows_by_tid.keys())
        raise ValueError(
            f"No active players found for tid={my_tid}. "
            f"Available tids with active players: {available}"
        )
    my_roster_df = pd.DataFrame(my_rows)[OUTPUT_COLUMNS].reset_index(drop=True)

    # League dict — rostered opponents only (tid >= 0, excluding my team)
    league_rosters_dict: dict[str, pd.DataFrame] = {}
    for tid, rows in rows_by_tid.items():
        if tid == my_tid or tid < 0:
            continue
        label = team_label.get(tid, f"team_{tid}")
        league_rosters_dict[label] = (
            pd.DataFrame(rows)[OUTPUT_COLUMNS].reset_index(drop=True)
        )

    return my_roster_df, league_rosters_dict, cap_info


def extract_draft_picks(data: dict) -> list[dict]:
    """
    Extract draft picks from a ZenGM export.

    Each pick dict contains:
        tid       : int  — team owning the pick
        orig_tid  : int  — original team that owned it (may differ if traded)
        season    : int  — draft year
        round     : int  — 1 or 2 (or higher)
    """
    picks = []
    for raw in data.get("draftPicks", []):
        pick = {
            "tid": int(raw.get("tid", -1)),
            "orig_tid": int(raw.get("origTid", -1)),
            "season": int(raw.get("season", 0)),
            "round": int(raw.get("round", 1)),
        }
        if pick["tid"] >= 0:  # only include picks with valid owner
            picks.append(pick)
    return picks


def parse_league_json(
    filepath: str | Path,
    my_tid: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """
    Parse a ZenGM Basketball GM league export JSON file.

    Thin wrapper around ``parse_league_data`` that handles file I/O.

    Parameters
    ----------
    filepath : path to the ZenGM ``.json`` export file.
    my_tid   : team ID of the user's team (default 0).

    Returns
    -------
    (my_roster_df, league_rosters_dict, cap_info)

    Raises
    ------
    FileNotFoundError : if *filepath* does not exist.
    ValueError        : if no active players found for *my_tid*.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_league_data(data, my_tid=my_tid)
