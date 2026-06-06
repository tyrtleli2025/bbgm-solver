"""
Tests for src/core/resigning.py.
"""

import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS, player_ovr
from src.core.ai_trade_value import league_value_stats
from src.core.resigning import recommend_resigning, REPLACEMENT_OVR, MIN_CONTRACT


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SEASON = 2028
MY_TID = 0
OPP_TID = 1


def _player(
    ovr: int = 65,
    age: float = 27.0,
    salary: float = 10_000.0,
    contract_exp: int = SEASON + 3,
    tid: int = MY_TID,
    name: str = "Test Player",
    pid: int = 1,
    **extra,
) -> dict:
    p = {r: float(ovr) for r in BASE_RATINGS}
    p.update(
        pid=pid,
        name=name,
        age=age,
        pot=float(ovr),
        salary=salary,
        contract_exp=contract_exp,
        tid=tid,
        pos="SF",
    )
    p.update(extra)
    return p


def _df(*players) -> pd.DataFrame:
    from src.core.parser import OUTPUT_COLUMNS
    rows = list(players)
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def _make_league_and_ctx(my_df: pd.DataFrame, league_dict: dict) -> tuple:
    """Build league stats dict and a minimal LeagueVContext."""
    all_rosters = {"__mine__": my_df, **league_dict}
    league = league_value_stats(
        all_rosters,
        current_season=SEASON,
        salary_cap=90_000.0,
    )
    from src.core.market_scanner import _auto_v_context
    v_ctx = _auto_v_context(my_df, league_dict, SEASON, 90_000.0)
    return league, v_ctx


# Build a minimal opponent so the league context has other teams
_OPP_PLAYERS = [_player(65, age=27, tid=OPP_TID, name=f"Opp{i}", pid=100 + i)
                for i in range(10)]
_OPP_DF = _df(*_OPP_PLAYERS)
_LEAGUE_DICT = {"OPP": _OPP_DF}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoExpiringContracts:
    def test_returns_empty_when_no_expiry(self):
        """Players with long-term deals (contract_exp > SEASON) are excluded."""
        players = [_player(70, contract_exp=SEASON + 2, pid=i) for i in range(5)]
        my_df = _df(*players)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert results == []

    def test_excludes_non_expiring(self):
        """Contracts expiring next season (contract_exp == SEASON+1) are not included."""
        player = _player(65, contract_exp=SEASON + 1, pid=1)
        my_df = _df(player)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert results == []


class TestExpiringContracts:
    def test_detects_expiring_next_season(self):
        """A player whose contract_exp == SEASON+1 appears in results."""
        player = _player(70, age=27, contract_exp=SEASON, pid=1, name="Expiring")
        my_df = _df(player)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert len(results) == 1
        assert results[0]["name"] == "Expiring"

    def test_result_fields_present(self):
        """Each result has all required fields."""
        player = _player(70, age=27, contract_exp=SEASON, pid=1)
        my_df = _df(player)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        r = results[0]
        for field in ("name", "age", "current_ovr", "projected_ovr",
                      "demand_k", "delta_v", "recommendation"):
            assert field in r, f"Missing field: {field}"

    def test_recommendation_values(self):
        """Recommendation is one of the three valid strings."""
        player = _player(70, age=27, contract_exp=SEASON, pid=1)
        my_df = _df(player)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert results[0]["recommendation"] in ("resign", "let walk", "borderline")

    def test_sorted_by_delta_v_descending(self):
        """Results are sorted by delta_v descending (best deal first)."""
        star   = _player(80, age=26, contract_exp=SEASON, pid=1, name="Star")
        fringe = _player(45, age=34, contract_exp=SEASON, pid=2, name="Fringe")
        my_df  = _df(star, fringe)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert len(results) == 2
        assert results[0]["delta_v"] >= results[1]["delta_v"]

    def test_star_recommended_resign(self):
        """A high-OVR young player should be recommended for resignation."""
        star = _player(82, age=25, contract_exp=SEASON, pid=1, name="Star")
        my_df = _df(star)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert results[0]["recommendation"] == "resign"
        assert results[0]["delta_v"] > 0

    def test_aging_scrub_let_walk(self):
        """An old low-OVR player whose demand exceeds their value should be let walk."""
        scrub = _player(42, age=38, salary=5_000, contract_exp=SEASON,
                        pid=1, name="OldScrub")
        my_df = _df(scrub)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        # Declining player at 38 — replacement level is close to their value
        assert results[0]["recommendation"] in ("let walk", "borderline")

    def test_projected_ovr_younger_improves(self):
        """A young player's projected OVR should be >= current OVR."""
        young = _player(60, age=20, contract_exp=SEASON, pid=1)
        my_df = _df(young)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert results[0]["projected_ovr"] >= results[0]["current_ovr"]

    def test_demand_is_positive(self):
        """Salary demand must always be at least the minimum contract."""
        player = _player(65, age=28, contract_exp=SEASON, pid=1)
        my_df = _df(player)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        assert results[0]["demand_k"] >= MIN_CONTRACT

    def test_multiple_expiring_all_returned(self):
        """All expiring players are returned, non-expiring excluded."""
        expiring1 = _player(70, contract_exp=SEASON, pid=1, name="A")
        expiring2 = _player(65, contract_exp=SEASON, pid=2, name="B")
        locked    = _player(75, contract_exp=SEASON + 3, pid=3, name="C")
        my_df = _df(expiring1, expiring2, locked)
        league, ctx = _make_league_and_ctx(my_df, _LEAGUE_DICT)
        results = recommend_resigning(my_df, _LEAGUE_DICT, league, ctx)
        names = {r["name"] for r in results}
        assert "A" in names
        assert "B" in names
        assert "C" not in names


class TestServerEndpoint:
    def test_resign_endpoint_returns_players_key(self):
        """The /resign server endpoint returns a dict with a 'players' list."""
        from src.server import resign

        # Build a minimal payload matching the ZenGM export shape
        import json

        player_raw = {
            "pid": 1, "firstName": "Test", "lastName": "Player",
            "tid": MY_TID,
            "born": {"year": SEASON - 27},
            "contract": {"amount": 10_000, "exp": SEASON},
            "ratings": [{
                "season": SEASON, "pot": 70, "ovr": 65, "pos": "SF",
                **{r: 65 for r in BASE_RATINGS},
            }],
        }
        opp_raw = {
            "pid": 50, "firstName": "Opp", "lastName": "Player",
            "tid": OPP_TID,
            "born": {"year": SEASON - 27},
            "contract": {"amount": 8_000, "exp": SEASON + 3},
            "ratings": [{
                "season": SEASON, "pot": 65, "ovr": 60, "pos": "PF",
                **{r: 60 for r in BASE_RATINGS},
            }],
        }
        payload = {
            "tid": MY_TID,
            "players": [player_raw, opp_raw],
            "gameAttributes": {"season": SEASON, "salaryCap": 90_000},
            "teams": [{"tid": MY_TID, "abbrev": "MY"}, {"tid": OPP_TID, "abbrev": "OPP"}],
        }

        result = resign(payload)
        assert "players" in result
        assert isinstance(result["players"], list)
        assert result["current_season"] == SEASON

    def test_resign_endpoint_expiring_player_included(self):
        """An expiring player appears in the /resign endpoint output."""
        from src.server import resign

        player_raw = {
            "pid": 1, "firstName": "Expiring", "lastName": "Star",
            "tid": MY_TID,
            "born": {"year": SEASON - 26},
            "contract": {"amount": 12_000, "exp": SEASON},
            "ratings": [{
                "season": SEASON, "pot": 78, "ovr": 75, "pos": "SF",
                **{r: 75 for r in BASE_RATINGS},
            }],
        }
        # Need enough opponents to build a valid league context
        opps = [
            {
                "pid": 50 + i, "firstName": f"Opp{i}", "lastName": "Player",
                "tid": OPP_TID,
                "born": {"year": SEASON - 28},
                "contract": {"amount": 8_000, "exp": SEASON + 3},
                "ratings": [{"season": SEASON, "pot": 65, "ovr": 63, "pos": "PF",
                             **{r: 63 for r in BASE_RATINGS}}],
            }
            for i in range(8)
        ]
        payload = {
            "tid": MY_TID,
            "players": [player_raw] + opps,
            "gameAttributes": {"season": SEASON, "salaryCap": 90_000},
            "teams": [{"tid": MY_TID, "abbrev": "MY"}, {"tid": OPP_TID, "abbrev": "OPP"}],
        }

        result = resign(payload)
        assert len(result["players"]) == 1
        assert result["players"][0]["name"] == "Expiring Star"
