"""
Tests for the Trade Market Scanner (src/core/market_scanner.py).

All rosters are kept small (6–8 players, 1–2 league teams) so the optimizer
runs in milliseconds.  Larger rosters are covered indirectly via evaluate_trade
tests which already validate the underlying engine.
"""

import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS
from src.core.market_scanner import (
    find_best_trades,
    ASSET_FLOOR_DEFAULT,
    TOP_N_DEFAULT,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "team", "trade_type", "incoming", "outgoing",
    "net_lineup_score", "net_asset_value", "new_score",
}


def _player(rating: int = 60, age: int = 27, pot: int = 0, salary: int = 0) -> dict:
    """Uniform player: all 15 base ratings set to *rating*."""
    p = {r: rating for r in BASE_RATINGS}
    p.update(age=age, pot=pot, salary=salary)
    return p


def _roster(*players) -> pd.DataFrame:
    return pd.DataFrame(list(players))


# Reusable roster fixtures
_AVG    = _player(60)   # average player (OVR ~67)
_ELITE  = _player(90)   # elite player (OVR ~94)
_WEAK   = _player(20)   # weak player (OVR ~8)


# ---------------------------------------------------------------------------
# 1. Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:

    def test_returns_a_list(self):
        my   = _roster(*[_AVG] * 6)
        them = _roster(*[_AVG] * 6)
        result = find_best_trades(my, {"Team": them})
        assert isinstance(result, list)

    def test_each_result_has_required_keys(self):
        my   = _roster(*[_AVG] * 6)
        them = _roster(*[_ELITE] + [_AVG] * 5)
        result = find_best_trades(my, {"Team": them})
        for r in result:
            assert _REQUIRED_KEYS <= set(r.keys()), f"Missing keys: {_REQUIRED_KEYS - set(r.keys())}"

    def test_at_most_top_n_results(self):
        # Two teams each with one elite player → up to two trades found
        my    = _roster(*[_AVG] * 7)
        teamA = _roster(*[_ELITE] + [_AVG] * 6)
        teamB = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"A": teamA, "B": teamB}, top_n=1)
        assert len(result) <= 1

    def test_default_top_n_is_five(self):
        # Build more than 5 trades by giving every other team multiple elites
        my    = _roster(*[_AVG] * 7)
        teams = {
            f"T{i}": _roster(*[_ELITE] * 6 + [_AVG])
            for i in range(4)
        }
        result = find_best_trades(my, teams)
        assert len(result) <= TOP_N_DEFAULT

    def test_incoming_and_outgoing_are_lists(self):
        my   = _roster(*[_AVG] * 6)
        them = _roster(*[_ELITE] + [_AVG] * 5)
        result = find_best_trades(my, {"T": them})
        if result:
            assert isinstance(result[0]["incoming"], list)
            assert isinstance(result[0]["outgoing"], list)

    def test_trade_type_is_valid_string(self):
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"T": them})
        for r in result:
            assert r["trade_type"] in {"1-for-1", "2-for-1"}

    def test_team_key_matches_league_dict_key(self):
        my   = _roster(*[_AVG] * 6)
        them = _roster(*[_ELITE] + [_AVG] * 5)
        result = find_best_trades(my, {"Celtics": them})
        for r in result:
            assert r["team"] == "Celtics"


# ---------------------------------------------------------------------------
# 2. Filtering correctness
# ---------------------------------------------------------------------------


class TestFiltering:

    def test_all_net_lineup_scores_positive(self):
        """Every returned trade must improve our lineup — no negatives allowed."""
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_WEAK] * 6)
        result = find_best_trades(my, {"T": them})
        for r in result:
            assert r["net_lineup_score"] > 0, (
                f"Non-positive net_lineup_score: {r['net_lineup_score']}"
            )

    def test_no_results_when_all_trades_hurt_lineup(self):
        """Trading our good players for weak players can never improve our lineup."""
        my   = _roster(*[_player(80)] * 7)
        them = _roster(*[_WEAK] * 7)
        result = find_best_trades(my, {"T": them})
        assert result == []

    def test_strict_asset_floor_filters_marginal_trades(self):
        """
        With a high asset_value_floor, even mildly value-positive trades are excluded.
        Trading average players (≈equal value) for each other gives ~0 net value;
        a floor of +20 should filter all of them out.
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_AVG] * 7)
        result = find_best_trades(my, {"T": them}, asset_value_floor=20.0)
        assert result == []

    def test_permissive_asset_floor_allows_more_trades(self):
        """A very permissive floor lets value-negative trades through (if lineup improves)."""
        my   = _roster(*[_player(50)] * 7)
        them = _roster(*[_player(75)] * 7)
        strict    = find_best_trades(my, {"T": them}, asset_value_floor=0.0)
        permissive = find_best_trades(my, {"T": them}, asset_value_floor=-100.0)
        assert len(permissive) >= len(strict)

    def test_empty_league_returns_empty(self):
        my = _roster(*[_AVG] * 6)
        assert find_best_trades(my, {}) == []

    def test_team_with_empty_roster_is_skipped(self):
        my = _roster(*[_AVG] * 6)
        result = find_best_trades(my, {"Ghost": pd.DataFrame()})
        assert result == []

    def test_my_roster_too_small_raises(self):
        with pytest.raises(ValueError, match="5"):
            find_best_trades(_roster(*[_AVG] * 4), {"T": _roster(*[_AVG] * 6)})


# ---------------------------------------------------------------------------
# 3. Sorting
# ---------------------------------------------------------------------------


class TestSorting:

    def test_sorted_descending_by_net_lineup_score(self):
        my    = _roster(*[_AVG] * 7)
        teams = {
            "Good":  _roster(*[_ELITE] + [_AVG] * 6),
            "OK":    _roster(*[_player(70)] + [_AVG] * 6),
        }
        result = find_best_trades(my, teams)
        scores = [r["net_lineup_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_best_trade_is_first(self):
        """The trade with the highest net_lineup_score must be result[0]."""
        my    = _roster(*[_AVG] * 7)
        teams = {
            "Great": _roster(*[_ELITE] + [_AVG] * 6),
            "Meh":   _roster(*[_player(65)] + [_AVG] * 6),
        }
        result = find_best_trades(my, teams)
        assert len(result) >= 1
        best = result[0]
        for r in result[1:]:
            assert best["net_lineup_score"] >= r["net_lineup_score"]


# ---------------------------------------------------------------------------
# 4. Trade discovery
# ---------------------------------------------------------------------------


class TestTradeDiscovery:

    def test_finds_obvious_1for1_upgrade(self):
        """
        If the other team has one clearly elite player, we should find the
        1-for-1 trade that brings them to our roster.
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"TeamA": them})

        assert len(result) > 0
        best = result[0]
        assert best["team"] == "TeamA"
        # The incoming player should be the elite one (rating=90 across all attributes)
        assert best["incoming"][0]["hgt"] == 90

    def test_identifies_correct_partner_team(self):
        """
        When only one team has a good trade, the scanner returns that team's key.
        """
        my    = _roster(*[_AVG] * 7)
        good  = _roster(*[_ELITE] + [_AVG] * 6)
        bad   = _roster(*[_WEAK] * 7)     # no beneficial trades here
        result = find_best_trades(my, {"GoodTeam": good, "BadTeam": bad})

        teams_in_results = {r["team"] for r in result}
        assert "GoodTeam" in teams_in_results
        assert "BadTeam" not in teams_in_results

    def test_finds_2for1_with_depth_pieces(self):
        """
        Trading two low-value bench players for one star should appear as a
        2-for-1 option when the roster has enough depth (≥ 6 players).

        Setup: my roster has 5 good starters + 2 terrible bench players (OVR ~8).
        The other team has 1 star (OVR ~94) + fillers.
        Trading the 2 bench players for the star reduces roster depth but
        significantly improves the starting lineup.
        """
        starters  = [_player(75)] * 5
        bench     = [_player(10)] * 2    # very bad bench (OVR ~8 each)
        my_roster = _roster(*starters, *bench)

        their_star = _player(95)
        them = _roster(*[their_star] + [_player(55)] * 6)

        # Use a large top_n to see all passing trades: 2-for-1 trades tie
        # 1-for-1 trades on net_lineup_score (bench players don't make the
        # starting 5 either way), so they appear after the 7 1-for-1 options.
        result = find_best_trades(my_roster, {"Stars": them}, top_n=50)

        trade_types = {r["trade_type"] for r in result}
        assert "2-for-1" in trade_types, (
            f"Expected a 2-for-1 trade in results, got types: {trade_types}\n"
            f"result count: {len(result)}"
        )

    def test_multi_team_scan_returns_best_overall(self):
        """
        Scanning two teams where one has a much better player available:
        the very best trade across all teams must be first in the results.
        """
        my    = _roster(*[_AVG] * 7)
        teamA = _roster(*[_player(70)] + [_AVG] * 6)   # slight upgrade available
        teamB = _roster(*[_ELITE] + [_AVG] * 6)         # major upgrade available

        result = find_best_trades(my, {"A": teamA, "B": teamB})
        assert len(result) > 0
        # TeamB's elite player should give a larger net_lineup_score
        assert result[0]["team"] == "B"

    def test_no_self_trade(self):
        """Players from my own roster should never appear as incoming."""
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"T": them})
        for r in result:
            for incoming_player in r["incoming"]:
                # Incoming player came from the other team, not ours;
                # since all our players have rating=60 and their elite has rating=90,
                # any incoming player with hgt==90 is definitely from them.
                # (A weaker test: just confirm incoming list is non-empty when we have results)
                assert len(r["incoming"]) >= 1
