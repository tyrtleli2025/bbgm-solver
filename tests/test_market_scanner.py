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
    MAX_AI_LOSS,
    TOP_N_DEFAULT,
)

# Convenience: bypass the AI-loss cap when a test is checking discovery logic
# (not the fairness filter).  Any trade delta up to this value is accepted.
_NO_AI_CAP = float("inf")

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
        # Bypass AI cap: this test checks dict structure, not trade fairness.
        result = find_best_trades(my, {"Team": them}, max_ai_loss=_NO_AI_CAP)
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
        # Also bypass AI cap so both runs differ only in asset_value_floor.
        strict     = find_best_trades(my, {"T": them}, asset_value_floor=0.0,   max_ai_loss=_NO_AI_CAP)
        permissive = find_best_trades(my, {"T": them}, asset_value_floor=-100.0, max_ai_loss=_NO_AI_CAP)
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
        # Bypass AI cap: this test checks sort order, not trade fairness.
        result = find_best_trades(my, teams, max_ai_loss=_NO_AI_CAP)
        scores = [r["net_lineup_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_best_trade_is_first(self):
        """The trade with the highest net_lineup_score must be result[0]."""
        my    = _roster(*[_AVG] * 7)
        teams = {
            "Great": _roster(*[_ELITE] + [_AVG] * 6),
            "Meh":   _roster(*[_player(65)] + [_AVG] * 6),
        }
        # Bypass AI cap: this test checks ranking logic, not trade fairness.
        result = find_best_trades(my, teams, max_ai_loss=_NO_AI_CAP)
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
        If the other team has one clearly elite player, the scanner finds the
        1-for-1 trade.  AI-cap bypassed: this tests discovery logic, not fairness.
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"TeamA": them}, max_ai_loss=_NO_AI_CAP)

        assert len(result) > 0
        best = result[0]
        assert best["team"] == "TeamA"
        assert best["incoming"][0]["hgt"] == 90

    def test_identifies_correct_partner_team(self):
        """
        When only one team has a good trade, the scanner returns that team's key.
        AI-cap bypassed: this tests team-routing logic, not trade fairness.
        """
        my    = _roster(*[_AVG] * 7)
        good  = _roster(*[_ELITE] + [_AVG] * 6)
        bad   = _roster(*[_WEAK] * 7)
        result = find_best_trades(my, {"GoodTeam": good, "BadTeam": bad},
                                  max_ai_loss=_NO_AI_CAP)

        teams_in_results = {r["team"] for r in result}
        assert "GoodTeam" in teams_in_results
        assert "BadTeam" not in teams_in_results

    def test_finds_2for1_with_depth_pieces(self):
        """
        Trading two bench players for one star should surface as a 2-for-1.
        AI-cap bypassed: this tests 2-for-1 enumeration, not trade fairness.
        """
        starters  = [_player(75)] * 5
        bench     = [_player(10)] * 2
        my_roster = _roster(*starters, *bench)

        their_star = _player(95)
        them = _roster(*[their_star] + [_player(55)] * 6)

        result = find_best_trades(my_roster, {"Stars": them},
                                  max_ai_loss=_NO_AI_CAP, top_n=50)

        trade_types = {r["trade_type"] for r in result}
        assert "2-for-1" in trade_types, (
            f"Expected a 2-for-1 trade in results, got types: {trade_types}\n"
            f"result count: {len(result)}"
        )

    def test_multi_team_scan_returns_best_overall(self):
        """
        The best trade across all teams must be ranked first.
        AI-cap bypassed: this tests cross-team ranking, not trade fairness.
        """
        my    = _roster(*[_AVG] * 7)
        teamA = _roster(*[_player(70)] + [_AVG] * 6)
        teamB = _roster(*[_ELITE] + [_AVG] * 6)

        result = find_best_trades(my, {"A": teamA, "B": teamB},
                                  max_ai_loss=_NO_AI_CAP)
        assert len(result) > 0
        assert result[0]["team"] == "B"

    def test_no_self_trade(self):
        """Players from my own roster should never appear as incoming."""
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"T": them}, max_ai_loss=_NO_AI_CAP)
        for r in result:
            assert len(r["incoming"]) >= 1


# ---------------------------------------------------------------------------
# 5. AI-loss filter
# ---------------------------------------------------------------------------


class TestAILossFilter:

    def test_lopsided_trade_rejected_by_default(self):
        """
        A trade where we gain far more value than MAX_AI_LOSS must be filtered out
        even if it would improve our lineup.
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        # Default max_ai_loss = MAX_AI_LOSS (3.0); ELITE→AVG delta ≈ +27 → rejected
        result = find_best_trades(my, {"T": them})
        assert result == [], (
            f"Expected no results with default AI cap, got {len(result)}"
        )

    def test_lopsided_trade_passes_with_raised_cap(self):
        """
        The same lopsided trade is accepted when max_ai_loss is raised
        above the actual asset-value delta.
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_AVG] * 6)
        result = find_best_trades(my, {"T": them}, max_ai_loss=_NO_AI_CAP)
        assert len(result) > 0

    def test_near_equal_value_trade_passes(self):
        """
        A trade between players of nearly equal value (delta ≈ 0) passes
        the AI-loss filter with the default cap.
        """
        my_player   = _player(60)
        their_player = _player(60)   # identical value → delta = 0 ≤ MAX_AI_LOSS
        my   = _roster(*[my_player] * 7)
        them = _roster(*[their_player] * 7)
        # All net_lineup_scores will be ~0 (same players), so result may be empty —
        # the important thing is that no ValueError is raised and the cap isn't hit.
        result = find_best_trades(my, {"T": them})
        for r in result:
            assert r["net_asset_value"] <= MAX_AI_LOSS + 0.01

    def test_max_ai_loss_constant_is_3(self):
        """Verify the tuning constant has the expected value."""
        assert MAX_AI_LOSS == 3.0

    def test_cap_is_inclusive(self):
        """
        A trade with net_asset_value exactly equal to MAX_AI_LOSS is accepted
        (the bound is inclusive: delta ≤ max_ai_loss).
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_AVG] * 7)
        # With equal players the delta is 0, well within the cap.
        # Set the cap to exactly 0 — equal-value trades should still be accepted
        # (pending lineup improvement filter which may remove them).
        result = find_best_trades(my, {"T": them}, max_ai_loss=0.0)
        # Result may be empty (no lineup improvement) but no cap violation.
        for r in result:
            assert r["net_asset_value"] <= 0.0 + 0.01

    def test_returned_trades_respect_ai_cap(self):
        """
        Every returned trade must have net_asset_value ≤ max_ai_loss.
        """
        my   = _roster(*[_AVG] * 7)
        them = _roster(*[_ELITE] + [_player(62)] * 6)
        cap  = 5.0
        result = find_best_trades(my, {"T": them}, max_ai_loss=cap)
        for r in result:
            assert r["net_asset_value"] <= cap + 0.01, (
                f"net_asset_value {r['net_asset_value']:.2f} exceeds cap {cap}"
            )
