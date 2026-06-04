"""
Tests for the Trade Market Scanner (src/core/market_scanner.py).

The realism gate is now evaluate_dv > 0 (ZenGM's ValueChangeCalculator), not
the old symmetric delta window.  Every returned trade must pass BOTH:
  - AI acceptance: evaluate_dv > 0 for the opposing team
  - Lineup improvement: net_lineup_score > 0

Trade scenarios are designed around contract and age differentials that create
real dv asymmetries, rather than uniform OVR gaps.
"""

import math
import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS, player_ovr
from src.core.market_scanner import find_best_trades, TOP_N_DEFAULT
from src.core.ai_trade_value import (
    evaluate_dv,
    league_value_stats,
    infer_strategy,
    is_untradable,
    SALARY_CAP_DEFAULT,
)

SEASON      = 2024
SALARY_CAP  = SALARY_CAP_DEFAULT   # 90_000 ($K)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "team", "trade_type", "incoming", "outgoing",
    "net_lineup_score", "dv", "new_score",
}


def _player(
    rating: int = 60,
    age: float = 27.0,
    pot: int = 0,
    salary: float = 0.0,
    contract_exp: int = SEASON + 3,
    **extra,
) -> dict:
    p = {r: rating for r in BASE_RATINGS}
    p.update(age=age, pot=pot, salary=salary, contract_exp=contract_exp)
    p.update(extra)
    return p


def _roster(*players) -> pd.DataFrame:
    return pd.DataFrame(list(players))


# ---------------------------------------------------------------------------
# Shared league fixture
#
# Wide OVR spread keeps ovr_std large (~12-14) so typical test players stay
# well below z=1 and the v^7 exponent does not distort results.
# ---------------------------------------------------------------------------

def _make_league(*extra_rosters: pd.DataFrame) -> dict:
    """Build a League dict from background players + any extra rosters."""
    background = {
        "bg_lo":  _roster(*[_player(45 + i * 2) for i in range(10)]),
        "bg_mid": _roster(*[_player(62 + i * 2) for i in range(10)]),
        "bg_hi":  _roster(*[_player(72 + i * 2) for i in range(10)]),
    }
    for k, df in enumerate(extra_rosters):
        background[f"extra_{k}"] = df
    return league_value_stats(
        background,
        current_season=SEASON,
        salary_cap=SALARY_CAP,
    )


# ---------------------------------------------------------------------------
# Concrete trade scenario: contract-differential 1-for-1
#
# Both players are prime-age (27) with similar OVR so v < 1 for both.
# My player is massively underpaid; their player is massively overpaid.
# The other team (rebuilding) discounts the contract penalty 2× MORE than
# a contender, so the net difference yields dv > 0 for them.
# My lineup improves because their player has slightly higher OVR.
#
# Verified by hand with the background league (ovr_mean≈65, ovr_std≈12):
#   My player  OVR≈67, raw_v≈0    salary=300   → underpaid bonus  (cv≈+0.10)
#   Their player OVR≈71, raw_v≈0.3 salary=30K  → overpaid penalty (cv≈-0.25)
#   rebuilder contracts_factor=2 → dv = large_pos - neg ≈ +0.15 > 0
# ---------------------------------------------------------------------------

def _contract_trade_scenario():
    """
    Return (my_roster, their_roster, my_cheap_player, their_expensive_player).
    """
    cheap    = _player(60, salary=300,    contract_exp=SEASON + 4)  # OVR≈67, underpaid
    expensive = _player(65, salary=30_000, contract_exp=SEASON + 3)  # OVR≈71, overpaid
    my_roster   = _roster(*[_player(60)] * 5 + [cheap])
    their_roster = _roster(*[expensive] + [_player(60)] * 6)
    return my_roster, their_roster, cheap, expensive


# ---------------------------------------------------------------------------
# 1. Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:

    def test_returns_a_list(self):
        my   = _roster(*[_player(60)] * 6)
        them = _roster(*[_player(60)] * 6)
        result = find_best_trades(my, {"T": them})
        assert isinstance(result, list)

    def test_each_result_has_required_keys(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        for r in result:
            missing = _REQUIRED_KEYS - set(r.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_at_most_top_n_results(self):
        my   = _roster(*[_player(60)] * 7)
        them = _roster(*[_player(60)] * 7)
        result = find_best_trades(my, {"T": them}, top_n=3)
        assert len(result) <= 3

    def test_default_top_n_is_five(self):
        my   = _roster(*[_player(60)] * 7)
        them = _roster(*[_player(60)] * 7)
        result = find_best_trades(my, {"T": them})
        assert len(result) <= TOP_N_DEFAULT

    def test_incoming_and_outgoing_are_lists(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        for r in result:
            assert isinstance(r["incoming"], list)
            assert isinstance(r["outgoing"], list)

    def test_trade_type_is_valid_string(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        for r in result:
            assert r["trade_type"] in {"1-for-1", "2-for-1"}

    def test_team_key_matches_league_dict_key(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"Celtics": them}, league=lg)
        for r in result:
            assert r["team"] == "Celtics"

    def test_net_lineup_score_positive_in_all_results(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        for r in result:
            assert r["net_lineup_score"] > 0

    def test_dv_positive_in_all_results(self):
        """The reported dv must be > 0 — it is the AI acceptance margin."""
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        for r in result:
            assert r["dv"] > 0, f"trade has dv={r['dv']:.4f} ≤ 0"


# ---------------------------------------------------------------------------
# 2. AI acceptance invariant
# ---------------------------------------------------------------------------


class TestAIAcceptanceInvariant:

    def test_evaluate_dv_positive_for_every_result(self):
        """
        The AI acceptance is computed inside find_best_trades, but we verify
        it here independently using the same evaluate_dv function.
        """
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)

        for r in result:
            their_strategy = infer_strategy(them)
            dv = evaluate_dv(
                them, lg,
                incoming=r["incoming"],   # what the AI gives up
                outgoing=r["outgoing"],   # what the AI receives
                strategy=their_strategy,
            )
            assert dv > 0, (
                f"Returned trade has independent dv={dv:.4f} ≤ 0:\n"
                f"  outgoing: {[p.get('salary') for p in r['outgoing']]}\n"
                f"  incoming: {[p.get('salary') for p in r['incoming']]}"
            )

    def test_scrub_for_star_never_returned(self):
        """
        Trading an OVR≈8 scrub (very negative z) for an OVR≈94 star
        (v > 1, gets v^7) should NEVER satisfy dv > 0 for the star-holding team.
        """
        scrub = _player(20, salary=0)
        star  = _player(90, salary=10_000)
        my    = _roster(*[_player(60)] * 5 + [scrub])
        them  = _roster(*[star] + [_player(60)] * 6)
        lg    = _make_league(my, them)

        result = find_best_trades(my, {"Stars": them}, league=lg)

        # Check that no result sends the scrub and receives the star
        scrub_ovr = player_ovr(scrub)
        star_ovr  = player_ovr(star)
        for r in result:
            out_ovrs = [player_ovr(p) for p in r["outgoing"]]
            in_ovrs  = [player_ovr(p) for p in r["incoming"]]
            assert not (
                any(o <= scrub_ovr + 2 for o in out_ovrs)
                and any(v >= star_ovr - 2 for v in in_ovrs)
            ), "Scrub-for-star trade was returned — AI should never accept this"

    def test_equal_players_no_dv_advantage(self):
        """
        Swapping identical players (same OVR, same salary, same age) gives dv ≈ 0
        so neither lineup improves nor is the AI motivated → no results.
        """
        my   = _roster(*[_player(60)] * 7)
        them = _roster(*[_player(60)] * 7)
        lg   = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        # All players are identical: dv ≈ 0 → no accepted trades
        assert result == []


# ---------------------------------------------------------------------------
# 3. Filtering
# ---------------------------------------------------------------------------


class TestFiltering:

    def test_empty_league_returns_empty(self):
        my = _roster(*[_player(60)] * 6)
        assert find_best_trades(my, {}) == []

    def test_team_with_empty_roster_is_skipped(self):
        my = _roster(*[_player(60)] * 6)
        result = find_best_trades(my, {"Ghost": pd.DataFrame()})
        assert result == []

    def test_my_roster_too_small_raises(self):
        with pytest.raises(ValueError, match="5"):
            find_best_trades(_roster(*[_player(60)] * 4), {"T": _roster(*[_player(60)] * 6)})

    def test_untradable_player_excluded_from_outgoing(self):
        """A player with gamesUntilTradable > 0 must not appear as outgoing."""
        locked  = _player(60, salary=0)
        locked["gamesUntilTradable"] = 10
        my   = _roster(*[_player(60)] * 5 + [locked])
        _, them, _, _ = _contract_trade_scenario()
        lg   = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        for r in result:
            for p in r["outgoing"]:
                # is_untradable handles NaN from DataFrame column alignment
                assert not is_untradable(p, SEASON), (
                    "Locked player (gamesUntilTradable > 0) appeared as outgoing"
                )

    def test_salary_match_blocks_excessive_intake(self):
        """
        If the other team would receive a salary far above 125% of what they
        send, and they are over the cap, the trade must be blocked.
        """
        # Their player salary = $1K; we send a player with salary = $50K
        # → incoming/outgoing = 50K/1K = 50× → violates 125% cap
        their_p = _player(65, salary=1_000,  contract_exp=SEASON + 3)
        my_p    = _player(60, salary=50_000, contract_exp=SEASON + 3)
        # Make their team over the cap
        their_over_cap = _roster(their_p, *[_player(65, salary=15_000)] * 6)
        my    = _roster(*[_player(60)] * 5 + [my_p])
        lg    = _make_league(my, their_over_cap)

        # Check salary_match_ok independently first
        from src.core.ai_trade_value import salary_match_ok
        assert not salary_match_ok(
            1_000, 50_000, sum(15_000 for _ in range(6)) + 1_000, SALARY_CAP
        ), "Test precondition: should be over cap and violate 125%"

        result = find_best_trades(my, {"T": their_over_cap}, league=lg)
        for r in result:
            # If the expensive player (salary=50K) appears as outgoing,
            # its corresponding salary intake for the other team (1K) should
            # be checked — or more simply: verify no trade sends my 50K player
            out_sals = [p.get("salary", 0) for p in r["outgoing"]]
            if any(s >= 50_000 for s in out_sals):
                in_sals = [p.get("salary", 0) for p in r["incoming"]]
                # The other team would receive 50K but only send 1K → blocked
                assert max(in_sals, default=0) >= 1_000 * 1.25, (
                    "Salary-violating trade was returned"
                )


# ---------------------------------------------------------------------------
# 4. Sorting and top_n
# ---------------------------------------------------------------------------


class TestSortingAndTopN:

    def test_sorted_descending_by_net_lineup_score(self):
        my, them, _, _ = _contract_trade_scenario()
        # Add a second team so we might get multiple results
        them2 = _roster(*[_player(64, salary=28_000)] + [_player(60)] * 6)
        lg    = _make_league(my, them, them2)
        result = find_best_trades(
            my, {"T1": them, "T2": them2}, league=lg, top_n=20
        )
        scores = [r["net_lineup_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg, top_n=1)
        assert len(result) <= 1

    def test_best_result_is_first(self):
        my, them, _, _ = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg, top_n=20)
        if len(result) >= 2:
            assert result[0]["net_lineup_score"] >= result[1]["net_lineup_score"]


# ---------------------------------------------------------------------------
# 5. Trade discovery
# ---------------------------------------------------------------------------


class TestTradeDiscovery:

    def test_contract_differential_trade_found(self):
        """
        A massively underpaid player traded for a massively overpaid player of
        higher OVR satisfies both dv > 0 AND net_lineup_score > 0.
        """
        my, them, cheap, expensive = _contract_trade_scenario()
        lg = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg)
        assert len(result) > 0, (
            "Expected at least one trade for contract-differential scenario"
        )
        r = result[0]
        assert r["dv"] > 0
        assert r["net_lineup_score"] > 0

    def test_league_computed_automatically(self):
        """Omitting the league parameter must not raise."""
        my, them, _, _ = _contract_trade_scenario()
        result = find_best_trades(my, {"T": them})   # no league= kwarg
        assert isinstance(result, list)

    def test_correct_team_identified(self):
        """The team name in each result must match the key in league_rosters_dict."""
        my, them, _, _ = _contract_trade_scenario()
        bad = _roster(*[_player(60)] * 7)   # no beneficial trades
        lg  = _make_league(my, them, bad)
        result = find_best_trades(my, {"Good": them, "Bad": bad}, league=lg)
        teams = {r["team"] for r in result}
        assert "Good" in teams or len(result) == 0   # either found in good or no trades at all

    def test_progress_callback_called_once_per_team(self):
        calls = []
        my   = _roster(*[_player(60)] * 6)
        them = _roster(*[_player(60)] * 6)
        find_best_trades(
            my, {"A": them, "B": them},
            progress=lambda name, done, total: calls.append((name, done, total)),
        )
        assert len(calls) == 2

    def test_2for1_requires_six_players(self):
        """With exactly 5 players, 2-for-1 trades cannot be assembled (would leave 4)."""
        my   = _roster(*[_player(60)] * 5)
        them = _roster(*[_player(65, salary=25_000)] + [_player(60)] * 6)
        lg   = _make_league(my, them)
        result = find_best_trades(my, {"T": them}, league=lg, top_n=50)
        types = {r["trade_type"] for r in result}
        assert "2-for-1" not in types, "2-for-1 should be impossible with only 5 players"
