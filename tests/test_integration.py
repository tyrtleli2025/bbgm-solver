"""
Integration tests using real league JSON fixtures.

These tests exercise the full find_best_trades pipeline against actual
BBGM league exports, comparing J-based (instantaneous lineup score) vs
V-based (horizon-aware title equity) trade evaluation.

Skipped automatically when the fixture files are not present.
"""

from __future__ import annotations

import os
import statistics
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_FIXTURE_1 = os.path.join(_REPO_ROOT, "BBGM_League_16_2002_regular_season_27-22.json")
_FIXTURE_2 = os.path.join(_REPO_ROOT, "BBGM_League_19_2016_preseason.json")

_MY_TID = 7   # Brooklyn in fixture 1


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parsed_league():
    if not os.path.exists(_FIXTURE_1):
        pytest.skip(f"League fixture not found: {_FIXTURE_1}")

    from src.core.parser import parse_league_json
    from src.core.ai_trade_value import league_value_stats

    my_df, opp_dict, cap_info = parse_league_json(_FIXTURE_1, my_tid=_MY_TID)
    all_rosters = {"__mine__": my_df, **opp_dict}
    league = league_value_stats(
        all_rosters,
        salary_cap=cap_info["salary_cap"],
        salary_cap_type=cap_info["salary_cap_type"],
        soft_cap_trade_match=cap_info["soft_cap_trade_match"],
    )
    return my_df, opp_dict, cap_info, league


# ---------------------------------------------------------------------------
# Smoke tests — both modes must complete without error
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_j_mode_returns_list(self, parsed_league):
        my_df, opp_dict, cap_info, league = parsed_league
        from src.core.market_scanner import find_best_trades

        result = find_best_trades(
            my_df, opp_dict,
            league=league, salary_cap=cap_info["salary_cap"], top_n=5,
        )
        assert isinstance(result, list)
        for t in result:
            assert t["dv"] > 0
            assert t["net_lineup_score"] > 0

    def test_v_mode_returns_list(self, parsed_league):
        my_df, opp_dict, cap_info, league = parsed_league
        from src.core.market_scanner import find_best_trades

        result = find_best_trades(
            my_df, opp_dict,
            league=league, salary_cap=cap_info["salary_cap"], top_n=5,
            use_v_function=True,
        )
        assert isinstance(result, list)
        for t in result:
            assert t["dv"] > 0
            assert t["net_lineup_score"] > 0

    def test_v_mode_sorted_descending(self, parsed_league):
        my_df, opp_dict, cap_info, league = parsed_league
        from src.core.market_scanner import find_best_trades

        result = find_best_trades(
            my_df, opp_dict,
            league=league, salary_cap=cap_info["salary_cap"], top_n=10,
            use_v_function=True,
        )
        scores = [t["net_lineup_score"] for t in result]
        assert scores == sorted(scores, reverse=True), (
            "V-mode results must be sorted by net_lineup_score descending"
        )


# ---------------------------------------------------------------------------
# Age preference test — V should value youth more than J
# ---------------------------------------------------------------------------

class TestVPrefersYouth:

    def test_v_rates_young_player_higher_than_old(self, parsed_league):
        """
        V-based ΔV for a young incoming player (age ≤ 24) must exceed the ΔV
        for the oldest player of comparable OVR on the same roster.

        This is the litmus test from CLAUDE.md: the solver must not trade away
        a 20-year-old for a 35-year-old of equal current OVR.
        """
        my_df, opp_dict, cap_info, league = parsed_league
        from src.value import LeagueState, LeagueVContext

        ls  = LeagueState.from_file(_FIXTURE_1, my_tid=_MY_TID)
        ctx = LeagueVContext(ls, H=5)

        # Collect all tradeable opponent players with OVR between 60 and 75
        from src.core.formulas import player_ovr
        from src.core.ai_trade_value import is_untradable

        current_season = ls.current_season
        candidates = []
        for _, df in opp_dict.items():
            for i in range(len(df)):
                row = df.iloc[i].to_dict()
                if is_untradable(row, current_season, False):
                    continue
                ovr = player_ovr(row)
                if 60 <= ovr <= 75:
                    candidates.append(row)

        if not candidates:
            pytest.skip("No OVR-60-75 tradeable players found in fixture")

        young = [p for p in candidates if float(p.get("age", 99)) <= 24]
        old   = [p for p in candidates if float(p.get("age", 0))  >= 33]

        if not young or not old:
            pytest.skip("Need both young (≤24) and old (≥33) OVR-60-75 players to compare")

        # Best young and old by OVR (within similar range)
        best_young = max(young, key=lambda p: player_ovr(p))
        best_old   = max(old,   key=lambda p: player_ovr(p))

        dv_young = ctx.delta_v(add_players=[best_young], remove_players=[])
        dv_old   = ctx.delta_v(add_players=[best_old],   remove_players=[])

        assert dv_young > dv_old, (
            f"V should value the young player (age={best_young.get('age')}, "
            f"OVR={player_ovr(best_young)}, ΔV={dv_young:.4f}) more than the old one "
            f"(age={best_old.get('age')}, OVR={player_ovr(best_old)}, ΔV={dv_old:.4f}). "
            f"If this fails, the aging model projection is not reaching the V function."
        )

    def test_j_vs_v_incoming_age_preference(self, parsed_league):
        """
        V-based find_best_trades should not prefer older incoming players than J.
        If both methods find trades, the mean age of V-recommended incoming players
        must be no more than 3 years older than J's mean (and ideally younger).
        """
        my_df, opp_dict, cap_info, league = parsed_league
        from src.core.market_scanner import find_best_trades

        j_trades = find_best_trades(
            my_df, opp_dict,
            league=league, salary_cap=cap_info["salary_cap"], top_n=10,
        )
        v_trades = find_best_trades(
            my_df, opp_dict,
            league=league, salary_cap=cap_info["salary_cap"], top_n=10,
            use_v_function=True,
        )

        if not j_trades or not v_trades:
            pytest.skip("No trades found — cannot compare age preferences")

        def _mean_age(trades: list[dict]) -> float:
            ages = [float(p.get("age", 30)) for t in trades for p in t["incoming"]]
            return statistics.mean(ages) if ages else 30.0

        j_age = _mean_age(j_trades)
        v_age = _mean_age(v_trades)

        assert v_age <= j_age + 3.0, (
            f"V-based trades preferred incoming players averaging {v_age:.1f} years old, "
            f"vs J-based at {j_age:.1f}. V should prefer younger or equal players "
            f"because it projects ratings forward — old players' athleticism declines sharply."
        )
