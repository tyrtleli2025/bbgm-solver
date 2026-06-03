"""
Tests for the trade engine (src/core/trade_engine.py).

Three themes:
  1. calculate_asset_value — each bonus/penalty component in isolation.
  2. evaluate_trade — correct before/after roster mechanics.
  3. evaluate_trade — directional: good trades improve scores, bad trades hurt.
"""

import math
import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS, player_ovr
from src.core.trade_engine import (
    calculate_asset_value,
    total_asset_value,
    evaluate_trade,
    POT_WEIGHT,
    AGE_PENALTY,
    CONTRACT_PENALTY,
    SALARY_SCALE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _player(**kw) -> dict:
    """All 15 base ratings default to 50; age=27 (prime), pot=0 (no upside), salary=0."""
    p = {r: 50 for r in BASE_RATINGS}
    p["age"]    = 27
    p["pot"]    = 0       # zero by default → no potential bonus
    p["salary"] = 0
    p.update(kw)
    return p


def _roster(*players) -> pd.DataFrame:
    return pd.DataFrame(list(players))


# Pre-compute the OVR for a default all-50 player (used in several assertions).
_BASE_OVR: int = player_ovr({r: 50 for r in BASE_RATINGS})

# ---------------------------------------------------------------------------
# 1. calculate_asset_value — component isolation
# ---------------------------------------------------------------------------


class TestCalculateAssetValue:

    # --- base OVR ---

    def test_base_value_tracks_ovr(self):
        elite = _player(**{r: 90 for r in BASE_RATINGS})
        poor  = _player(**{r: 20 for r in BASE_RATINGS})
        assert calculate_asset_value(elite) > calculate_asset_value(poor)

    def test_no_bonus_penalty_value_equals_ovr(self):
        """Prime age (27), zero pot, zero salary → value == OVR exactly."""
        p = _player()   # age=27, pot=0, salary=0
        assert calculate_asset_value(p) == pytest.approx(float(_BASE_OVR))

    # --- potential bonus ---

    def test_young_high_pot_worth_more_than_same_ovr_prime(self):
        prime    = _player(age=27)                              # no bonus
        prospect = _player(age=21, pot=_BASE_OVR + 20)         # big upside
        assert calculate_asset_value(prospect) > calculate_asset_value(prime)

    def test_potential_bonus_magnitude(self):
        """Exact bonus: (pot − ovr) × youth_factor × POT_WEIGHT."""
        ovr = _BASE_OVR
        age = 20
        pot = ovr + 30
        youth_factor = (26.0 - age) / 6.0   # = 1.0 at age 20
        expected_bonus = (pot - ovr) * youth_factor * POT_WEIGHT

        prime    = _player(age=27)               # no bonus baseline
        prospect = _player(age=age, pot=pot)

        diff = calculate_asset_value(prospect) - calculate_asset_value(prime)
        assert diff == pytest.approx(expected_bonus, abs=0.05)

    def test_pot_bonus_zero_at_age_26(self):
        """At age 26 youth_factor = 0 → no bonus regardless of pot."""
        no_bonus = _player(age=26, pot=_BASE_OVR + 25)
        baseline = _player(age=27)
        # Both get no potential bonus; value should equal their respective OVRs
        assert calculate_asset_value(no_bonus) == pytest.approx(float(_BASE_OVR))
        assert calculate_asset_value(baseline) == pytest.approx(float(_BASE_OVR))

    def test_pot_bonus_zero_when_pot_le_ovr(self):
        """No bonus when pot ≤ ovr even if age < 26."""
        p = _player(age=21, pot=_BASE_OVR - 5)  # pot < ovr
        assert calculate_asset_value(p) == pytest.approx(float(_BASE_OVR))

    def test_youth_factor_decays_with_age(self):
        """Older prospects get less of a bonus for the same pot gap."""
        pot = _BASE_OVR + 20
        young  = _player(age=19, pot=pot)
        older  = _player(age=24, pot=pot)
        assert calculate_asset_value(young) > calculate_asset_value(older)

    # --- age decline ---

    def test_no_penalty_below_31(self):
        """Ages 28, 29, 30 should all equal the base OVR (no penalty)."""
        for age in (28, 29, 30):
            p = _player(age=age)
            assert calculate_asset_value(p) == pytest.approx(float(_BASE_OVR)), f"age={age}"

    def test_age_penalty_exact_at_34(self):
        """Penalty at age 34: (34 - 30) × AGE_PENALTY."""
        expected_penalty = (34 - 30) * AGE_PENALTY
        prime   = _player(age=28)
        veteran = _player(age=34)
        diff = calculate_asset_value(prime) - calculate_asset_value(veteran)
        assert diff == pytest.approx(expected_penalty, abs=0.05)

    def test_age_31_penalty_equals_one_year(self):
        age_30 = _player(age=30)
        age_31 = _player(age=31)
        diff = calculate_asset_value(age_30) - calculate_asset_value(age_31)
        assert diff == pytest.approx(AGE_PENALTY, abs=0.05)

    def test_albatross_value_can_be_negative(self):
        """Old, overpaid player with low OVR can go negative."""
        albatross = _player(age=37, salary=25_000, **{r: 40 for r in BASE_RATINGS})
        assert calculate_asset_value(albatross) < 0

    # --- contract efficiency ---

    def test_fair_contract_no_penalty(self):
        """Salary at or below fair market does not reduce value."""
        ovr = _BASE_OVR
        fair_m = max(0.0, (ovr - 40.0) * 0.5)      # $M at fair market
        fair_raw = int(fair_m * SALARY_SCALE * 0.8)  # 80% of fair → definitely under

        no_contract = _player(salary=0)
        fair_deal   = _player(salary=fair_raw)
        assert calculate_asset_value(fair_deal) == pytest.approx(
            calculate_asset_value(no_contract), abs=0.01
        )

    def test_overpaid_contract_penalized(self):
        """$20M salary for an average player is a significant overpay."""
        free_agent = _player(salary=0)
        overpaid   = _player(salary=20_000)   # $20M raw → $20M @ default scale
        assert calculate_asset_value(free_agent) > calculate_asset_value(overpaid)

    def test_contract_penalty_magnitude(self):
        """Penalty scales linearly with overpay in $M × CONTRACT_PENALTY."""
        ovr = _BASE_OVR
        fair_m = max(0.0, (ovr - 40.0) * 0.5)
        overpay_m = 8.0
        raw_salary = int((fair_m + overpay_m) * SALARY_SCALE)

        baseline  = _player(salary=0)
        overpaid  = _player(salary=raw_salary)
        diff = calculate_asset_value(baseline) - calculate_asset_value(overpaid)
        assert diff == pytest.approx(overpay_m * CONTRACT_PENALTY, abs=0.1)

    # --- missing fields ---

    def test_missing_metadata_defaults_gracefully(self):
        """Player with only base ratings (no age/pot/salary) must not crash."""
        p = {r: 50 for r in BASE_RATINGS}
        val = calculate_asset_value(p)
        assert math.isfinite(val)
        # Defaults: age=27 (no penalty), pot=0→ovr (no bonus), salary=0
        assert val == pytest.approx(float(_BASE_OVR))


# ---------------------------------------------------------------------------
# 2. evaluate_trade — structure and mechanics
# ---------------------------------------------------------------------------


class TestEvaluateTradeStructure:

    def _roster8(self, rating: int = 60) -> pd.DataFrame:
        return _roster(*[_player(**{r: rating for r in BASE_RATINGS}) for _ in range(8)])

    def test_required_keys_present(self):
        roster = self._roster8()
        result = evaluate_trade(roster, [_player(**{r: 60 for r in BASE_RATINGS})], [0])
        for key in (
            "net_lineup_score", "net_asset_value",
            "old_score", "new_score",
            "old_lineup", "new_lineup",
            "incoming_value", "outgoing_value",
        ):
            assert key in result

    def test_net_lineup_score_is_new_minus_old(self):
        roster = self._roster8()
        result = evaluate_trade(roster, [_player(**{r: 60 for r in BASE_RATINGS})], [0])
        assert result["net_lineup_score"] == pytest.approx(
            result["new_score"] - result["old_score"]
        )

    def test_net_asset_value_is_incoming_minus_outgoing(self):
        roster = self._roster8()
        result = evaluate_trade(roster, [_player(**{r: 60 for r in BASE_RATINGS})], [0])
        assert result["net_asset_value"] == pytest.approx(
            result["incoming_value"] - result["outgoing_value"]
        )

    def test_even_trade_near_zero_deltas(self):
        """Swapping a player for an identical clone should produce ~0 changes."""
        clone = _player(**{r: 65 for r in BASE_RATINGS})
        roster = _roster(*[clone for _ in range(6)])
        result = evaluate_trade(roster, incoming_players=[clone], outgoing_players=[0])
        assert abs(result["net_lineup_score"]) < 0.5
        assert abs(result["net_asset_value"])  < 0.01

    def test_integer_index_outgoing(self):
        roster = self._roster8()
        result = evaluate_trade(roster, [_player(**{r: 60 for r in BASE_RATINGS})], outgoing_players=[3])
        assert len(result["new_lineup"]["lineup"]) == 5

    def test_series_outgoing_accepted(self):
        """Passing a pd.Series (roster row) as outgoing should not crash."""
        roster = self._roster8(rating=70)
        series = roster.iloc[1]
        result = evaluate_trade(roster, [_player(**{r: 70 for r in BASE_RATINGS})], [series])
        assert "net_lineup_score" in result

    def test_outgoing_index_out_of_range_raises(self):
        roster = self._roster8()
        with pytest.raises(ValueError, match="out of range"):
            evaluate_trade(roster, [], outgoing_players=[99])

    def test_post_trade_roster_too_small_raises(self):
        roster = _roster(*[_player() for _ in range(5)])
        with pytest.raises(ValueError, match="5"):
            evaluate_trade(roster, incoming_players=[], outgoing_players=[0])

    def test_old_new_lineups_are_optimize_rotation_dicts(self):
        roster = self._roster8()
        result = evaluate_trade(roster, [_player(**{r: 60 for r in BASE_RATINGS})], [0])
        for key in ("lineup", "synergy", "score", "ovr_sum"):
            assert key in result["old_lineup"]
            assert key in result["new_lineup"]


# ---------------------------------------------------------------------------
# 3. evaluate_trade — directional correctness
# ---------------------------------------------------------------------------


class TestEvaluateTradeDirectional:

    def test_acquiring_elite_player_improves_lineup_score(self):
        """
        Trading a mediocre player for an elite one must raise the lineup score.
        """
        roster  = _roster(*[_player(**{r: 55 for r in BASE_RATINGS}) for _ in range(7)])
        elite   = _player(**{r: 90 for r in BASE_RATINGS})
        result  = evaluate_trade(roster, incoming_players=[elite], outgoing_players=[0])
        assert result["net_lineup_score"] > 0

    def test_losing_elite_player_hurts_lineup_score(self):
        """
        Trading an elite player away for a weak player must lower the lineup score.
        """
        average = [_player(**{r: 55 for r in BASE_RATINGS}) for _ in range(5)]
        elite   = _player(**{r: 90 for r in BASE_RATINGS})
        weak    = _player(**{r: 30 for r in BASE_RATINGS})
        roster  = _roster(*average, elite)   # elite is at index 5
        result  = evaluate_trade(roster, incoming_players=[weak], outgoing_players=[5])
        assert result["net_lineup_score"] < 0

    def test_acquiring_young_prospect_improves_asset_value(self):
        """
        A young player with high potential has a larger asset value than a
        same-OVR, same-ratings player of prime age with no upside.
        """
        plain_player = _player(**{r: 55 for r in BASE_RATINGS}, age=27, pot=0)
        roster  = _roster(*[plain_player for _ in range(7)])
        # Incoming: same base ratings, but age=20 and high pot
        ovr_55_ratings = {r: 55 for r in BASE_RATINGS}
        prospect = _player(**ovr_55_ratings, age=20, pot=90)
        result = evaluate_trade(roster, incoming_players=[prospect], outgoing_players=[0])
        assert result["net_asset_value"] > 0

    def test_trading_for_overpaid_veteran_hurts_asset_value(self):
        """
        Acquiring an old, overpaid player should reduce total asset value even
        if their base OVR is similar to the player leaving.
        """
        prime   = _player(**{r: 65 for r in BASE_RATINGS}, age=27, salary=0)
        albatross = _player(**{r: 65 for r in BASE_RATINGS}, age=35, salary=30_000)
        roster  = _roster(*[prime for _ in range(7)])
        result  = evaluate_trade(roster, incoming_players=[albatross], outgoing_players=[0])
        assert result["net_asset_value"] < 0

    def test_total_asset_value_reflects_roster_quality(self):
        """A roster of elite players should have higher total value than a weak roster."""
        elite_roster = _roster(*[_player(**{r: 85 for r in BASE_RATINGS}) for _ in range(10)])
        weak_roster  = _roster(*[_player(**{r: 35 for r in BASE_RATINGS}) for _ in range(10)])
        assert total_asset_value(elite_roster) > total_asset_value(weak_roster)
