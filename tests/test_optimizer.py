"""
Tests for the lineup optimizer (src/core/optimizer.py).

Two themes:
  1. Structural correctness — return shape, types, edge cases.
  2. Synergy-aware selection — the optimizer must prefer perimeter-skilled
     lineups because the perimFactor in offensive synergy heavily punishes
     lineups without shooters / passers / ball-handlers.
"""

import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS, player_ovr
from src.core.optimizer import optimize_rotation, lineup_score, OFF_WEIGHT, DEF_WEIGHT
from src.core.synergy import calculate_lineup_synergy

# ---------------------------------------------------------------------------
# Shared player archetypes
# ---------------------------------------------------------------------------

def _player(**overrides) -> dict:
    """All base ratings default to 50; apply overrides on top."""
    p = {r: 50 for r in BASE_RATINGS}
    p.update(overrides)
    return p


def _roster(*players) -> pd.DataFrame:
    return pd.DataFrame(list(players))


# Perimeter specialist: elite shooter + passer + ball-handler.
# High tp/drb/pss/spd push count_3, count_B, count_Ps to near-5,
# which maximises the perimFactor and therefore offensive synergy.
PERIMETER = _player(tp=90, drb=85, pss=85, spd=80, oiq=80)

# Interior big: great at interior skills but low on every perimeter rating.
# Including even one of these collapses the perimFactor significantly.
POST = _player(hgt=90, stre=85, ins=80, diq=80, reb=80,
               tp=10, drb=10, pss=10, spd=35)


# ---------------------------------------------------------------------------
# 1. Structural correctness
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_required_keys_present(self):
        roster = _roster(*[_player() for _ in range(6)])
        result = optimize_rotation(roster)
        assert {"lineup", "synergy", "score", "ovr_sum"} <= set(result.keys())

    def test_lineup_is_exactly_five_rows(self):
        roster = _roster(*[_player() for _ in range(8)])
        result = optimize_rotation(roster)
        assert len(result["lineup"]) == 5

    def test_lineup_has_base_rating_columns(self):
        roster = _roster(*[_player() for _ in range(6)])
        lineup = optimize_rotation(roster)["lineup"]
        for col in BASE_RATINGS:
            assert col in lineup.columns

    def test_synergy_keys(self):
        roster = _roster(*[_player() for _ in range(6)])
        syn = optimize_rotation(roster)["synergy"]
        assert set(syn.keys()) == {"off", "def", "reb"}

    def test_score_is_finite_and_positive(self):
        roster = _roster(*[_player() for _ in range(6)])
        score = optimize_rotation(roster)["score"]
        import math
        assert math.isfinite(score)
        assert score > 0

    def test_ovr_sum_matches_lineup(self):
        roster = _roster(*[_player() for _ in range(6)])
        result = optimize_rotation(roster)
        expected = sum(player_ovr(result["lineup"].iloc[i]) for i in range(5))
        assert result["ovr_sum"] == expected

    def test_score_equals_formula(self):
        """score == ovr_sum + OFF_WEIGHT * off + DEF_WEIGHT * def."""
        roster = _roster(*[_player() for _ in range(6)])
        result = optimize_rotation(roster)
        expected = (
            result["ovr_sum"]
            + OFF_WEIGHT * result["synergy"]["off"]
            + DEF_WEIGHT * result["synergy"]["def"]
        )
        assert result["score"] == pytest.approx(expected)

    def test_score_exceeds_bare_ovr_sum(self):
        """Synergy adds a non-negative bonus so score >= ovr_sum always."""
        roster = _roster(*[_player() for _ in range(6)])
        result = optimize_rotation(roster)
        assert result["score"] >= result["ovr_sum"]

    def test_exactly_five_player_roster(self):
        """Only one combination exists — must still work."""
        roster = _roster(*[_player() for _ in range(5)])
        result = optimize_rotation(roster)
        assert len(result["lineup"]) == 5

    def test_fewer_than_five_raises(self):
        roster = _roster(*[_player() for _ in range(4)])
        with pytest.raises(ValueError, match="5 players"):
            optimize_rotation(roster)

    def test_extra_columns_are_ignored(self):
        """Non-rating columns (e.g. player name) must not cause errors."""
        players = [_player() for _ in range(6)]
        for i, p in enumerate(players):
            p["name"] = f"Player {i}"
        roster = pd.DataFrame(players)
        result = optimize_rotation(roster)
        assert len(result["lineup"]) == 5


# ---------------------------------------------------------------------------
# 2. Weak-player exclusion
# ---------------------------------------------------------------------------

class TestWeakPlayerExclusion:
    def test_all_zero_player_is_excluded(self):
        """An all-0 player has the lowest possible OVR and should never appear."""
        good   = [_player(**{r: 80 for r in BASE_RATINGS}) for _ in range(5)]
        weak   = _player(**{r: 0  for r in BASE_RATINGS})
        roster = _roster(*good, weak)
        result = optimize_rotation(roster)
        # The weak player has all ratings = 0; the good players all have 80.
        # Confirm no selected player has, e.g., tp == 0 when the good players
        # have tp == 80.
        assert all(result["lineup"]["tp"] == 80)

    def test_optimal_score_is_better_than_lineup_with_weak_player(self):
        """The optimizer's score must beat any lineup that includes the weak player."""
        good   = [_player(**{r: 80 for r in BASE_RATINGS}) for _ in range(5)]
        weak   = _player(**{r: 0  for r in BASE_RATINGS})
        roster = _roster(*good, weak)
        result = optimize_rotation(roster)

        # Manually score any lineup that includes the weak player (index 5)
        weak_row   = roster.iloc[5]
        forced_bad = [roster.iloc[i] for i in range(4)] + [weak_row]
        bad_result = lineup_score(forced_bad)

        assert result["score"] > bad_result["score"]


# ---------------------------------------------------------------------------
# 3. Synergy-aware selection
# ---------------------------------------------------------------------------

class TestSynergyAwareSelection:
    def test_perimeter_lineup_is_selected_over_mixed(self):
        """
        Roster of 5 perimeter players + 1 post player.
        The all-perimeter lineup wins because:
          (a) Perimeter players have higher OVRs (tp/drb/pss hit high-coefficient ratings).
          (b) perimFactor rewards shooters + passers + ball-handlers, multiplying off synergy.
        """
        roster  = _roster(PERIMETER, PERIMETER, PERIMETER, PERIMETER, PERIMETER, POST)
        result  = optimize_rotation(roster)
        lineup  = result["lineup"]

        # Every selected player must have tp=90 (a perimeter player attribute).
        # The POST player has tp=10, so its presence is detectable.
        assert all(lineup["tp"] == 90), (
            f"POST player (tp=10) was included in the optimal lineup:\n"
            f"{lineup[['tp', 'drb', 'pss']].to_string()}"
        )

    def test_all_post_lineup_severely_penalized_by_perim_factor(self):
        """
        Five interior-only players collapse count_3/count_B/count_Ps to ~0,
        driving perimFactor to ~0 and cutting off synergy by more than 5×.
        This is the engine's explicit punishment for lineups with no perimeter
        skills (the reference: 'A team of five non-shooters is severely punished').
        """
        syn_all_perim = calculate_lineup_synergy([PERIMETER] * 5)
        syn_all_post  = calculate_lineup_synergy([POST] * 5)
        assert syn_all_perim["off"] > syn_all_post["off"] * 5, (
            f"Expected all-perimeter off >> all-post off: "
            f"{syn_all_perim['off']:.4f} vs {syn_all_post['off']:.4f}"
        )

    def test_one_athletic_post_player_does_not_hurt_off_synergy(self):
        """
        Replacing one perimeter player with a high-athleticism post player can
        raise count_A (athleticism) and count_Po (post play), *increasing* off
        synergy — provided the remaining four players keep perimFactor clamped
        at 1.0.  Documents surprising-but-correct emergent engine behaviour.
        """
        syn_all_perim = calculate_lineup_synergy([PERIMETER] * 5)
        syn_mixed     = calculate_lineup_synergy([PERIMETER] * 4 + [POST])
        # Mixed lineup off synergy must be at least 85% of the all-perimeter
        # baseline (in practice it is slightly higher due to POST athleticism).
        assert syn_mixed["off"] >= syn_all_perim["off"] * 0.85, (
            f"Mixed off={syn_mixed['off']:.4f} dropped unexpectedly below "
            f"85% of perimeter baseline {syn_all_perim['off']:.4f}"
        )

    def test_optimizer_score_all_perimeter_beats_all_mixed_combos(self):
        """
        For a roster of 5P + 1 POST, there are C(6,5)=6 lineups.
        The one with all 5 perimeter players must beat every other combination.
        """
        roster = _roster(PERIMETER, PERIMETER, PERIMETER, PERIMETER, PERIMETER, POST)
        result = optimize_rotation(roster)
        optimal_score = result["score"]

        import itertools
        for combo in itertools.combinations(range(6), 5):
            if 5 in combo:   # this combo includes the POST player (index 5)
                players = [roster.iloc[i] for i in combo]
                s = lineup_score(players)
                assert optimal_score >= s["score"], (
                    f"Combo {combo} score={s['score']:.2f} beats optimal {optimal_score:.2f}"
                )

    def test_synergy_bonus_is_positive_for_perimeter_lineup(self):
        """The perimeter lineup's off synergy bonus must be non-trivial."""
        roster = _roster(*[PERIMETER] * 5 + [POST])
        result = optimize_rotation(roster)
        synergy_bonus = OFF_WEIGHT * result["synergy"]["off"] + DEF_WEIGHT * result["synergy"]["def"]
        assert synergy_bonus > 1.0, f"Synergy bonus={synergy_bonus:.3f} unexpectedly low"


# ---------------------------------------------------------------------------
# 4. lineup_score helper
# ---------------------------------------------------------------------------

class TestLineupScore:
    def test_returns_correct_keys(self):
        players = [_player() for _ in range(5)]
        result  = lineup_score(players)
        assert {"score", "ovr_sum", "synergy"} <= set(result.keys())

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="5 players"):
            lineup_score([_player()] * 4)

    def test_score_formula(self):
        players = [_player() for _ in range(5)]
        result  = lineup_score(players)
        expected = (
            result["ovr_sum"]
            + OFF_WEIGHT * result["synergy"]["off"]
            + DEF_WEIGHT * result["synergy"]["def"]
        )
        assert result["score"] == pytest.approx(expected)

    def test_higher_rated_lineup_scores_higher(self):
        low_lineup  = lineup_score([_player(**{r: 30 for r in BASE_RATINGS})] * 5)
        high_lineup = lineup_score([_player(**{r: 80 for r in BASE_RATINGS})] * 5)
        assert high_lineup["score"] > low_lineup["score"]
