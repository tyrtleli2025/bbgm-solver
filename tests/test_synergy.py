"""Tests for Layer 3 synergy calculations (src/core/synergy.py)."""

import math
import pytest

from src.core.formulas import BASE_RATINGS, composite_rating
from src.core.synergy import (
    sigmoid,
    skills_count,
    calculate_lineup_synergy,
    _SKILL_PARAMS,
    _SKILL_A,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _player(val: float) -> dict:
    """Uniform player: all 15 base ratings set to *val*."""
    return {r: val for r in BASE_RATINGS}


AVG_LINEUP   = [_player(50)  for _ in range(5)]
STAR_LINEUP  = [_player(100) for _ in range(5)]
WORST_LINEUP = [_player(0)   for _ in range(5)]


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_at_cutoff_is_half(self):
        # By definition sigmoid(b, a, b) = 0.5 for any a, b
        for b in (0.57, 0.59, 0.61, 0.63, 0.68):
            assert sigmoid(b, 15, b) == pytest.approx(0.5)

    def test_well_above_cutoff_approaches_one(self):
        assert sigmoid(1.0, 15, 0.59) > 0.99

    def test_well_below_cutoff_approaches_zero(self):
        assert sigmoid(0.0, 15, 0.59) < 0.01

    def test_output_in_unit_interval(self):
        # Mathematically (0, 1) open, but float saturates to exactly 0.0 or 1.0
        # at extreme inputs — allow the closed interval.
        for x in (-10.0, 0.0, 0.5, 1.0, 10.0):
            v = sigmoid(x, 15, 0.59)
            assert 0.0 <= v <= 1.0

    def test_monotone_increasing(self):
        prev = sigmoid(-5.0, 15, 0.59)
        for x in (-3.0, 0.0, 0.59, 0.8, 2.0):
            curr = sigmoid(x, 15, 0.59)
            assert curr > prev
            prev = curr

    def test_steepness_a_affects_sharpness(self):
        # Larger a → value further from 0.5 at same offset from b
        gentle = sigmoid(0.7, 5,  0.59)
        steep  = sigmoid(0.7, 15, 0.59)
        assert steep > gentle  # both above 0.5 but steep is further


# ---------------------------------------------------------------------------
# skills_count — the critical bound invariant
# ---------------------------------------------------------------------------


class TestSkillsCount:
    def test_all_counts_bounded_zero_to_five(self):
        """Core invariant: each skill count must be in [0, 5]."""
        for name, lineup in [("avg", AVG_LINEUP), ("star", STAR_LINEUP), ("worst", WORST_LINEUP)]:
            counts = skills_count(lineup)
            for tag, val in counts.items():
                assert 0.0 <= val <= 5.0, f"[{name}] {tag}={val:.4f} out of [0, 5]"

    def test_all_eight_skill_tags_present(self):
        counts = skills_count(AVG_LINEUP)
        assert set(counts.keys()) == {"3", "A", "B", "Di", "Dp", "Po", "Ps", "R"}

    def test_star_lineup_counts_exceed_worst(self):
        star  = skills_count(STAR_LINEUP)
        worst = skills_count(WORST_LINEUP)
        for tag in star:
            assert star[tag] >= worst[tag], f"{tag}: star={star[tag]:.3f} < worst={worst[tag]:.3f}"

    def test_counts_monotone_with_rating(self):
        """Higher ratings → higher composite → higher sigmoid contribution per player."""
        low_p  = _player(20)
        high_p = _player(80)
        for tag, (composite, cutoff) in _SKILL_PARAMS.items():
            s_low  = sigmoid(composite_rating(low_p,  composite), _SKILL_A, cutoff)
            s_high = sigmoid(composite_rating(high_p, composite), _SKILL_A, cutoff)
            assert s_low <= s_high, f"{tag}: low={s_low:.4f} > high={s_high:.4f}"

    def test_single_elite_player_bounded(self):
        """Even one all-100 player among four all-0 players stays in [0, 5]."""
        mixed = [_player(100)] + [_player(0)] * 4
        counts = skills_count(mixed)
        for tag, val in counts.items():
            assert 0.0 <= val <= 5.0

    def test_missing_ratings_do_not_crash_or_escape_bounds(self):
        partial_lineup = [{"hgt": 80, "spd": 70}] * 5
        counts = skills_count(partial_lineup)
        for tag, val in counts.items():
            assert 0.0 <= val <= 5.0


# ---------------------------------------------------------------------------
# calculate_lineup_synergy
# ---------------------------------------------------------------------------


class TestLineupSynergy:
    def test_returns_correct_keys(self):
        result = calculate_lineup_synergy(AVG_LINEUP)
        assert set(result.keys()) == {"off", "def", "reb"}

    def test_all_synergies_non_negative(self):
        for name, lineup in [("avg", AVG_LINEUP), ("star", STAR_LINEUP), ("worst", WORST_LINEUP)]:
            result = calculate_lineup_synergy(lineup)
            for key, val in result.items():
                assert val >= 0.0, f"[{name}] {key}={val} is negative"

    def test_star_lineup_beats_worst_on_all_axes(self):
        star  = calculate_lineup_synergy(STAR_LINEUP)
        worst = calculate_lineup_synergy(WORST_LINEUP)
        for key in ("off", "def", "reb"):
            assert star[key] > worst[key], f"{key}: star={star[key]:.4f} <= worst={worst[key]:.4f}"

    def test_off_synergy_at_most_one(self):
        # off / 17 ≤ 1, then multiplied by ≤ 1 perim factor → cannot exceed 1
        for lineup in (AVG_LINEUP, STAR_LINEUP, WORST_LINEUP):
            assert calculate_lineup_synergy(lineup)["off"] <= 1.0 + 1e-9

    def test_def_synergy_at_most_one(self):
        # Theoretical max of raw def sum ≈ 5 < 6 divisor → always < 1
        for lineup in (AVG_LINEUP, STAR_LINEUP, WORST_LINEUP):
            assert calculate_lineup_synergy(lineup)["def"] <= 1.0 + 1e-9

    def test_reb_synergy_at_most_half(self):
        # Max raw reb sum ≈ 2 / 4 = 0.5
        for lineup in (AVG_LINEUP, STAR_LINEUP, WORST_LINEUP):
            assert calculate_lineup_synergy(lineup)["reb"] <= 0.5 + 1e-9

    def test_wrong_size_raises_value_error(self):
        with pytest.raises(ValueError, match="5 players"):
            calculate_lineup_synergy([_player(50)] * 4)
        with pytest.raises(ValueError, match="5 players"):
            calculate_lineup_synergy([_player(50)] * 6)

    def test_partial_player_dict_handled(self):
        """Lineups with incomplete base-rating dicts must not crash."""
        partial = [{"hgt": 75, "tp": 80}] * 5
        result = calculate_lineup_synergy(partial)
        assert set(result.keys()) == {"off", "def", "reb"}
        for val in result.values():
            assert math.isfinite(val)

    def test_mixed_lineup_between_star_and_worst(self):
        """Avg lineup synergy should fall between star and worst."""
        star  = calculate_lineup_synergy(STAR_LINEUP)
        worst = calculate_lineup_synergy(WORST_LINEUP)
        avg   = calculate_lineup_synergy(AVG_LINEUP)
        for key in ("off", "def", "reb"):
            assert worst[key] <= avg[key] <= star[key], (
                f"{key}: worst={worst[key]:.4f}, avg={avg[key]:.4f}, star={star[key]:.4f}"
            )
