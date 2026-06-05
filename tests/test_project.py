"""
Tests for src/project.py — deterministic player rating projection.
"""

import math
import pytest

from src.core.formulas import BASE_RATINGS
from src.project import (
    project_ratings,
    project_ovr,
    project_contract_status,
    estimate_next_contract,
    _base_change_expected,
    _age_modifier,
    _change_limits,
)

CURRENT_SEASON = 2024


def _uniform_player(rating: int = 60) -> dict:
    return {r: float(rating) for r in BASE_RATINGS}


# ---------------------------------------------------------------------------
# Base change table
# ---------------------------------------------------------------------------

class TestBaseChange:
    def test_young_positive(self):
        assert _base_change_expected(19) == pytest.approx(2.0)
        assert _base_change_expected(21) == pytest.approx(2.0)

    def test_prime_zero(self):
        assert _base_change_expected(26) == pytest.approx(0.0)
        assert _base_change_expected(27) == pytest.approx(0.0)

    def test_veteran_negative(self):
        assert _base_change_expected(32) < 0
        assert _base_change_expected(38) < 0

    def test_monotone_decline_after_27(self):
        ages = [27, 29, 31, 33, 37, 42]
        changes = [_base_change_expected(a) for a in ages]
        for i in range(len(changes) - 1):
            assert changes[i] >= changes[i + 1]


# ---------------------------------------------------------------------------
# Age modifiers
# ---------------------------------------------------------------------------

class TestAgeModifier:
    def test_shooting_durable_at_35(self):
        """Shooting modifier should counteract the negative base change."""
        for r in ("ft", "fg", "tp"):
            assert _age_modifier(r, 35) == pytest.approx(2.0)

    def test_iq_fast_growth_when_young(self):
        assert _age_modifier("oiq", 20) == pytest.approx(4.0)
        assert _age_modifier("diq", 22) == pytest.approx(3.0)

    def test_speed_steep_decline_after_30(self):
        assert _age_modifier("spd", 31) == pytest.approx(-3.0)
        assert _age_modifier("spd", 38) == pytest.approx(-4.0)

    def test_jumping_worst_aging_curve(self):
        assert _age_modifier("jmp", 27) == pytest.approx(-3.0)
        assert _age_modifier("jmp", 35) == pytest.approx(-4.0)

    def test_strength_never_modified(self):
        for age in (20, 30, 40):
            assert _age_modifier("stre", age) == pytest.approx(0.0)

    def test_height_never_modified(self):
        for age in (18, 25, 40):
            assert _age_modifier("hgt", age) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Change limits
# ---------------------------------------------------------------------------

class TestChangeLimits:
    def test_height_frozen(self):
        lo, hi = _change_limits("hgt", 25)
        assert lo == hi == 0.0

    def test_iq_high_ceiling_for_young(self):
        _, hi = _change_limits("oiq", 19)
        assert hi == pytest.approx(32.0)
        _, hi2 = _change_limits("diq", 24)
        assert hi2 == pytest.approx(9.0)

    def test_speed_tight_upside(self):
        lo, hi = _change_limits("spd", 25)
        assert hi == pytest.approx(2.0)
        assert lo == pytest.approx(-12.0)

    def test_skill_tight_range(self):
        lo, hi = _change_limits("drb", 27)
        assert lo == pytest.approx(-2.0)
        assert hi == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# project_ratings trajectories
# ---------------------------------------------------------------------------

class TestProjectRatings:

    def test_young_player_overall_improves(self):
        """A 19-year-old with average ratings should have higher OVR after 3 years."""
        p = _uniform_player(55)
        age = 19.0
        ovr_0 = project_ovr(p)
        ovr_3 = project_ovr(project_ratings(p, age, 3))
        assert ovr_3 > ovr_0, f"Young player should improve: {ovr_0:.1f} → {ovr_3:.1f}"

    def test_prime_player_mostly_stable(self):
        """A 26-year-old should change very little over 2 years."""
        p = _uniform_player(70)
        ovr_0 = project_ovr(p)
        ovr_2 = project_ovr(project_ratings(p, 26.0, 2))
        assert abs(ovr_2 - ovr_0) < 5, f"Prime-age OVR should be stable: {ovr_0:.1f} → {ovr_2:.1f}"

    def test_old_player_declines(self):
        """A 35-year-old should have lower OVR after 3 years."""
        p = _uniform_player(65)
        ovr_0 = project_ovr(p)
        ovr_3 = project_ovr(project_ratings(p, 35.0, 3))
        assert ovr_3 < ovr_0, f"Old player should decline: {ovr_0:.1f} → {ovr_3:.1f}"

    def test_old_shooter_vs_athlete_decline(self):
        """
        At age 35, shooting skill should decline much less than athleticism.
        Create two identical players; give one high shooting and one high speed.
        After projection the shooter retains more value.
        """
        base = _uniform_player(55)

        # Shooter: high ft, fg, tp
        shooter = dict(base)
        shooter.update(ft=85, fg=85, tp=85, spd=55, jmp=55)

        # Athlete: high spd, jmp
        athlete = dict(base)
        athlete.update(ft=55, fg=55, tp=55, spd=85, jmp=85)

        years = 5
        proj_shooter = project_ratings(shooter, 35.0, years)
        proj_athlete = project_ratings(athlete, 35.0, years)

        shooter_spd_drop  = shooter["spd"]  - proj_shooter["spd"]
        athlete_spd_drop  = athlete["spd"]  - proj_athlete["spd"]
        shooter_tp_drop   = shooter["tp"]   - proj_shooter["tp"]
        athlete_tp_drop   = athlete["tp"]   - proj_athlete["tp"]

        # Athlete's speed should drop more than shooter's speed (same base 55)
        assert athlete_spd_drop > shooter_spd_drop, (
            "Athletes should lose more speed than non-athletes of same age"
        )
        # Shooter's tp should drop less than athlete's tp (same base 55)
        assert shooter_tp_drop <= athlete_tp_drop + 2, (
            "Shooter's 3pt should decline no more than athlete's"
        )

    def test_iq_grows_fast_for_young(self):
        """For a 19-year-old, oiq / diq should grow substantially."""
        p = _uniform_player(50)
        projected = project_ratings(p, 19.0, 3)
        assert projected["oiq"] > p["oiq"] + 5, (
            f"Young IQ should grow fast: {p['oiq']} → {projected['oiq']:.1f}"
        )

    def test_ratings_stay_in_bounds(self):
        """Projected ratings must always stay in [0, 100]."""
        extreme_young = {r: 5.0 for r in BASE_RATINGS}
        extreme_old   = {r: 95.0 for r in BASE_RATINGS}
        for player, age in [(extreme_young, 18.0), (extreme_old, 40.0)]:
            for years in [1, 3, 5]:
                proj = project_ratings(player, age, years)
                for r, v in proj.items():
                    assert 0.0 <= v <= 100.0, (
                        f"Rating {r} out of bounds: {v:.1f} (age={age}, years={years})"
                    )

    def test_height_never_changes(self):
        """hgt must be frozen after projection."""
        p = _uniform_player(60)
        hgt_before = p["hgt"]
        proj = project_ratings(p, 19.0, 5)
        assert proj["hgt"] == pytest.approx(hgt_before), "Height must not change"

    def test_years_forward_zero(self):
        """Zero years forward returns the same ratings."""
        p = _uniform_player(72)
        proj = project_ratings(p, 25.0, 0)
        for r in BASE_RATINGS:
            assert proj[r] == pytest.approx(p[r])


# ---------------------------------------------------------------------------
# project_ovr
# ---------------------------------------------------------------------------

class TestProjectOvr:
    def test_higher_ratings_higher_ovr(self):
        low  = project_ovr(_uniform_player(40))
        high = project_ovr(_uniform_player(80))
        assert high > low

    def test_ovr_in_reasonable_range(self):
        for rating in (20, 50, 80):
            ovr = project_ovr(_uniform_player(rating))
            assert 0 <= ovr <= 100


# ---------------------------------------------------------------------------
# project_contract_status
# ---------------------------------------------------------------------------

class TestContractStatus:
    def test_under_contract(self):
        assert project_contract_status(
            contract_exp=2027, current_season=2024, target_season=2026
        )

    def test_exactly_expiring(self):
        # Contract exp=2026 means the player is under contract through 2026
        assert project_contract_status(
            contract_exp=2026, current_season=2024, target_season=2026
        )

    def test_expired(self):
        assert not project_contract_status(
            contract_exp=2025, current_season=2024, target_season=2026
        )


# ---------------------------------------------------------------------------
# estimate_next_contract
# ---------------------------------------------------------------------------

class TestEstimateNextContract:
    def test_star_earns_more_than_scrub(self):
        star_salary   = estimate_next_contract(85, age=27)
        scrub_salary  = estimate_next_contract(55, age=27)
        assert star_salary > scrub_salary

    def test_min_contract_floor(self):
        bad_salary = estimate_next_contract(30, age=35)
        assert bad_salary >= 1200.0

    def test_max_contract_cap(self):
        elite_salary = estimate_next_contract(95, age=25)
        assert elite_salary <= 50_000.0

    def test_loyalty_discount(self):
        away_salary  = estimate_next_contract(75, age=28, is_my_team=False)
        home_salary  = estimate_next_contract(75, age=28, is_my_team=True)
        assert home_salary < away_salary

    def test_contract_increases_with_ovr(self):
        salaries = [estimate_next_contract(ovr, age=27) for ovr in [50, 60, 70, 80]]
        for i in range(len(salaries) - 1):
            assert salaries[i] <= salaries[i + 1], (
                f"Salary should increase with OVR: {salaries}"
            )
