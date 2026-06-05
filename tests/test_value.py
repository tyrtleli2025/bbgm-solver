"""
Tests for src/value.py — horizon-aware team value function V.

Litmus tests:
  1. Best team in the league has the highest V.
  2. Worst team has V near 0 (likely never makes playoffs).
  3. A young prospect of equal current OVR has higher V than an old player.
  4. Trading the young prospect for the old player lowers V (THE key test).
"""

import math
import pytest

from src.core.formulas import BASE_RATINGS, player_ovr as _player_ovr
from src.value import (
    LeagueState,
    LeagueVContext,
    compute_V,
    asset_marginal_value,
    VALUE_BY_PICK,
    _mov,
    _pick_ovr,
    _title_prob,
)

CURRENT_SEASON = 2024
SALARY_CAP     = 90_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player(
    pid:          int,
    tid:          int,
    rating:       int   = 60,
    age:          float = 27.0,
    salary:       float = 5_000.0,
    contract_exp: int   = CURRENT_SEASON + 3,
    pot:          float = 0.0,
) -> dict:
    p = {r: float(rating) for r in BASE_RATINGS}
    p.update(
        pid=float(pid), tid=tid, age=age,
        salary=salary, contract_exp=contract_exp,
        pot=pot or float(rating), name=f"p{pid}",
    )
    return p


def _make_league(num_teams: int = 30, my_tid: int = 0) -> LeagueState:
    """Create a minimal synthetic league."""
    players = []
    pid = 1
    for tid in range(num_teams):
        # 12 average players per team
        for _ in range(12):
            players.append(_player(pid, tid, rating=60, age=27, salary=5_000))
            pid += 1

    teams = [{"tid": i, "abbrev": f"T{i:02d}"} for i in range(num_teams)]
    return LeagueState(
        players=players, teams=teams, picks=[],
        current_season=CURRENT_SEASON, salary_cap=SALARY_CAP,
        my_tid=my_tid, num_teams=num_teams,
    )


def _elite_league(my_tid: int = 0) -> LeagueState:
    """Create a league where my_tid has elite players and others are average."""
    players = []
    pid = 1
    for tid in range(30):
        rating = 90 if tid == my_tid else 60
        for _ in range(12):
            players.append(_player(pid, tid, rating=rating, age=27, salary=5_000))
            pid += 1
    teams = [{"tid": i, "abbrev": f"T{i:02d}"} for i in range(30)]
    return LeagueState(
        players=players, teams=teams, picks=[],
        current_season=CURRENT_SEASON, salary_cap=SALARY_CAP,
        my_tid=my_tid,
    )


# ---------------------------------------------------------------------------
# MOV formula
# ---------------------------------------------------------------------------

class TestMov:
    def test_better_team_has_higher_mov(self):
        good = sorted([80.0] * 10, reverse=True)
        bad  = sorted([60.0] * 10, reverse=True)
        assert _mov(good) > _mov(bad)

    def test_positive_for_strong_team(self):
        elite = sorted([85.0] * 10, reverse=True)
        assert _mov(elite) > 0

    def test_negative_for_weak_team(self):
        weak = sorted([40.0] * 10, reverse=True)
        assert _mov(weak) < 0


# ---------------------------------------------------------------------------
# Title probability
# ---------------------------------------------------------------------------

class TestTitleProb:
    def test_best_team_highest_prob(self):
        movs = {0: 10.0, 1: 5.0, 2: 3.0, 3: 0.0}
        probs = {tid: _title_prob(movs, tid, beta=0.15) for tid in movs}
        assert max(probs, key=probs.get) == 0

    def test_probs_sum_to_one(self):
        movs = {i: float(i) for i in range(8)}
        total = sum(_title_prob(movs, tid, beta=0.15) for tid in movs)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_equal_movs_equal_probs(self):
        movs = {0: 5.0, 1: 5.0}
        assert _title_prob(movs, 0, 0.15) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Pick tables
# ---------------------------------------------------------------------------

class TestPickTables:
    def test_pick1_highest_value(self):
        assert VALUE_BY_PICK[1] > VALUE_BY_PICK[10] > VALUE_BY_PICK[30]

    def test_pick_ovr_r1_better_than_r2(self):
        assert _pick_ovr(1,  round_num=1) > _pick_ovr(1,  round_num=2)
        assert _pick_ovr(15, round_num=1) > _pick_ovr(15, round_num=2)


# ---------------------------------------------------------------------------
# LeagueVContext construction and V ordering
# ---------------------------------------------------------------------------

class TestLeagueVContext:

    def test_elite_team_has_higher_v_than_league(self):
        """The elite team (tid=0) should have far higher V than an average team (tid=1)."""
        # Build a single league where tid=0 is elite, everyone else is average.
        # Compare V when we ARE the elite team vs. when we are an average team.
        players = []
        pid = 1
        for tid in range(30):
            rating = 90 if tid == 0 else 60     # tid=0 is always elite
            for _ in range(12):
                players.append(_player(pid, tid, rating=rating, age=27))
                pid += 1
        teams = [{"tid": i} for i in range(30)]

        def _v(my_tid: int) -> float:
            ls  = LeagueState(players=players, teams=teams, picks=[],
                              current_season=CURRENT_SEASON, salary_cap=SALARY_CAP,
                              my_tid=my_tid)
            return compute_V(my_tid, ls, H=3)

        v_elite = _v(my_tid=0)   # elite team
        v_avg   = _v(my_tid=1)   # average team

        assert v_elite > v_avg, (
            f"Elite team V ({v_elite:.4f}) should exceed average team V ({v_avg:.4f})"
        )

    def test_v_bounded_0_to_1(self):
        """V is a discounted sum of probabilities, so it must be in (0, H)."""
        ls  = _make_league()
        ctx = LeagueVContext(ls, H=5)
        # Even the best possible V ≤ H (if P(title)=1 every season and γ=1)
        assert 0.0 <= ctx.v_current <= 5.0

    def test_v_current_matches_compute_v(self):
        """LeagueVContext.v_current must equal the standalone compute_V."""
        ls  = _make_league()
        ctx = LeagueVContext(ls, H=3, gamma=0.95, beta=0.15)
        v1  = ctx.v_current
        v2  = compute_V(ls.my_tid, ls, H=3, gamma=0.95, beta=0.15)
        assert v1 == pytest.approx(v2, rel=1e-4)

    def test_adding_star_increases_v(self):
        """Adding a superstar to my roster should increase V."""
        ls    = _make_league()
        ctx   = LeagueVContext(ls, H=3)
        star  = _player(9999, my_tid := ls.my_tid, rating=95, age=24, salary=8_000)
        dv    = ctx.delta_v(add_players=[star], remove_players=[])
        assert dv > 0, f"Adding a star should raise V; got ΔV={dv:.4f}"

    def test_removing_star_decreases_v(self):
        """Removing a star from a stellar roster should decrease V."""
        # Build a team with a star and 11 average
        players = []
        pid = 1
        star_dict = _player(pid, my_tid := 0, rating=90, age=26, salary=10_000)
        players.append(star_dict)
        pid += 1
        for _ in range(11):
            players.append(_player(pid, 0, rating=60, age=27))
            pid += 1
        for tid in range(1, 30):
            for _ in range(12):
                players.append(_player(pid, tid, rating=60, age=27))
                pid += 1

        teams = [{"tid": i} for i in range(30)]
        ls    = LeagueState(players=players, teams=teams, picks=[],
                            current_season=CURRENT_SEASON, salary_cap=SALARY_CAP,
                            my_tid=0)
        ctx   = LeagueVContext(ls, H=3)
        dv    = ctx.delta_v(add_players=[], remove_players=[star_dict])
        assert dv < 0, f"Removing star should lower V; got ΔV={dv:.4f}"


# ---------------------------------------------------------------------------
# THE LITMUS TEST: young prospect vs. old player of equal current OVR
# ---------------------------------------------------------------------------

class TestYoungVsOld:

    def _prospect_and_veteran(self, rating: int = 67):
        """Return two players with equal current OVR but very different ages."""
        prospect = _player(101, 0, rating=rating, age=20.0,
                           salary=2_000, contract_exp=CURRENT_SEASON + 3)
        veteran  = _player(102, 0, rating=rating, age=35.0,
                           salary=2_000, contract_exp=CURRENT_SEASON + 3)
        # Confirm equal current OVR
        assert abs(
            _player_ovr(prospect) - _player_ovr(veteran)
        ) < 3, "Players must have similar current OVR for this test to be meaningful"
        return prospect, veteran

    def _league_with_swap(
        self,
        keep_player: dict,
        baseline_rating: int = 60,
    ) -> LeagueState:
        """League where my team has 11 average players + keep_player."""
        players = [keep_player]
        pid = 200
        for _ in range(11):
            players.append(_player(pid, 0, rating=baseline_rating, age=27))
            pid += 1
        for tid in range(1, 30):
            for _ in range(12):
                players.append(_player(pid, tid, rating=baseline_rating, age=27))
                pid += 1
        teams = [{"tid": i} for i in range(30)]
        return LeagueState(
            players=players, teams=teams, picks=[],
            current_season=CURRENT_SEASON, salary_cap=SALARY_CAP, my_tid=0,
        )

    def test_prospect_has_higher_v_than_veteran(self):
        """A young prospect has higher V than an old player of equal current OVR."""
        prospect, veteran = self._prospect_and_veteran()

        ls_p  = self._league_with_swap(prospect)
        ls_v  = self._league_with_swap(veteran)

        v_prospect = compute_V(0, ls_p, H=5)
        v_veteran  = compute_V(0, ls_v, H=5)

        assert v_prospect > v_veteran, (
            f"Prospect V ({v_prospect:.4f}) should exceed veteran V ({v_veteran:.4f}). "
            f"The aging model must project the prospect improving and veteran declining."
        )

    def test_trading_prospect_for_veteran_lowers_v(self):
        """
        THE KEY TEST: trading away the young prospect for the old veteran
        (same current OVR) must lower V.
        """
        prospect, veteran = self._prospect_and_veteran()
        ls  = self._league_with_swap(prospect)
        ctx = LeagueVContext(ls, H=5)

        # Simulate the trade: out=prospect, in=veteran
        dv = ctx.delta_v(
            add_players=[veteran],
            remove_players=[prospect],
        )
        assert dv < 0, (
            f"Trading young prospect for old veteran of equal OVR must lower V. "
            f"Got ΔV={dv:.4f}. "
            f"If this fails, the aging model is not correctly projecting forward."
        )

    def test_trading_veteran_for_prospect_raises_v(self):
        """Trading away the veteran and acquiring the prospect raises V."""
        prospect, veteran = self._prospect_and_veteran()
        ls  = self._league_with_swap(veteran)
        ctx = LeagueVContext(ls, H=5)

        dv = ctx.delta_v(
            add_players=[prospect],
            remove_players=[veteran],
        )
        assert dv > 0, (
            f"Trading old veteran for young prospect of equal OVR must raise V. "
            f"Got ΔV={dv:.4f}."
        )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_v_under_100ms(self):
        """compute_V must run in under 100ms for a 30-team league."""
        import time
        ls    = _make_league(num_teams=30)
        start = time.perf_counter()
        compute_V(0, ls, H=5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, (
            f"compute_V took {elapsed_ms:.0f}ms; must be < 100ms"
        )

    def test_delta_v_fast_after_precompute(self):
        """delta_v must be fast after LeagueVContext is constructed."""
        import time
        ls  = _make_league(num_teams=30)
        ctx = LeagueVContext(ls, H=5)

        star = _player(9999, 0, rating=90, age=24)

        start = time.perf_counter()
        for _ in range(20):
            ctx.delta_v(add_players=[star], remove_players=[])
        elapsed_ms = (time.perf_counter() - start) * 1000 / 20
        assert elapsed_ms < 20, (
            f"delta_v averaged {elapsed_ms:.1f}ms per call; target < 20ms"
        )
