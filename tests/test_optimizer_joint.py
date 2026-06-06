"""
Tests for src/core/optimizer_joint.py.
"""

import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS
from src.core.ai_trade_value import league_value_stats
from src.core.optimizer_joint import optimize_decisions

SEASON = 2028
MY_TID = 0
OPP_TID = 1


def _player(
    ovr: int = 65,
    age: float = 27.0,
    salary: float = 10_000.0,
    contract_exp: int = SEASON + 3,
    tid: int = MY_TID,
    name: str = "Player",
    pid: int = 1,
    **extra,
) -> dict:
    p = {r: float(ovr) for r in BASE_RATINGS}
    p.update(
        pid=pid, name=name, age=age, pot=float(ovr),
        salary=salary, contract_exp=contract_exp, tid=tid, pos="SF",
    )
    p.update(extra)
    return p


def _df(*players) -> pd.DataFrame:
    from src.core.parser import OUTPUT_COLUMNS
    df = pd.DataFrame(list(players))
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def _cap_info(salary_cap: float = 90_000.0) -> dict:
    return dict(
        salary_cap=salary_cap,
        salary_cap_type="soft",
        soft_cap_trade_match=1.25,
        current_season=SEASON,
        team_strategies={MY_TID: "contending", OPP_TID: "rebuilding"},
    )


def _build(my_players, opp_players):
    """Build my_df, league_dict, league, cap_info for a test scenario."""
    my_df = _df(*my_players)
    opp_df = _df(*opp_players)
    league_dict = {"OPP": opp_df}
    all_rosters = {"__mine__": my_df, **league_dict}
    ci = _cap_info()
    league = league_value_stats(
        all_rosters,
        current_season=SEASON,
        salary_cap=ci["salary_cap"],
        team_strategies=ci.get("team_strategies", {}),
    )
    return my_df, league_dict, league, ci


# ---------------------------------------------------------------------------
# Baseline: no expiring contracts, no beneficial trades
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_returns_required_keys(self):
        my_players = [_player(70, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        for key in ("actions", "let_walk", "projected_v", "initial_v", "remaining_cap_k"):
            assert key in result, f"Missing key: {key}"

    def test_actions_is_list(self):
        my_players = [_player(70, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        assert isinstance(result["actions"], list)
        assert isinstance(result["let_walk"], list)

    def test_projected_v_is_float(self):
        my_players = [_player(70, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        assert isinstance(result["projected_v"], float)
        assert isinstance(result["initial_v"], float)

    def test_projected_v_geq_initial_v(self):
        """Only positive-ΔV actions are applied, so V never decreases."""
        my_players = [_player(70, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        assert result["projected_v"] >= result["initial_v"] - 1e-9


# ---------------------------------------------------------------------------
# Action field presence
# ---------------------------------------------------------------------------

class TestActionFields:
    def test_resign_action_has_required_fields(self):
        """If a resign action is applied, it must have name, demand_k, delta_v."""
        expiring = _player(80, age=26, salary=15_000, contract_exp=SEASON, pid=1, name="Star")
        locked   = [_player(65, pid=10+i, name=f"Lock{i}") for i in range(7)]
        opp_players = [_player(55, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build([expiring] + locked, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        resign_actions = [a for a in result["actions"] if a.get("type") == "resign"]
        for a in resign_actions:
            for field in ("type", "name", "age", "demand_k", "delta_v",
                          "current_ovr", "projected_ovr"):
                assert field in a, f"Resign action missing field: {field}"

    def test_trade_action_has_required_fields(self):
        """Trade actions must have team, incoming, outgoing, delta_v."""
        my_players  = [_player(70, pid=i, name=f"My{i}", salary=8_000) for i in range(8)]
        opp_players = [_player(85, tid=OPP_TID, pid=100+i, name=f"Opp{i}", salary=5_000)
                       for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        trade_actions = [a for a in result["actions"] if a.get("type") == "trade"]
        for a in trade_actions:
            for field in ("type", "team", "incoming", "outgoing", "delta_v", "description"):
                assert field in a, f"Trade action missing field: {field}"

    def test_no_private_fields_in_output(self):
        """Internal _ fields must not appear in the output."""
        my_players = [_player(70, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        for action in result["actions"]:
            for key in action:
                assert not key.startswith("_"), f"Private field leaked: {key}"


# ---------------------------------------------------------------------------
# Expiring contract handling
# ---------------------------------------------------------------------------

class TestExpiringContracts:
    def test_expiring_star_not_in_let_walk(self):
        """A high-value expiring player should be resigned, not let walk."""
        expiring = _player(85, age=25, salary=10_000, contract_exp=SEASON, pid=1, name="Star")
        locked   = [_player(65, pid=10+i, name=f"Lock{i}") for i in range(7)]
        opp_players = [_player(55, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build([expiring] + locked, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        let_walk_names = {p["name"] for p in result["let_walk"]}
        assert "Star" not in let_walk_names

    def test_expiring_player_appears_somewhere(self):
        """Every expiring player ends up either in actions or let_walk."""
        expiring = _player(70, age=28, salary=8_000, contract_exp=SEASON, pid=1, name="Exp")
        locked   = [_player(65, pid=10+i, name=f"Lock{i}") for i in range(7)]
        opp_players = [_player(55, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build([expiring] + locked, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        action_names = {a.get("name") for a in result["actions"]}
        let_walk_names = {p["name"] for p in result["let_walk"]}
        assert "Exp" in action_names or "Exp" in let_walk_names

    def test_no_expiring_no_let_walk(self):
        """With no expiring contracts, let_walk is empty."""
        my_players = [_player(70, contract_exp=SEASON+3, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        assert result["let_walk"] == []

    def test_let_walk_has_name_age_ovr(self):
        """let_walk entries have name, age, current_ovr."""
        expiring = _player(42, age=38, salary=2_000, contract_exp=SEASON, pid=1, name="OldGuy")
        locked   = [_player(65, pid=10+i, name=f"Lock{i}") for i in range(7)]
        opp_players = [_player(55, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build([expiring] + locked, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        for p in result["let_walk"]:
            assert "name" in p
            assert "age" in p
            assert "current_ovr" in p


# ---------------------------------------------------------------------------
# State management guards
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_no_player_appears_in_both_actions_and_let_walk(self):
        """A player can't be both resigned (in actions) and in let_walk."""
        expiring = _player(75, age=26, salary=12_000, contract_exp=SEASON, pid=1, name="Exp")
        locked   = [_player(65, pid=10+i, name=f"Lock{i}") for i in range(7)]
        opp_players = [_player(55, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build([expiring] + locked, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        action_names = {a.get("name") for a in result["actions"]}
        walk_names   = {p["name"] for p in result["let_walk"]}
        # Intersection must be empty
        assert not (action_names & walk_names), (
            f"Player appears in both actions and let_walk: {action_names & walk_names}"
        )

    def test_no_duplicate_player_in_applied_actions(self):
        """The same player pid must not appear as incoming in more than one action."""
        my_players  = [_player(70, pid=i, name=f"My{i}", salary=8_000) for i in range(8)]
        opp_players = [_player(80, tid=OPP_TID, pid=100+i, name=f"Star{i}", salary=6_000)
                       for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        # Collect all pids received across all trade actions
        acquired: list[int] = []
        for action in result["actions"]:
            for p in action.get("incoming") or []:
                pid = p.get("pid")
                if pid is not None:
                    acquired.append(int(float(pid)))
        assert len(acquired) == len(set(acquired)), (
            f"Same player acquired twice: {[p for p in acquired if acquired.count(p) > 1]}"
        )

    def test_no_player_sent_twice(self):
        """A player sent in one trade must not appear as outgoing in another."""
        my_players  = [_player(70, pid=i, name=f"My{i}", salary=8_000) for i in range(8)]
        opp_players = [_player(80, tid=OPP_TID, pid=100+i, name=f"Star{i}", salary=6_000)
                       for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)

        sent: list[int] = []
        for action in result["actions"]:
            for p in action.get("outgoing") or []:
                pid = p.get("pid")
                if pid is not None:
                    sent.append(int(float(pid)))
        assert len(sent) == len(set(sent)), (
            f"Same player sent twice: {[p for p in sent if sent.count(p) > 1]}"
        )

    def test_cap_decreases_after_expensive_resign(self):
        """Resigning an expensive player reduces remaining cap space."""
        expiring = _player(80, age=26, salary=5_000, contract_exp=SEASON, pid=1, name="Star")
        locked   = [_player(65, pid=10+i, name=f"Lock{i}", salary=8_000) for i in range(7)]
        opp_players = [_player(55, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build([expiring] + locked, opp_players)

        base_payroll = sum(p["salary"] for p in [expiring] + locked)
        result = optimize_decisions(my_df, league_dict, league, ci)

        cap = float(ci["salary_cap"])
        # After resigning a player at a higher salary, remaining cap should be
        # less than cap minus base_payroll
        resign_actions = [a for a in result["actions"] if a.get("type") == "resign"]
        if resign_actions:
            expected_max_cap = cap - base_payroll
            assert result["remaining_cap_k"] <= expected_max_cap + 1  # +1 for rounding


# ---------------------------------------------------------------------------
# Cap tracking
# ---------------------------------------------------------------------------

class TestCapTracking:
    def test_remaining_cap_is_non_negative(self):
        my_players = [_player(70, pid=i, name=f"My{i}") for i in range(8)]
        opp_players = [_player(65, tid=OPP_TID, pid=100+i, name=f"Opp{i}") for i in range(8)]
        my_df, league_dict, league, ci = _build(my_players, opp_players)
        result = optimize_decisions(my_df, league_dict, league, ci)
        assert result["remaining_cap_k"] >= 0


# ---------------------------------------------------------------------------
# Server endpoint
# ---------------------------------------------------------------------------

class TestServerEndpoint:
    def test_optimize_endpoint_returns_expected_shape(self):
        from src.server import optimize as server_optimize

        players_raw = [
            {
                "pid": i, "firstName": f"My{i}", "lastName": "Player",
                "tid": MY_TID,
                "born": {"year": SEASON - 27},
                "contract": {"amount": 8_000, "exp": SEASON + 3},
                "ratings": [{"season": SEASON, "pot": 70, "ovr": 67, "pos": "SF",
                             **{r: 67 for r in BASE_RATINGS}}],
            }
            for i in range(8)
        ] + [
            {
                "pid": 100 + i, "firstName": f"Opp{i}", "lastName": "Player",
                "tid": OPP_TID,
                "born": {"year": SEASON - 27},
                "contract": {"amount": 7_000, "exp": SEASON + 3},
                "ratings": [{"season": SEASON, "pot": 65, "ovr": 63, "pos": "PF",
                             **{r: 63 for r in BASE_RATINGS}}],
            }
            for i in range(8)
        ]

        payload = {
            "tid": MY_TID,
            "players": players_raw,
            "gameAttributes": {"season": SEASON, "salaryCap": 90_000},
            "teams": [
                {"tid": MY_TID, "abbrev": "MY", "strategy": "contending"},
                {"tid": OPP_TID, "abbrev": "OPP", "strategy": "rebuilding"},
            ],
        }

        result = server_optimize(payload)
        for key in ("actions", "let_walk", "projected_v", "initial_v", "remaining_cap_k"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["actions"], list)
        assert result["projected_v"] >= result["initial_v"] - 1e-9
