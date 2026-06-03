"""
Tests for the ZenGM league export parser (src/core/parser.py).

All tests use an in-memory mock JSON written to a pytest tmp_path temp file —
no real league export is required.
"""

import json
import math
import pytest
import pandas as pd

from src.core.formulas import BASE_RATINGS
from src.core.parser import (
    parse_league_json,
    OUTPUT_COLUMNS,
    META_COLUMNS,
    _TID_FREE_AGENT,
    _TID_DRAFT_PROSPECT,
    _TID_RETIRED,
)

# ---------------------------------------------------------------------------
# Mock data factory
# ---------------------------------------------------------------------------

_CURRENT_SEASON = 2024

_BASE_RATINGS_VALUES = {
    "hgt": 72, "stre": 68, "spd": 81, "jmp": 76, "endu": 63,
    "ins": 55, "dnk": 70, "ft": 78, "fg": 67, "tp": 82,
    "oiq": 77, "diq": 66, "drb": 80, "pss": 74, "reb": 58,
}

def _ratings_entry(**overrides) -> dict:
    r = dict(_BASE_RATINGS_VALUES)
    r.update({"season": _CURRENT_SEASON, "pot": 85, "ovr": 74, "pos": "PG"})
    r.update(overrides)
    return r

def _player(
    pid: int,
    first: str,
    last: str,
    tid: int,
    born_year: int = 1998,
    salary: int = 10_000,
    retired_year=None,
    ratings: list | None = None,
    **kw,
) -> dict:
    p: dict = {
        "pid":        pid,
        "firstName":  first,
        "lastName":   last,
        "tid":        tid,
        "born":       {"year": born_year, "loc": "USA"},
        "contract":   {"amount": salary, "exp": 2026},
        "ratings":    ratings if ratings is not None else [_ratings_entry(**kw)],
    }
    if retired_year is not None:
        p["retiredYear"] = retired_year
    return p


def _make_export(*players, teams=None, season=_CURRENT_SEASON) -> dict:
    """Minimal ZenGM export dict with the given players."""
    if teams is None:
        teams = [
            {"tid": 0, "abbrev": "MyT", "name": "My Team"},
            {"tid": 1, "abbrev": "OPP", "name": "Opponent"},
            {"tid": 2, "abbrev": "THD", "name": "Third Team"},
        ]
    return {
        "gameAttributes": [{"key": "season", "value": season}],
        "teams": teams,
        "players": list(players),
    }


def _write_and_parse(tmp_path, export: dict, my_tid: int = 0):
    f = tmp_path / "league.json"
    f.write_text(json.dumps(export))
    return parse_league_json(f, my_tid=my_tid)


# ---------------------------------------------------------------------------
# Fixtures (shared export used by several test classes)
# ---------------------------------------------------------------------------

@pytest.fixture
def full_export():
    """
    Six-player export covering all filtering cases:
      - p1, p2: active on my team (tid=0)
      - p3:     active on opponent (tid=1)
      - p4:     free agent (tid=-1)  → active but no team
      - p5:     retired (tid=0 but retiredYear set)  → excluded
      - p6:     draft prospect (tid=-2)  → excluded
    """
    return _make_export(
        _player(1, "Alice",  "Active",  tid=0,  born_year=1998, salary=10_000),
        _player(2, "Bob",    "Bench",   tid=0,  born_year=2001, salary=5_000),
        _player(3, "Carol",  "Opp",     tid=1,  born_year=1996, salary=15_000),
        _player(4, "Dave",   "Free",    tid=_TID_FREE_AGENT, born_year=1995),
        _player(5, "Eve",    "Retired", tid=0,  retired_year=2022),
        _player(6, "Frank",  "Draft",   tid=_TID_DRAFT_PROSPECT),
    )


# ---------------------------------------------------------------------------
# 1. Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:

    def test_returns_tuple_of_two(self, tmp_path, full_export):
        result = _write_and_parse(tmp_path, full_export)
        assert isinstance(result, tuple) and len(result) == 2

    def test_my_roster_is_dataframe(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        assert isinstance(my_df, pd.DataFrame)

    def test_league_dict_is_dict_of_dataframes(self, tmp_path, full_export):
        _, league = _write_and_parse(tmp_path, full_export)
        assert isinstance(league, dict)
        for v in league.values():
            assert isinstance(v, pd.DataFrame)

    def test_all_base_rating_columns_present(self, tmp_path, full_export):
        my_df, league = _write_and_parse(tmp_path, full_export)
        for df in [my_df] + list(league.values()):
            for col in BASE_RATINGS:
                assert col in df.columns, f"Missing column: {col}"

    def test_meta_columns_present(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        for col in META_COLUMNS:
            assert col in my_df.columns, f"Missing meta column: {col}"

    def test_output_columns_match_constant(self, tmp_path, full_export):
        my_df, league = _write_and_parse(tmp_path, full_export)
        for df in [my_df] + list(league.values()):
            assert list(df.columns) == OUTPUT_COLUMNS

    def test_base_ratings_are_float_dtype(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        for col in BASE_RATINGS:
            assert pd.api.types.is_float_dtype(my_df[col]), f"{col} is not float"


# ---------------------------------------------------------------------------
# 2. Active-player filtering
# ---------------------------------------------------------------------------


class TestActivePlayerFiltering:

    def test_my_roster_contains_only_active_players(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        # Only Alice (pid=1) and Bob (pid=2) are active on tid=0
        assert set(my_df["pid"]) == {1, 2}

    def test_retired_player_excluded(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        # Eve (pid=5) is retired; she must not appear in my roster
        assert 5 not in my_df["pid"].values

    def test_draft_prospect_excluded(self, tmp_path, full_export):
        my_df, league = _write_and_parse(tmp_path, full_export)
        all_pids = list(my_df["pid"])
        for df in league.values():
            all_pids.extend(df["pid"].tolist())
        # Frank (pid=6) is a draft prospect; must not appear anywhere
        assert 6 not in all_pids

    def test_free_agent_not_in_league_dict(self, tmp_path, full_export):
        _, league = _write_and_parse(tmp_path, full_export)
        all_pids = []
        for df in league.values():
            all_pids.extend(df["pid"].tolist())
        # Dave (pid=4) is a free agent; must not appear in league dict
        assert 4 not in all_pids

    def test_free_agent_not_in_my_roster(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        assert 4 not in my_df["pid"].values

    def test_retired_via_tid_minus3_excluded(self, tmp_path):
        export = _make_export(
            _player(1, "Good",    "Player", tid=0),
            _player(2, "Retired", "Player", tid=_TID_RETIRED),
        )
        my_df, _ = _write_and_parse(tmp_path, export)
        assert 2 not in my_df["pid"].values


# ---------------------------------------------------------------------------
# 3. Value mapping
# ---------------------------------------------------------------------------


class TestValueMapping:

    def test_base_ratings_mapped_correctly(self, tmp_path):
        """Every base rating from the JSON must appear verbatim in the DataFrame."""
        export = _make_export(_player(1, "Test", "Player", tid=0))
        my_df, _ = _write_and_parse(tmp_path, export)
        row = my_df.iloc[0]
        for stat, expected in _BASE_RATINGS_VALUES.items():
            assert row[stat] == pytest.approx(float(expected)), (
                f"{stat}: expected {expected}, got {row[stat]}"
            )

    def test_age_calculated_from_born_year(self, tmp_path):
        """age = current_season − born.year."""
        export = _make_export(_player(1, "Young", "Player", tid=0, born_year=1998))
        my_df, _ = _write_and_parse(tmp_path, export, my_tid=0)
        expected_age = _CURRENT_SEASON - 1998   # = 26
        assert my_df.iloc[0]["age"] == pytest.approx(float(expected_age))

    def test_salary_stored_in_thousands(self, tmp_path):
        """contract.amount is in $K and must be preserved as-is (not converted)."""
        export = _make_export(_player(1, "Rich", "Player", tid=0, salary=25_000))
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["salary"] == pytest.approx(25_000.0)

    def test_pot_extracted_from_ratings(self, tmp_path):
        """pot comes from the latest ratings entry, not the player root."""
        export = _make_export(
            _player(1, "Prospect", "Player", tid=0, pot=92)
        )
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["pot"] == pytest.approx(92.0)

    def test_pot_defaults_to_zero_when_absent(self, tmp_path):
        """When 'pot' is missing from the ratings entry, default is 0."""
        ratings_no_pot = [{"season": _CURRENT_SEASON, **_BASE_RATINGS_VALUES}]
        export = _make_export(
            _player(1, "NoPot", "Player", tid=0, ratings=ratings_no_pot)
        )
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["pot"] == 0.0

    def test_name_assembled_from_first_last(self, tmp_path):
        export = _make_export(_player(1, "LeBron", "James", tid=0))
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["name"] == "LeBron James"

    def test_pos_extracted_from_ratings(self, tmp_path):
        export = _make_export(_player(1, "Point", "Guard", tid=0, pos="PG"))
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["pos"] == "PG"

    def test_tid_preserved_in_column(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        assert all(my_df["tid"] == 0)

    def test_latest_ratings_entry_is_used(self, tmp_path):
        """When a player has multiple rating seasons, the last one must be used."""
        multi_ratings = [
            {**_BASE_RATINGS_VALUES, "season": 2022, "hgt": 60, "pot": 70},
            {**_BASE_RATINGS_VALUES, "season": 2023, "hgt": 65, "pot": 75},
            {**_BASE_RATINGS_VALUES, "season": 2024, "hgt": 72, "pot": 85},  # current
        ]
        export = _make_export(
            _player(1, "Multi", "Season", tid=0, ratings=multi_ratings)
        )
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["hgt"] == 72.0
        assert my_df.iloc[0]["pot"] == 85.0

    def test_missing_base_rating_defaults_to_50(self, tmp_path):
        """A ratings entry missing some keys should default those to 50."""
        sparse = [{"season": 2024, "hgt": 80}]   # only hgt provided
        export = _make_export(
            _player(1, "Sparse", "Player", tid=0, ratings=sparse)
        )
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["hgt"] == 80.0
        assert my_df.iloc[0]["spd"] == 50.0   # defaulted


# ---------------------------------------------------------------------------
# 4. Team grouping and league_rosters_dict
# ---------------------------------------------------------------------------


class TestTeamGrouping:

    def test_my_roster_has_correct_team_only(self, tmp_path, full_export):
        my_df, _ = _write_and_parse(tmp_path, full_export)
        assert all(my_df["tid"] == 0)

    def test_opponent_team_appears_in_league_dict(self, tmp_path, full_export):
        _, league = _write_and_parse(tmp_path, full_export)
        assert "OPP" in league

    def test_my_team_not_in_league_dict(self, tmp_path, full_export):
        _, league = _write_and_parse(tmp_path, full_export)
        # My team's abbrev is "MyT"
        assert "MyT" not in league

    def test_opponent_roster_has_correct_players(self, tmp_path, full_export):
        _, league = _write_and_parse(tmp_path, full_export)
        # Carol (pid=3) is the only active player on tid=1 ("OPP")
        assert list(league["OPP"]["pid"]) == [3]

    def test_team_label_uses_abbrev(self, tmp_path, full_export):
        _, league = _write_and_parse(tmp_path, full_export)
        assert "OPP" in league        # abbrev from teams list

    def test_team_label_falls_back_to_team_N_when_no_abbrev(self, tmp_path):
        teams_no_abbrev = [{"tid": 0}, {"tid": 7}]
        export = _make_export(
            _player(1, "A", "B", tid=0),
            _player(2, "C", "D", tid=7),
            teams=teams_no_abbrev,
        )
        _, league = _write_and_parse(tmp_path, export)
        assert "team_7" in league

    def test_custom_my_tid(self, tmp_path):
        """Passing my_tid=1 should return team-1 roster and exclude it from league."""
        export = _make_export(
            _player(1, "Ours",  "Player", tid=1),
            _player(2, "Other", "Player", tid=0),
        )
        my_df, league = _write_and_parse(tmp_path, export, my_tid=1)
        assert list(my_df["pid"]) == [1]
        # team 0 should be in league dict, team 1 should not
        assert any("0" in k or "MyT" in k or "team_0" in k for k in league)

    def test_invalid_my_tid_raises_value_error(self, tmp_path):
        export = _make_export(_player(1, "A", "B", tid=0))
        with pytest.raises(ValueError, match="tid=99"):
            _write_and_parse(tmp_path, export, my_tid=99)


# ---------------------------------------------------------------------------
# 5. Season / age edge cases
# ---------------------------------------------------------------------------


class TestSeasonAndAge:

    def test_season_read_from_list_format(self, tmp_path):
        """Standard list-of-dicts gameAttributes format."""
        export = _make_export(_player(1, "A", "B", tid=0, born_year=1994))
        # _make_export already uses list format; season=2024
        my_df, _ = _write_and_parse(tmp_path, export)
        assert my_df.iloc[0]["age"] == pytest.approx(2024 - 1994)

    def test_season_read_from_dict_format(self, tmp_path):
        """Some exports store gameAttributes as a plain dict."""
        export = {
            "gameAttributes": {"season": 2025},
            "teams": [{"tid": 0, "abbrev": "MyT"}],
            "players": [_player(1, "A", "B", tid=0, born_year=1995)],
        }
        f = tmp_path / "league.json"
        f.write_text(json.dumps(export))
        my_df, _ = parse_league_json(f)
        assert my_df.iloc[0]["age"] == pytest.approx(2025 - 1995)

    def test_missing_season_defaults_age_to_27(self, tmp_path):
        """Without a season we cannot compute age; default to 27."""
        export = {
            "gameAttributes": [],
            "teams": [{"tid": 0, "abbrev": "X"}],
            "players": [_player(1, "A", "B", tid=0, born_year=1990)],
        }
        f = tmp_path / "league.json"
        f.write_text(json.dumps(export))
        my_df, _ = parse_league_json(f)
        assert my_df.iloc[0]["age"] == pytest.approx(27.0)

    def test_zero_salary_when_contract_absent(self, tmp_path):
        """Players with no contract key should have salary=0."""
        export = {
            "gameAttributes": [{"key": "season", "value": 2024}],
            "teams": [{"tid": 0, "abbrev": "X"}],
            "players": [{
                "pid": 1, "firstName": "A", "lastName": "B",
                "tid": 0,
                "born": {"year": 1995, "loc": ""},
                "ratings": [_ratings_entry()],
                # no "contract" key
            }],
        }
        f = tmp_path / "league.json"
        f.write_text(json.dumps(export))
        my_df, _ = parse_league_json(f)
        assert my_df.iloc[0]["salary"] == pytest.approx(0.0)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_league_json(tmp_path / "does_not_exist.json")
