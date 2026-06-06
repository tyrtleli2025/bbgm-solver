#!/usr/bin/env python3
"""
BBGM Solver — command-line interface.

Usage
-----
    python main.py --file league.json
    python main.py --file league.json --tid 3
    python main.py --file league.json --top 10
    python main.py --file league.json --search --depth 3 --beam 5
    python main.py --serve                        # start local server + print bookmarklet
    python main.py --serve --port 9999
    python main.py --serve --ssl-cert cert.pem --ssl-key key.pem
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from src.core.ai_trade_value import league_value_stats
from src.core.formulas import player_ovr
from src.core.market_scanner import find_best_trades
from src.core.optimizer import optimize_rotation
from src.core.parser import parse_league_json
from src.core.trade_search import beam_search, SearchResult, TradeStep

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_W = 64          # total line width for separators
_NAME_W = 24     # player name column width
_POS_W  = 5      # position column width


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _rule(char: str = "─", width: int = _W) -> str:
    return char * width


def _section(title: str, char: str = "─") -> str:
    pad = max(0, _W - len(title) - 5)
    return f"\n{char * 3} {title} {char * pad}"


def _signed(val: float) -> str:
    """Format a signed float with a direction arrow: '▲ +12.3' or '▼  -4.1'."""
    arrow = "▲" if val >= 0 else "▼"
    return f"{arrow} {val:+.1f}"


def _player_line(p: dict | object, label: str, prefix: str) -> str:
    """
    Render a single player row for trade display.

    *p* may be a plain dict (from market-scanner results) or a pd.Series row.
    """
    if hasattr(p, "get"):
        name = str(p.get("name") or f"pid_{p.get('pid', '?')}")
        pos  = str(p.get("pos")  or "")
        age  = p.get("age",  0) or 0
        ovr  = player_ovr(p)
    else:
        name, pos, age, ovr = str(p), "", 0, 0

    return (
        f"      {prefix} {label:<3} "
        f"{name:<{_NAME_W}} "
        f"{pos:<{_POS_W}} "
        f"age {int(age):<3} "
        f"OVR {ovr}"
    )


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------


def _print_banner(season: int, team_label: str, my_tid: int) -> None:
    bar = "═" * _W
    title = f"  BBGM Solver"
    meta  = f"Season {season}  •  Team: {team_label} (tid={my_tid})"
    print(f"\n╔{bar}╗")
    print(f"║{title:<{_W}}║")
    print(f"║  {meta:<{_W - 2}}║")
    print(f"╚{bar}╝")


def _print_lineup(result: dict) -> None:
    lineup  = result["lineup"]
    syn     = result["synergy"]
    score   = result["score"]
    ovr_sum = result["ovr_sum"]

    print()
    hdr = f"    {'#':<4}{'Name':<{_NAME_W + 1}}{'Pos':<{_POS_W + 1}}{'Age':<5}{'OVR'}"
    print(hdr)
    print(f"    {_rule(char='─', width=len(hdr) - 4)}")

    for i in range(len(lineup)):
        row  = lineup.iloc[i]
        name = str(row.get("name") or f"pid_{int(row.get('pid', i))}")
        pos  = str(row.get("pos")  or "")
        age  = int(row.get("age", 0) or 0)
        ovr  = player_ovr(row)
        print(f"    {i + 1:<4}{name:<{_NAME_W + 1}}{pos:<{_POS_W + 1}}{age:<5}{ovr}")

    print()
    print(f"    OVR Sum : {ovr_sum}")
    print(f"    Synergy : Off {syn['off']:.3f}  |  Def {syn['def']:.3f}  |  Reb {syn['reb']:.3f}")
    print(f"    Score   : {score:.1f}")
    print()


def _print_trades(trades: list[dict]) -> None:
    print()

    if not trades:
        print("    No beneficial trades found.")
        print()
        return

    for rank, t in enumerate(trades, 1):
        team  = t["team"]
        ttype = t["trade_type"]
        net_l = t["net_lineup_score"]
        dv    = t.get("dv", 0.0)

        print(
            f"  #{rank:<2}  {team:<10}  {ttype:<8}  "
            f"Lineup {_signed(net_l):>9}   dv {_signed(dv):>9}"
        )

        all_rows = (
            [(p, "IN",  "┌─") for p in t["incoming"]]
            + [(p, "OUT", "└─") for p in t["outgoing"]]
        )
        for idx, (p, label, _) in enumerate(all_rows):
            is_last = idx == len(all_rows) - 1
            prefix  = "└─" if is_last else ("┌─" if idx == 0 else "├─")
            print(_player_line(p, label, prefix))

        print()

    print(_rule())
    print()


def _print_sequence(result: SearchResult, rank: int) -> None:
    """Print one trade sequence from beam_search."""
    gain = result.total_j_gain
    traj = result.j_trajectory
    print(
        f"  #{rank:<2}  {len(result.sequence)}-step sequence  "
        f"J gain: {_signed(gain)}  "
        f"({traj[0]:.1f} → {traj[-1]:.1f})"
    )
    for step_i, step in enumerate(result.sequence, 1):
        print(
            f"\n      Step {step_i}  {step.team:<10}  {step.trade_type:<8}  "
            f"J: {step.j_before:.1f} → {step.j_after:.1f}  "
            f"dv: {_signed(step.dv)}"
        )
        all_rows = (
            [(p, "IN",  "┌─") for p in step.incoming]
            + [(p, "OUT", "└─") for p in step.outgoing]
        )
        for idx, (p, label, _) in enumerate(all_rows):
            is_last = idx == len(all_rows) - 1
            prefix  = "└─" if is_last else ("┌─" if idx == 0 else "├─")
            print(_player_line(p, label, prefix))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bbgm-solver",
        description="Optimal lineup finder and trade scanner for Basketball GM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py --file league.json
              python main.py --file league.json --tid 3
              python main.py --file league.json --top 10
              python main.py --file league.json --search --depth 3 --beam 5
        """),
    )
    ap.add_argument(
        "--file",
        required=False,
        default=None,
        metavar="PATH",
        help="Path to the ZenGM league export JSON file (required unless --serve).",
    )
    ap.add_argument(
        "--tid",
        type=int,
        default=0,
        metavar="N",
        help="Team ID of your team in the export (default: 0).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of trade recommendations to display (default: 5).",
    )
    ap.add_argument(
        "--search",
        action="store_true",
        help="Run depth-limited beam search for the best multi-step trade sequence.",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=3,
        metavar="D",
        help="Maximum trade-sequence length for --search (default: 3).",
    )
    ap.add_argument(
        "--beam",
        type=int,
        default=5,
        metavar="K",
        help="Beam width (candidate trades per node) for --search (default: 5).",
    )
    # ── serve mode ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Start a local HTTP server (default port 8888) and print the "
            "browser bookmarklet.  Eliminates the export-download-drag cycle."
        ),
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8888,
        metavar="PORT",
        help="Port for --serve mode (default: 8888).",
    )
    ap.add_argument(
        "--use-v",
        action="store_true",
        dest="use_v_function",
        help=(
            "Use horizon-aware V function (multi-year title equity) instead of "
            "J (instantaneous lineup score). V accounts for player aging, prospect "
            "upside, and draft picks over a 5-year horizon. Slower but strategically superior."
        ),
    )
    # ── serve mode ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--ssl-cert",
        metavar="CERT",
        dest="ssl_cert",
        help="PEM certificate file for HTTPS in --serve mode.",
    )
    ap.add_argument(
        "--ssl-key",
        metavar="KEY",
        dest="ssl_key",
        help="PEM private-key file for HTTPS in --serve mode.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # ── serve mode (mutually exclusive with --file) ────────────────────────
    if args.serve:
        from src.server import start_server  # deferred import (stdlib only)
        import logging
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
        start_server(
            port=args.port,
            ssl_cert=getattr(args, "ssl_cert", None),
            ssl_key=getattr(args, "ssl_key", None),
            use_v=args.use_v_function,
        )
        return 0   # start_server blocks until Ctrl-C; this line is unreachable

    # ── 1. Parse ─────────────────────────────────────────────────────────────
    if not args.file:
        print("Error: --file is required (or use --serve to start the local server).",
              file=sys.stderr)
        return 1

    print(f"\nLoading {args.file} …", end=" ", flush=True)
    try:
        my_roster_df, league_rosters_dict, cap_info = parse_league_json(
            args.file, my_tid=args.tid
        )
    except FileNotFoundError:
        print()
        print(f"Error: file not found — {args.file}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    n_my     = len(my_roster_df)
    n_teams  = len(league_rosters_dict)
    n_others = sum(len(df) for df in league_rosters_dict.values())
    print(f"done.")

    # Build a shared League dict containing real cap settings from the export
    _all_rosters = {"__mine__": my_roster_df, **league_rosters_dict}
    league = league_value_stats(
        _all_rosters,
        current_season=cap_info.get("current_season", 0),
        salary_cap=cap_info["salary_cap"],
        salary_cap_type=cap_info["salary_cap_type"],
        soft_cap_trade_match=cap_info["soft_cap_trade_match"],
    )

    print(
        f"  Cap: ${cap_info['salary_cap']/1000:.1f} M  "
        f"type={cap_info['salary_cap_type']}  "
        f"trade-match={cap_info['soft_cap_trade_match']*100:.0f} %"
    )
    print(
        f"  My roster  : {n_my} active players\n"
        f"  Opponents  : {n_teams} teams  ({n_others} players)\n"
    )

    # ── 2. Optimal lineup ────────────────────────────────────────────────────
    print(_section("OPTIMAL STARTING LINEUP", "━"))

    if n_my < 5:
        print(f"\n  Error: need at least 5 players on my roster, found {n_my}.")
        return 1

    print("\n  Optimising lineup …", end=" ", flush=True)
    lineup_result = optimize_rotation(my_roster_df)
    print("done.")

    _print_lineup(lineup_result)

    # ── 3. Market scan ───────────────────────────────────────────────────────
    print(_section("TOP TRADE RECOMMENDATIONS", "━"))
    print()

    if not league_rosters_dict:
        print("  No opponent rosters found in the export; skipping trade scan.")
        print()
        return 0

    print(f"  Scanning {n_teams} teams for 1-for-1 and 2-for-1 trades …\n")

    def _on_progress(team_name: str, n_done: int, n_total: int) -> None:
        bar_w   = 24
        filled  = int(bar_w * n_done / max(n_total, 1))
        bar     = "█" * filled + "░" * (bar_w - filled)
        print(
            f"\r  [{bar}] {n_done:>2}/{n_total}  {team_name:<12}",
            end="",
            flush=True,
        )

    # Extract current season from league dict (added by league_value_stats)
    current_season = int(league.get("current_season", 0))

    trades = find_best_trades(
        my_roster_df,
        league_rosters_dict,
        league=league,
        salary_cap=cap_info["salary_cap"],
        top_n=args.top,
        progress=_on_progress,
        use_v_function=args.use_v_function,
        current_season=current_season,
    )
    print(f"\r  Done — {len(trades)} trade(s) found.{' ' * 30}\n")

    _print_trades(trades)

    # ── 4. Beam search (optional) ─────────────────────────────────────────────
    if args.search:
        print(_section("BEST TRADE SEQUENCE (beam search)", "━"))
        print(
            f"\n  Searching depth={args.depth}, beam={args.beam} …",
            end=" ",
            flush=True,
        )

        if not league_rosters_dict:
            print("\n  No opponent rosters — skipping sequence search.")
            return 0

        sequences = beam_search(
            my_roster_df,
            league_rosters_dict,
            league=league,
            salary_cap=cap_info["salary_cap"],
            depth=args.depth,
            beam_width=args.beam,
            top_n=args.top,
            use_v_function=args.use_v_function,
            current_season=current_season,
        )
        print(f"done.  {len(sequences)} sequence(s) found.\n")

        if not sequences:
            print("  No improving trade sequences found.")
        else:
            for rank, seq in enumerate(sequences, 1):
                _print_sequence(seq, rank)

    return 0


if __name__ == "__main__":
    sys.exit(main())
