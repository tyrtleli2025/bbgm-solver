"""
Local solver HTTP server — eliminates the export-download-drag cycle.

Usage
-----
    python main.py --serve                    # port 8888
    python main.py --serve --port 9999
    python main.py --serve --ssl-cert c.pem --ssl-key k.pem   # HTTPS

The server exposes a single endpoint:

    POST http://localhost:8888/solve
    Content-Type: application/json

    {
      "players":       [...],        # ZenGM player objects (from IDB / export)
      "gameAttributes": [...],       # list-of-{key,value} or plain dict
      "teams":         [...],        # optional, for abbrev labels
      "tid":           0             # your team ID
    }

The payload shape is identical to a ZenGM league export JSON, so
``parse_league_data`` processes it directly.

CORS
----
All responses include ``Access-Control-Allow-Origin: *`` so the browser
bookmarklet can POST cross-origin.  OPTIONS preflight requests are handled.

HTTPS / mixed-content
---------------------
Browsers block HTTP requests from HTTPS pages (play.basketball-gm.com).
Options:
  1. Generate a self-signed cert and run with --ssl-cert / --ssl-key.
     openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \\
             -days 365 -nodes -subj '/CN=localhost'
  2. Use the clipboard fallback built into the bookmarklet (the solver then
     reads a pasted file or stdin).
"""

from __future__ import annotations

import json
import logging
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.core.ai_trade_value import league_value_stats
from src.core.market_scanner import find_best_trades, _build_cache, _compute_team_j
from src.core.optimizer import optimize_rotation
from src.core.parser import parse_league_data
from src.core.trade_search import beam_search

log = logging.getLogger(__name__)

DEFAULT_PORT: int = 8888

# ---------------------------------------------------------------------------
# Bookmarklet  (self-contained JS that reads ZenGM's IndexedDB and POSTs here)
# ---------------------------------------------------------------------------
# The `javascript:` prefix is printed by main.py; this constant holds the body
# so it can also be imported and tested independently.
#
# Size budget: < 2000 chars total (including `javascript:` prefix) so the
# snippet fits as a browser bookmark URL.
#
# Flow:
#   1. Extract league ID from pathname (/l/{lid}/...)
#   2. Prompt for team ID
#   3. Open IndexedDB "league{lid}" → read players, gameAttributes, teams
#   4. POST to localhost:8888/solve
#   5. Display top-5 trade recommendations in an alert
#   6. On network error (e.g. HTTPS→HTTP mixed-content block):
#      copy the payload JSON to the clipboard so the user can paste it into
#      a local solver run.

BOOKMARKLET_BODY: str = """\
(async()=>{
const m=location.pathname.match(/\\/l\\/(\\d+)/);
if(!m)return alert('Open a ZenGM league page first');
const lid=m[1],tid=+(prompt('Your team ID (see URL or Settings)?','0')||0);
const db=await new Promise((r,j)=>{
  const o=indexedDB.open('league'+lid);
  o.onsuccess=e=>r(e.target.result);o.onerror=j});
const A=s=>new Promise((r,j)=>{
  const q=db.transaction(s,'readonly').objectStore(s).getAll();
  q.onsuccess=e=>r(e.target.result);q.onerror=j});
const[pl,ga]=await Promise.all([A('players'),A('gameAttributes')]);
const tm=await A('teams').catch(()=>[]);
const body=JSON.stringify({players:pl,gameAttributes:ga,teams:tm,tid});
try{
  const d=await(await fetch('http://localhost:PORT/solve',{
    method:'POST',headers:{'Content-Type':'application/json'},body})).json();
  alert((d.trades||[]).slice(0,5).map((t,i)=>
    `#${i+1} ${t.team} ${t.trade_type} +${(t.net_lineup_score||0).toFixed(1)}J dv=${(t.dv||0).toFixed(2)}\\n`+
    (t.incoming||[]).map(p=>' IN: '+(p.name||p.pid)).join('\\n')+'\\n'+
    (t.outgoing||[]).map(p=>'OUT: '+(p.name||p.pid)).join('\\n')
  ).join('\\n---\\n')||'No trades found')
}catch(e){
  navigator.clipboard.writeText(body)
    .then(()=>alert('Server unreachable (HTTPS\\u2192HTTP blocked?).\\nPayload copied to clipboard.\\nRun: python main.py --serve\\nError: '+e.message))
    .catch(()=>alert('Error: '+e.message))
}
})()"""


# Second bookmarklet — team strength evaluator (POSTs to /evaluate)
BOOKMARKLET_EVAL_BODY: str = """\
(async()=>{
const m=location.pathname.match(/\\/l\\/(\\d+)/);
if(!m)return alert('Open a ZenGM league page first');
const lid=m[1],tid=+(prompt('Your team ID?','0')||0);
const db=await new Promise((r,j)=>{
  const o=indexedDB.open('league'+lid);
  o.onsuccess=e=>r(e.target.result);o.onerror=j});
const A=s=>new Promise((r,j)=>{
  const q=db.transaction(s,'readonly').objectStore(s).getAll();
  q.onsuccess=e=>r(e.target.result);q.onerror=j});
const[pl,ga]=await Promise.all([A('players'),A('gameAttributes')]);
const tm=await A('teams').catch(()=>[]);
const body=JSON.stringify({players:pl,gameAttributes:ga,teams:tm,tid});
try{
  const d=await(await fetch('http://localhost:PORT/evaluate',{
    method:'POST',headers:{'Content-Type':'application/json'},body})).json();
  const rank=d.my_rank,tot=d.total_teams,gap=d.gap_to_top;
  const medal=rank===1?'🏆':rank<=3?'🥇 contender':'⚠ not yet';
  const top5=(d.rankings||[]).slice(0,5).map((t,i)=>
    `${i+1}${t.is_mine?' ▶':'.  '} ${t.team.padEnd(6)} J=${t.j.toFixed(1)}`).join('\\n');
  const syn=d.my_synergy;
  alert(`${medal}\\nRank: ${rank}/${tot}  Gap to #1: ${gap>0?'+':''}${(-gap).toFixed(1)}J\\n\\nSynergy — Off:${syn.off.toFixed(3)} Def:${syn.def.toFixed(3)} Reb:${syn.reb.toFixed(3)}\\n\\nTop 5 teams:\\n${top5}\\n\\nYour lineup:\\n${(d.my_lineup||[]).join(', ')}`)
}catch(e){
  navigator.clipboard.writeText(body)
    .then(()=>alert('Blocked. Payload copied.\\nRun: python main.py --serve\\nError: '+e.message))
    .catch(()=>alert('Error: '+e.message))
}
})()"""


def _minify_bm(raw: str, port: int, scheme: str) -> str:
    import re
    body = raw.replace("\n", "").replace("PORT", str(port))
    body = body.replace("http://localhost", f"{scheme}://localhost")
    return f"javascript:{re.sub(r'  +', ' ', body)}"


def make_bookmarklet(port: int = DEFAULT_PORT, scheme: str = "http") -> str:
    """Return the trade-finder `javascript:` URL."""
    return _minify_bm(BOOKMARKLET_BODY, port, scheme)


def make_eval_bookmarklet(port: int = DEFAULT_PORT, scheme: str = "http") -> str:
    """Return the team-strength-evaluator `javascript:` URL."""
    return _minify_bm(BOOKMARKLET_EVAL_BODY, port, scheme)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Handle numpy scalars and pandas NA values."""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def _safe_dumps(obj: Any) -> bytes:
    return json.dumps(obj, default=_json_default).encode()


def _serialise_player(p: dict) -> dict:
    """Keep only display-relevant fields so the response stays compact."""
    return {
        "pid":    p.get("pid"),
        "name":   p.get("name") or f"pid_{p.get('pid','?')}",
        "age":    p.get("age"),
        "pos":    p.get("pos"),
        "salary": p.get("salary"),
    }


def _serialise_trades(trades: list[dict]) -> list[dict]:
    return [
        {
            "team":             t["team"],
            "trade_type":       t["trade_type"],
            "net_lineup_score": t.get("net_lineup_score"),
            "dv":               t.get("dv"),
            "new_score":        t.get("new_score"),
            "incoming":         [_serialise_player(p) for p in t.get("incoming", [])],
            "outgoing":         [_serialise_player(p) for p in t.get("outgoing", [])],
        }
        for t in trades
    ]


def _serialise_sequences(sequences: list) -> list[dict]:
    result = []
    for s in sequences:
        steps = [
            {
                "team":       step.team,
                "trade_type": step.trade_type,
                "dv":         step.dv,
                "j_before":   step.j_before,
                "j_after":    step.j_after,
                "incoming":   [_serialise_player(p) for p in step.incoming],
                "outgoing":   [_serialise_player(p) for p in step.outgoing],
            }
            for step in s.sequence
        ]
        result.append({
            "j_start":      s.j_start,
            "j_final":      s.j_final,
            "total_j_gain": s.total_j_gain,
            "steps":        steps,
        })
    return result


# ---------------------------------------------------------------------------
# Core solver pipeline
# ---------------------------------------------------------------------------


def solve(payload: dict) -> dict:
    """
    Run the full solver pipeline on a ZenGM-shaped payload dict.

    Parameters
    ----------
    payload : dict with keys 'players', 'gameAttributes', 'teams', 'tid'.

    Returns
    -------
    JSON-serialisable dict with 'trades' and 'sequences'.
    """
    my_tid = int(payload.get("tid", 0))
    my_roster_df, league_rosters_dict, cap_info = parse_league_data(
        payload, my_tid=my_tid
    )

    all_rosters = {"__mine__": my_roster_df, **league_rosters_dict}
    league = league_value_stats(
        all_rosters,
        current_season=cap_info.get("current_season", 0),
        salary_cap=cap_info["salary_cap"],
        salary_cap_type=cap_info["salary_cap_type"],
        soft_cap_trade_match=cap_info["soft_cap_trade_match"],
    )

    log.info(
        "Solving for tid=%d  my=%d players  opponents=%d teams  cap=$%.0fK",
        my_tid, len(my_roster_df), len(league_rosters_dict), cap_info["salary_cap"],
    )

    trades = find_best_trades(
        my_roster_df,
        league_rosters_dict,
        league=league,
        salary_cap=cap_info["salary_cap"],
        top_n=5,
    )

    return {
        "cap":       cap_info["salary_cap"],
        "trades":    _serialise_trades(trades),
        "sequences": [],
    }


# ---------------------------------------------------------------------------
# Team strength evaluator
# ---------------------------------------------------------------------------


def evaluate(payload: dict) -> dict:
    """
    Rank all teams in the league by lineup strength and return my team's
    position, synergy breakdown, and championship contention status.
    """
    my_tid = int(payload.get("tid", 0))
    my_roster_df, league_rosters_dict, cap_info = parse_league_data(
        payload, my_tid=my_tid
    )

    def _team_j(df):
        caches = [_build_cache(df.iloc[i]) for i in range(len(df))]
        return _compute_team_j(caches)

    # My team
    my_j       = _team_j(my_roster_df)
    my_detail  = optimize_rotation(my_roster_df)
    my_syn     = my_detail["synergy"]
    my_lineup  = [
        str(my_detail["lineup"].iloc[i].get("name") or f"pid_{my_detail['lineup'].iloc[i].get('pid','?')}")
        for i in range(len(my_detail["lineup"]))
    ]

    # All opponent teams
    rankings = []
    for team_name, their_df in league_rosters_dict.items():
        if len(their_df) < 5:
            continue
        rankings.append({"team": team_name, "j": _team_j(their_df), "is_mine": False})

    rankings.append({"team": "YOUR TEAM", "j": my_j, "is_mine": True})
    rankings.sort(key=lambda x: x["j"], reverse=True)

    my_rank   = next(i + 1 for i, t in enumerate(rankings) if t["is_mine"])
    top_j     = rankings[0]["j"]
    gap       = my_j - top_j   # negative = behind leader

    log.info("Evaluate: rank %d/%d  J=%.1f  gap=%.1f", my_rank, len(rankings), my_j, gap)

    return {
        "my_rank":     my_rank,
        "total_teams": len(rankings),
        "my_j":        my_j,
        "top_j":       top_j,
        "gap_to_top":  gap,
        "my_synergy":  {"off": my_syn["off"], "def": my_syn["def"], "reb": my_syn["reb"]},
        "my_lineup":   my_lineup,
        "rankings":    rankings,
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class SolverHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self._send(200, {})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in ("/solve", "/evaluate"):
            self._send(404, {"error": "endpoints: POST /solve  POST /evaluate"})
            return
        try:
            length  = int(self.headers.get("Content-Length", 0))
            body    = self.rfile.read(length)
            payload = json.loads(body)
            result  = solve(payload) if path == "/solve" else evaluate(payload)
            self._send(200, result)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            log.exception("Server error on %s", path)
            self._send(500, {"error": str(exc)})

    def _send(self, code: int, body: dict) -> None:
        data = _safe_dumps(body)
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in _CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:   # suppress default log
        log.debug(fmt, *args)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server(
    port: int = DEFAULT_PORT,
    ssl_cert: str | None = None,
    ssl_key:  str | None = None,
) -> None:
    """
    Start the solver HTTP server and block until Ctrl-C.

    Parameters
    ----------
    port     : TCP port to listen on (default 8888).
    ssl_cert : path to PEM certificate file for HTTPS (optional).
    ssl_key  : path to PEM private-key file for HTTPS (optional).

    HTTPS / mixed-content note
    --------------------------
    Browsers block HTTP requests from HTTPS pages (play.basketball-gm.com).
    To enable HTTPS:
        openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \\
                -days 365 -nodes -subj '/CN=localhost'
        python main.py --serve --ssl-cert cert.pem --ssl-key key.pem
    Then visit https://localhost:<port>/solve in the browser once and accept
    the self-signed-cert warning before using the bookmarklet.
    """
    scheme = "http"
    httpd  = HTTPServer(("localhost", port), SolverHandler)

    if ssl_cert and ssl_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(ssl_cert, ssl_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    bm      = make_bookmarklet(port, scheme=scheme)
    bm_eval = make_eval_bookmarklet(port, scheme=scheme)
    print()
    print("=" * 70)
    print(f"  BBGM Solver server  —  {scheme}://localhost:{port}")
    print("=" * 70)
    print()
    print("── BOOKMARKLET 1: Trade Finder ──────────────────────────────────────")
    print("  Save as a bookmark, click on any ZenGM league page.")
    print()
    print(bm)
    print()
    print("── BOOKMARKLET 2: Team Strength / Championship Readiness ────────────")
    print("  Ranks all teams by lineup strength; shows your synergy breakdown.")
    print()
    print(bm_eval)
    print()
    print("── NOTES ───────────────────────────────────────────────────────────")
    if scheme == "http":
        print("  ⚠  play.basketball-gm.com is served over HTTPS; the bookmarklet")
        print("     fetch will be blocked by mixed-content rules.")
        print("     Fallback: the bookmarklet copies the JSON payload to the")
        print("     clipboard so you can paste it as a file or use stdin.")
        print("     Robust fix: run with a self-signed HTTPS cert:")
        print("       openssl req -x509 -newkey rsa:4096 -keyout key.pem \\")
        print("               -out cert.pem -days 365 -nodes -subj '/CN=localhost'")
        print(f"       python main.py --serve --port {port} \\")
        print("               --ssl-cert cert.pem --ssl-key key.pem")
        print("     Then open https://localhost:{port}/ once and accept the cert.")
    else:
        print(f"  ✓ HTTPS enabled — bookmarklet can POST from play.basketball-gm.com")
    print()
    print("  Press Ctrl-C to stop.")
    print("=" * 70)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()
        sys.exit(0)
