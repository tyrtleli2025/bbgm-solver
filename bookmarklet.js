/**
 * BBGM Solver Bookmarklet
 * =======================
 * Save this as a browser bookmark with the URL set to the single-line
 * javascript: snippet at the bottom of this file.
 *
 * Prerequisites
 * -------------
 *   python main.py --serve          (starts on http://localhost:8888)
 *
 * What it does
 * ------------
 *   1. Reads the league ID from the ZenGM URL   (/l/{lid}/...)
 *   2. Prompts for your team ID
 *   3. Opens IndexedDB "league{lid}" and reads players + gameAttributes + teams
 *   4. POSTs {players, gameAttributes, teams, tid} to localhost:8888/solve
 *   5. Displays the top-5 trade recommendations in an alert
 *   6. Falls back to copying the JSON payload to the clipboard if the POST
 *      fails (e.g. mixed-content block: HTTPS page → HTTP server)
 *
 * Mixed-content workaround
 * ------------------------
 *   play.basketball-gm.com is served over HTTPS; a plain http://localhost
 *   POST will be blocked. Options:
 *     A) Run with a self-signed HTTPS cert (see main.py --serve --ssl-cert).
 *     B) Use the clipboard fallback: the bookmarklet copies the JSON payload,
 *        then run:
 *            python main.py --file -     (if you add stdin support)
 *        or save the clipboard to a .json file and pass it with --file.
 *
 * Annotated (readable) version
 * -----------------------------
 */
(async () => {
  // ── 1. League ID from URL ────────────────────────────────────────────────
  const m = location.pathname.match(/\/l\/(\d+)/);
  if (!m) return alert('Open a ZenGM league page first (/l/{id}/...)');
  const lid = m[1];

  // ── 2. Team ID ────────────────────────────────────────────────────────────
  const tid = +(prompt('Your team ID (check URL or Team > Settings)', '0') || 0);

  // ── 3. IndexedDB reads ───────────────────────────────────────────────────
  const db = await new Promise((res, rej) => {
    const r = indexedDB.open('league' + lid);
    r.onsuccess = e => res(e.target.result);
    r.onerror   = rej;
  });

  const getAll = store => new Promise((res, rej) => {
    const req = db.transaction(store, 'readonly').objectStore(store).getAll();
    req.onsuccess = e => res(e.target.result);
    req.onerror   = rej;
  });

  const [players, gameAttributes] = await Promise.all([
    getAll('players'),
    getAll('gameAttributes'),
  ]);
  const teams = await getAll('teams').catch(() => []);

  // ── 4. POST to solver ─────────────────────────────────────────────────────
  const body = JSON.stringify({ players, gameAttributes, teams, tid });

  try {
    const response = await fetch('http://localhost:8888/solve', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();

    // ── 5. Display results ─────────────────────────────────────────────────
    const lines = (data.trades || []).slice(0, 5).map((t, i) =>
      `#${i+1} ${t.team} ${t.trade_type}  +${(t.net_lineup_score||0).toFixed(1)}J  dv=${(t.dv||0).toFixed(2)}\n` +
      (t.incoming || []).map(p => `  IN:  ${p.name || p.pid}`).join('\n') + '\n' +
      (t.outgoing || []).map(p => ` OUT:  ${p.name || p.pid}`).join('\n')
    );
    alert(lines.join('\n─────\n') || 'No trades found');

  } catch (err) {
    // ── 6. Clipboard fallback ──────────────────────────────────────────────
    navigator.clipboard.writeText(body)
      .then(() => alert(
        'Server unreachable (HTTPS → HTTP blocked?).\n' +
        'Payload (' + Math.round(body.length / 1024) + ' KB) copied to clipboard.\n' +
        'Start server:  python main.py --serve\n' +
        'Or run with HTTPS cert — see bookmarklet.js for instructions.\n' +
        'Error: ' + err.message
      ))
      .catch(() => alert('Error: ' + err.message));
  }
})();

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * BOOKMARK URL  (copy exactly this one line; < 2000 chars)
 * ═══════════════════════════════════════════════════════════════════════════
 * Paste into browser address bar or bookmark URL field:
 *
javascript:(async()=>{const m=location.pathname.match(/\/l\/(\d+)/);if(!m)return alert('Open a ZenGM league page first');const lid=m[1],tid=+(prompt('Your team ID?','0')||0);const db=await new Promise((r,j)=>{const o=indexedDB.open('league'+lid);o.onsuccess=e=>r(e.target.result);o.onerror=j});const A=s=>new Promise((r,j)=>{const q=db.transaction(s,'readonly').objectStore(s).getAll();q.onsuccess=e=>r(e.target.result);q.onerror=j});const[pl,ga]=await Promise.all([A('players'),A('gameAttributes')]);const tm=await A('teams').catch(()=>[]);const body=JSON.stringify({players:pl,gameAttributes:ga,teams:tm,tid});try{const d=await(await fetch('http://localhost:8888/solve',{method:'POST',headers:{'Content-Type':'application/json'},body})).json();alert((d.trades||[]).slice(0,5).map((t,i)=>`#${i+1} ${t.team} ${t.trade_type} +${(t.net_lineup_score||0).toFixed(1)}J dv=${(t.dv||0).toFixed(2)}\n`+(t.incoming||[]).map(p=>' IN: '+(p.name||p.pid)).join('\n')+'\n'+(t.outgoing||[]).map(p=>'OUT: '+(p.name||p.pid)).join('\n')).join('\n---\n')||'No trades found')}catch(e){navigator.clipboard.writeText(body).then(()=>alert('Blocked (HTTPS→HTTP). Payload copied to clipboard.\nRun: python main.py --serve\nError: '+e.message)).catch(()=>alert(''+e))}})()
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */
