# Basketball GM Solver — Build Plan (v1, keep-it-simple)

A staged plan for building a near-optimal GM solver for ZenGM Basketball. The guiding principle: **start with the cheap analytic model and greedy search, prove it works in the real game, and only add complexity (deeper search, RL) where the simple version measurably falls short.**

---

## Guiding Principles

1. **Proxy first, sim second.** The game's `team.ovr` → predicted-MOV regression is a near-analytic objective. Optimize against it first; use the full simulation only to *validate* and *correct*.
2. **Smallest controllable problem first.** Lineup → single trade → multi-asset trade → draft/FA → multi-season. Each builds on the last.
3. **Always validate against the real game.** A solver that wins on the proxy but not in actual ZenGM seasons is broken. Play full seasons and measure real wins/titles.
4. **Defer RL.** Greedy + local search will get you most of the way. RL is Phase 5+, not Phase 1.

---

## Objective Function

What "optimal" means, in one line:

```
maximize   J = α · MOV_playoff(roster)  +  (1 − α) · MOV_regular(roster)
subject to salary cap, roster size (10–15), and the legal action space
```

- `MOV_regular`, `MOV_playoff` from the team-ovr regressions (see model reference doc).
- `α` ∈ [0, 1] tunes "win now / win the title" vs "win the regular season." Start with α ≈ 0.5.
- **Correction:** compute the *real* objective with the synergy-aware efficiency model, not the raw ovr-sum, because the proxy is blind to fit/usage/depth interactions.

---

## Phase 0 — Foundations (scaffolding)

**Goal:** get game state in and out, and have a trusted reimplementation of the scoring math.

- [ ] **Data I/O.** Decide how to read/write league state. Simplest path:
  - Export a league file (JSON) or stats CSV from the ZenGM UI.
  - Solve offline.
  - Apply decisions manually (or via a small script that edits the league JSON and re-imports).
  - *Note:* full automation requires hooking the in-browser worker (`self.bbgm` global) — defer that; manual apply is fine for v1.
- [ ] **Reimplement the scoring functions** in your language (Python prototype already exists): composite ratings, synergy, player ovr, team ovr / predicted MOV.
- [ ] **Validation harness.** Pull a known roster, compute your team ovr, and confirm it matches the number ZenGM displays. This is the trust check for everything downstream.

**Deliverable:** a `model` library that turns a roster of base ratings into team ovr and predicted MOV, verified against the game.

---

## Phase 1 — Evaluation Function (the core)

**Goal:** one function `evaluate(roster) → score` that you trust.

- [ ] **Tier 1 (fast, analytic):** `team.ovr` → predicted MOV. Use for bulk pruning.
- [ ] **Tier 2 (synergy-aware efficiency model):** the per-possession ORtg/DRtg model built from the documented probabilities (turnover, shot type, make prob, rebound, synergy). Already prototyped — evaluates a *lineup*, not just a roster. Use for ranking finalists.
- [ ] **Tier 3 (ground truth):** the actual simulation. Two options, simplest first:
  - **(a) Faithful Monte Carlo:** reimplement the possession loop and run N possessions. Slower but portable.
  - **(b) Real GameSim:** play seasons inside ZenGM with your decisions and read back results. The ultimate check.
- [ ] **Pick the blend.** v1: rank with Tier 1, re-rank top candidates with Tier 2, validate finalists with Tier 3.

**Deliverable:** a 3-tier evaluator and a documented rule for when to use each tier.

---

## Phase 2 — Action Space and Constraints

**Goal:** model the levers the GM actually controls.

| Lever | v1 scope | Constraints to encode |
|---|---|---|
| **Lineup / rotation** | choose best starting 5 + rotation | none mechanical (positions are labels) |
| **Trades** | 1-for-1, 2-for-1 | trade value match, salary matching rules, cap |
| **Draft** | pick best available by value | pick slot, prospect ratings/potential |
| **Free agency** | sign value-positive players | salary cap, contract length |
| **Contracts** | re-sign / let walk | cap space, player age curve |

- [ ] Encode a **trade-value proxy** (ovr-driven, youth/upside bonus, salary drag) — prototyped.
- [ ] Encode **salary cap + roster size** as hard constraints.
- [ ] Start by treating draft/FA/contracts as *read-only* (don't optimize them yet); focus on lineup + trades.

**Deliverable:** functions that enumerate legal actions for each lever, with constraints applied.

---

## Phase 3 — Search / Optimization (simple first)

**Goal:** find good decisions with the dumbest method that works.

- [ ] **Lineup:** brute-force enumerate (C(9,5) ≈ 126 combos), rank by Tier-2 evaluator. *Done.*
- [ ] **Single trade:** enumerate value-balanced deals league-wide, recompute team ovr + best-lineup net, keep improvers. *Done.*
- [ ] **Multi-asset trade:** bounded search over 2–3 player packages targeting the league's top alphas. The natural next step.
- [ ] **Roster hill-climbing:** from the current roster, repeatedly apply the single best available action (trade / signing / cut) until no action improves `J`. This is your v1 "solver."

**Deliverable:** a greedy/local-search loop that outputs a ranked list of recommended actions.

---

## Phase 4 — Validation and Iteration

**Goal:** prove the solver actually wins, and fix the eval where it doesn't.

- [ ] **Baseline.** Sim N full seasons in ZenGM with no intervention; record wins, playoff results, titles.
- [ ] **Solver run.** Apply the solver's recommended moves; sim the same N seasons; compare.
- [ ] **Diagnose gaps.** Wherever the proxy said "good" but the sim disagreed, identify the missing factor (usually synergy, usage collision, defense averaging, or fatigue) and fold it into the Tier-2 model.
- [ ] **Repeat.** Each loop tightens the proxy toward the sim.

**Deliverable:** a measured win-rate / title-rate improvement over baseline, plus a more accurate evaluator.

---

## Phase 5+ — Advanced (defer until Phase 1–4 are solid)

Only reach for these once greedy + local search plateaus:

- **Multi-season planning.** Player aging curves, prospect development, and contract timing make this a sequential decision problem — the natural home for RL or deep tree search.
- **RL agent.** State = league + roster; actions = the Phase-2 levers; reward = the Tier-3 sim outcome (wins / titles). Train against the real game as the environment.
- **Full automation.** Hook the in-browser worker so the solve→apply→sim loop runs without manual steps.

These are explicitly **out of scope for v1.**

---

## Suggested Tech Stack (v1)

- **Language:** Python (fast to prototype; the scoring math is already ported).
- **Core libs:** `pandas`/`numpy` for roster math; `itertools` for enumeration. No ML framework needed yet.
- **Game interface:** manual JSON/CSV export-import to start.
- **Ground truth:** play seasons in ZenGM directly.

---

## Recommended Order to Start Tomorrow

1. Finish Phase 0: verify your `team.ovr` matches the game exactly. (Trust nothing until this passes.)
2. Lock the Tier-1 + Tier-2 evaluators from Phase 1.
3. Wire up the greedy roster hill-climber (Phase 3) using only lineup + single-trade actions.
4. Run the Phase-4 baseline-vs-solver season comparison on one team (e.g. the Indiana case).
5. Read the gap, improve Tier-2, repeat.

Everything past step 5 — multi-asset trades, draft/FA optimization, RL — is a deliberate "later."
