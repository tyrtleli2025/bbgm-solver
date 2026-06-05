# bbgm-solver — context for Claude Code

This project finds optimal Basketball GM moves for ZenGM. Three reference docs are
ground truth: `engine_reference.md` (game sim / win model), `bbgm_trade_ai_reference.md` 
(the CPU trade-accept logic), and `develop_freeagency_draft_reference.md` (player aging, 
contracts, draft generation). Match them exactly.

## Critical architectural rule: THREE separate value functions

- **dv (the GATEKEEPER):** "Will the AI accept?" — a faithful port of ZenGM's
ValueChangeCalculator (z-scored value, EXPONENT=7, team strategy, contracts).
The AI accepts iff the OTHER team's dv > 0.

- **J (the OLD OBJECTIVE):** "Current-season synergy win model" — the existing 
lineup/synergy formula in optimizer.py / market_scanner.py's _fast_optimize.
One-step reward only; does NOT account for player development or future value.
Keep this for validation/comparison, but it's being replaced.

- **V (the NEW OBJECTIVE):** "Multi-year championship equity" — deterministic 
projection of rosters 5 years forward using the player development model, 
converting team strength to title probability via softmax, then summing 
discounted probabilities. V = Σ γ^t × P(title in season t). This is the metric 
that replaces J everywhere because it accounts for aging, contracts, draft picks, 
and prospect upside. Never use J as an acceptance gate (that was the old bug).

## Architecture layers (in build order)

1. **project.py** — Deterministic player rating projection using the aging model 
   from develop_freeagency_draft_reference.md. Takes current ratings + age, outputs 
   projected ratings N years forward. No noise; uses expected values.

2. **value.py** — The V function. Projects all rosters forward, evaluates team 
   strength, converts to title probability, computes discounted sum. Uses ACTUAL 
   prospect ratings from the league export (not VALUE_BY_PICK averages) to account 
   for draft class strength.

3. **Integration** — Replace J with ΔV in the beam search trade evaluator. Keep dv 
   gatekeeper unchanged. A trade is profitable when ΔV > 0 AND other team's dv > 0.

## Conventions

- Make changes test-first; every new module gets a tests/ file. Run pytest before finishing.
- Keep the fast _PlayerCache / _fast_optimize path — it's the old J model and still useful for validation.
- γ (discount factor), H (horizon), beta (softmax temperature), and replacement_level_ovr should be configurable parameters with sensible defaults (γ=0.95, H=5, beta=0.15, replacement_level_ovr=40).