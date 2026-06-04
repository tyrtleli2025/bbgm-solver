# bbgm-solver — context for Claude Code

This project finds optimal Basketball GM moves for ZenGM. Two reference docs are
ground truth: `engine_reference.md` (game sim / win model) and
`bbgm_trade_ai_reference.md` (the CPU trade-accept logic). Match them exactly.

## Critical architectural rule: TWO separate value functions
- **dv (the GATEKEEPER):** "Will the AI accept?" — a faithful port of ZenGM's
  ValueChangeCalculator (z-scored value, EXPONENT=7, team strategy, contracts).
  The AI accepts iff the OTHER team's dv > 0.
- **J (the OBJECTIVE):** "Does this make MY team better?" — the existing
  lineup/synergy win model in optimizer.py / market_scanner.py's _fast_optimize.
Never collapse these into one symmetric value test. The current code's bug is
using J-style symmetric value matching as the acceptance gate.

## Conventions
- Make changes test-first; every new module gets a tests/ file. Run pytest before finishing.
- Keep the fast _PlayerCache / _fast_optimize path — it's the objective J and it's fast.