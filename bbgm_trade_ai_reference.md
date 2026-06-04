# ZenGM Basketball — Trade AI Reference

> Extracted from [zengm-games/zengm](https://github.com/zengm-games/zengm).
> Source: `src/worker/core/trade/{propose,summary,makeItWork,getPickValues,isUntradable}.ts`, `src/worker/core/team/ValueChangeCalculator.ts`, `src/worker/core/player/{value,valueCombineOvrPot}.ts`

This document describes exactly how the CPU decides whether to accept a trade. It is the missing piece that makes proposed trades *plausible*: a trade is only realistic if the receiving AI team's value change `dv > 0`.

---

## TL;DR — Why "my worst player for the league MJ" gets rejected

The AI accepts a trade **iff its own value increases** (`dv > 0`). Value is summed per asset, and the summation applies a brutal nonlinearity:

```
asset_contribution = (v > 1) ? v^7 : v        # EXPONENT = 7 for basketball
```

where `v` is the player's **z-scored** value (standard deviations above league-mean value). A superstar at `v = 2.5` contributes `2.5^7 ≈ 610`. A pile of below-average role players each have `v < 1`, so they contribute only `v` (linearly) — and you'd need **hundreds** of them to match one star. This is by design: stars are nearly impossible to acquire without sending back a comparable star.

**Implication for your solver:** filter every candidate trade through this `dv > 0` test for the *other* team before ever proposing it. That single filter eliminates all the impossible "fleece" trades.

---

## 1. The Acceptance Rule

Source: `trade/propose.ts`

```
dv = ValueChangeCalculator.evaluate({
    tid:            otherTeam,              # the AI team being asked
    pidsAdd:        userPlayersOffered,     # what AI receives
    pidsRemove:     aiPlayersRequested,     # what AI gives up
    dpidsAdd/Remove: ... picks ...,
    tradingPartnerTid: userTid,
})

if (dv > 0)  → ACCEPT
else         → REJECT
```

Rejection message reveals how close you were (useful as a search signal):

| Condition | Message |
|---|---|
| `dv > -2` | "Close, but not quite good enough." |
| `dv > -5` | "That's not a good deal for me." |
| `dv ≤ -5` | "What, are you crazy?!" |

The AI **does not care if the trade is fair to you** — only whether *its* value goes up. There is no separate "is this lopsided" check.

---

## 2. The Value-Change Formula

Source: `team/ValueChangeCalculator.ts → evaluate()`

```
dv = sumValues(assetsAdded, strategy, tid, includeInjuries=true)
   − sumValues(assetsRemoved, strategy, tid)
```

Both sides are built from individual asset values (players + picks), each passed through the adjustments below, then summed with the EXPONENT nonlinearity.

---

## 3. Base Player Value (`p.value`)

Source: `player/value.ts`, `player/valueCombineOvrPot.ts`

A 0–100-ish number representing a player's general worth (independent of team fit). Steps:

1. **Normalize ovr & pot** to a standard league scale (mean ~47, std ~10 for basketball):
   ```
   ovr_norm = (ovr − leagueOvrMean)/leagueOvrStd × 10 + 47
   ```
2. **Blend in recent performance (PER):** if the player has stats,
   ```
   current = 0.8 × ovr_norm + 0.2 × (31.693 + 1.531 × PER_recent)
   ```
   (PER is minutes-weighted over the last 1–2 seasons; scaled down if < 2000 min.)
3. **Combine current with potential by age** (`valueCombineOvrPot`):

   | Age | Value formula |
   |---|---|
   | ≤19 | 0.7·pot + 0.3·cur |
   | 20 | 0.65·pot + 0.35·cur |
   | 21–22 | 0.6·pot + 0.4·cur |
   | 23 | 0.55·pot + 0.45·cur |
   | 24 | 0.45·pot + 0.55·cur |
   | 25 | 0.3·pot + 0.7·cur |
   | 26 | 0.15·pot + 0.85·cur |
   | 27 | 0.025·pot + 0.95·cur |
   | 28 | 0.95·cur |
   | 29 | 0.94·cur |
   | 30 | 0.93·cur |
   | 31–33 | 0.92·cur |
   | 34–38 | 0.91·cur |
   | 39+ | 0.90·cur |

   Young players are valued mostly on **potential**; players 28+ purely on current ability with a small age-decay.

---

## 4. Z-Score Normalization

Source: `ValueChangeCalculator.ts → zscore()`

Before entering the trade math, every player's value is converted to standard deviations above the league mean:

```
v = (p.value − playerOvrMean) / playerOvrStd
```

So `v = 0` is a league-average player, `v = 1` is one std above, `v = 2+` is a star. This is what makes the `v^7` term meaningful.

---

## 5. The EXPONENT Nonlinearity (the heart of it)

Source: `ValueChangeCalculator.ts → sumValues()`

```
EXPONENT = 7   # basketball (baseball/football 3, hockey 3.5)

per_asset = (adjustedValue > 1) ? adjustedValue^7 : adjustedValue
total = Σ per_asset
```

- Below-average and average players (`v ≤ 1`) count **linearly**.
- Stars (`v > 1`) count **to the 7th power**.

Concrete scale (approximate, before other adjustments):

| Player z-value `v` | Contribution |
|---|---|
| 0.5 | 0.5 |
| 1.0 | 1.0 |
| 1.5 | ~17 |
| 2.0 | ~128 |
| 2.5 | ~610 |
| 3.0 | ~2187 |

**This is why depth cannot buy a star, and why one star ≈ one star is the only realistic 1-for-1 swap at the top.**

---

## 6. Team Strategy Adjustment

Source: `sumValues()` — each team has a strategy of `"contending"` or `"rebuilding"` (set elsewhere based on roster age/quality). Applied to `v` **before** the exponent:

### Rebuilding teams (value youth & picks, dump age)
| Asset | Multiplier |
|---|---|
| future draft pick | ×1.10 |
| age ≤19 | ×1.075 |
| age 20 | ×1.05 |
| age 21 | ×1.0375 |
| age 22 | ×1.025 |
| age 23 | ×1.0125 |
| age 27 | ×0.975 |
| age 28 | ×0.95 |
| age ≥29 | ×0.90 |

### Contending teams (don't care about potential, want now)
| Asset | Multiplier |
|---|---|
| future draft pick | ×0.825 |
| age ≤19 | ×0.80 |
| age 20 | ×0.825 |
| age 21 | ×0.85 |
| age 22 | ×0.875 |
| age 23 | ×0.925 |
| age 24 | ×0.95 |

**Implication:** sell youth/picks to rebuilders, sell win-now veterans to contenders. The same player has different value to different teams — this is the lever that makes lopsided-looking-but-accepted trades possible.

---

## 7. Contract Value Adjustment

Source: `getContractValue()` + `sumValues()`

```
expiring contract (exp this season, or next if offseason)  → contractValue = 0  (ignored)

else:
  expectedSalary = slope × (v − MIN_VALUE)              # what a player of this value "should" earn
  contractValue  = expectedSalary − actualSalary/cap     # +overpaid penalty / underpaid bonus
  contractValue  = min(contractValue, 0.1)               # capped boost, uncapped penalty

then added to player value:
  v += contractsFactor × contractValue
  contractsFactor = (strategy === "rebuilding") ? 2 : 0.5
```

- Underpaid players get a small bonus (≤0.1); badly overpaid players get a large penalty.
- Rebuilding teams weight contracts **4× more** than contenders (they care about cap flexibility).
- `MIN_VALUE = −0.5`, `MAX_VALUE = 2` for basketball.

---

## 8. Other Per-Asset Adjustments

Applied in `getPlayers()` / `sumValues()`:

- **Injury** (only for AI's incoming players, `includeInjuries=true`):
  ```
  if gamesRemaining > 75:  v −= 0.75·v
  else:                    v −= v · gamesRemaining/100
  ```
- **Negative-value players:** `if v < 0: v /= 20` — bad players barely register (so the AI doesn't think a trade is impossible just because of throw-ins).
- **Just-drafted players:** floored at 0 (can be cut, so never negative).
- **AI self-overvalue fudge:** in **AI-to-AI** trades only, outgoing positive-value players are ×1.05 (AI is harder to pry from). Not applied when trading with the user.
- **Difficulty fudge:** `1 + 0.1 × difficulty` (e.g. ×0.975 easy, ×1.025 hard, ×1.1 insane) applied to outgoing player values.

---

## 9. Draft Pick Valuation

Source: `getPickInfo()`, `getPickNumber()`, `getEstPicks()`, `trade/getPickValues.ts`

1. **Estimate pick slot.** Teams ranked by estimated win% (blend of current record and team ovr):
   ```
   estWinPct(rank) = 0.25 + 0.5 × (numTeams − 1 − rank)/(numTeams − 1)   # 25%–75%
   ```
   Worse teams → earlier (better) picks.
2. **Future picks regress to a target** (uncertainty): toward `0.25×numPicks` (AI's own, assumed good) or `0.75×numPicks` (user's, assumed bad). Weighted by how many seasons out (0–5).
3. **User-pick penalty / AI-pick bonus** when trading with the user: user picks pushed later by `+numPicks/3.5` (×difficulty); AI picks pulled earlier by `−numPicks/3.5`. The AI systematically thinks *your* picks are worse than they are.
4. **Slot → value** via the `estValues` table (historical value by draft position), then z-scored. Pick value floored at 0.1 (rookies can be cut).
5. **Anti-fleece on bulk picks:** giving away 3+ first-rounders multiplies their value by `1 + (n−2)/5`, so dumping many picks costs more than linear.

---

## 10. Hard Constraints (checked in `summary.ts`)

These block a trade regardless of `dv`:

- **Soft cap (default):** if a team is over the cap, the salary it takes back must be `< softCapTradeSalaryMatch%` of the salary it sends out. **Default = 125%.**
- **Hard cap:** a trade cannot increase a team's payroll if doing so puts/keeps them over the cap.
- **No cap (`salaryCapType: "none"`):** no salary constraint.

### Untradable players (`isUntradable.ts`)
- Expired contracts during the post-playoffs/pre-free-agency window.
- Recently signed/acquired players (`gamesUntilTradable > 0`).
- (God Mode bypasses all of this.)

---

## 11. How the AI Builds a Trade (`makeItWork.ts`)

Relevant for **generating** plausible trades. When the AI fills out or counters a deal it does a greedy search:

- Maintains a `LookingFor` spec (target `positions`, `skills`, whether it wants `draftPicks`, `prospects`, `bestCurrentPlayers`).
- Repeatedly calls `tryAddAsset`: scores each addable asset and adds the one that best moves `dv` toward acceptance (or matches what it's looking for), recomputing `dv` via the ValueChangeCalculator each step.
- Stops when `dv > 0` for both sides (a mutually acceptable deal) or gives up.

You can mirror this exact loop: start from the player you want, then greedily add your assets until the *other* team's `dv > 0` while your own roster value stays acceptable.

---

## 12. Putting It Together — Plausible-Trade Generator (next step)

A trade `T` between you and team `B` is **plausible** iff:

```
1. dv_B = evaluate(B receives your outgoing, gives up incoming) > 0
2. Salary-matching constraint satisfied (≤125% soft cap, or hard-cap rule)
3. No untradable players involved
```

To find *good* plausible trades for your solver:

```
for each opposing team B:
    for each target asset you want from B:
        greedily assemble the minimal package of YOUR assets such that dv_B > 0
            (using B's strategy-adjusted, exponent-weighted value math)
        if constraints pass:
            new_roster = apply(T)
            improvement = J(new_roster) − J(current_roster)     # your win-model objective
            record (T, improvement)
rank by improvement, keep positive ones
```

The key correction vs. your current approach: **score the outgoing/incoming package with the AI's `dv` formula (exponent, strategy, contracts, picks) to test acceptance**, then score the *resulting roster* with your own win model (team ovr / synergy efficiency) to test desirability. Two different value functions, two different purposes:

- **`dv` (this document):** "Will the AI say yes?" — gatekeeper.
- **`J` (win model / prior docs):** "Does this make *my* team better?" — objective.

A trade must pass the first and maximize the second.
