# ZenGM Basketball GM — Complete Simulation Model Reference

> Extracted from the [zengm-games/zengm](https://github.com/zengm-games/zengm) codebase.
> Source files: `src/worker/core/GameSim.basketball/index.ts`, `src/common/constants.basketball.ts`, `src/worker/core/player/ovr.basketball.ts`, `src/worker/core/player/compositeRating.ts`, `src/worker/core/team/ovr.basketball.ts`

---

## Architecture Overview

The model has four layers, each deterministically derived from the one below:

1. **Base Ratings** (per player, 0–100) — the only true degrees of freedom
2. **Composite Ratings** (per player, 0–1) — weighted normalized averages of base ratings
3. **Team Composites + Synergy** (per lineup, per possession) — on-court aggregation with nonlinear interaction terms
4. **Per-Possession Outcome Probabilities** — ratios of offense/defense composites that determine every game event

A game is ~200 simulated possessions. A season is 82 games. The champion wins the playoff bracket. Everything reduces to player ratings.

---

## Layer 1: Base Ratings

Every player is a vector of 15 ratings, each in [0, 100]:

| Engine Key | CSV Column | What It Represents |
|---|---|---|
| `hgt` | Hgt | Height / size |
| `stre` | Str | Strength |
| `spd` | Spd | Speed |
| `jmp` | Jmp | Jumping / vertical |
| `endu` | End | Endurance / stamina |
| `ins` | Ins | Inside scoring ability |
| `dnk` | Dnk | Dunking |
| `ft` | FT | Free throw shooting |
| `fg` | 2Pt | Mid-range / 2-point shooting |
| `tp` | 3Pt | Three-point shooting |
| `oiq` | oIQ | Offensive IQ |
| `diq` | dIQ | Defensive IQ |
| `drb` | Drb | Dribbling / ball handling |
| `pss` | Pss | Passing |
| `reb` | Reb | Rebounding instinct |

These are the *only* inputs. Everything else is derived.

---

## Layer 2: Composite Ratings

### Formula

Each composite rating `c_k` for a player is computed as:

```
c_k = clamp( (Σ_i  w_ki × x_i) / (100 × Σ_i  w_ki),  0,  1 )
```

Where `x_i` is either a base rating (0–100) or a literal constant (e.g., 50, used as a baseline term), and `w_ki` is the weight for component `i` in composite `k`.

Source: `src/worker/core/player/compositeRating.ts`

### Complete Weight Table

Source: `src/common/constants.basketball.ts` → `COMPOSITE_WEIGHTS`

#### Offensive Composites

| Composite | Components → Weights | Skill Label / Cutoff |
|---|---|---|
| **usage** | ins(1.5), dnk(1), fg(1), tp(1), spd(0.5), hgt(0.5), drb(0.5), oiq(0.5) | V / 0.61 |
| **dribbling** | drb(1), spd(1) | B / 0.68 |
| **passing** | drb(0.4), pss(1), oiq(0.5) | Ps / 0.63 |
| **shootingAtRim** | hgt(2), stre(0.3), dnk(0.3), oiq(0.2) | — |
| **shootingLowPost** | hgt(1), stre(0.6), spd(0.2), ins(1), oiq(0.4) | Po / 0.61 |
| **shootingMidRange** | oiq(−0.5), fg(1), stre(0.2) | — |
| **shootingThreePointer** | oiq(0.1), tp(1) | 3 / 0.59 |
| **shootingFT** | ft(1) | — |
| **drawingFouls** | hgt(1), spd(1), drb(1), dnk(1), oiq(1) | — |
| **turnovers** | **50**(0.5), ins(1), pss(1), oiq(−1) | — |

#### Defensive Composites

| Composite | Components → Weights | Skill Label / Cutoff |
|---|---|---|
| **defense** | hgt(1), stre(1), spd(1), jmp(0.5), diq(2) | — |
| **defenseInterior** | hgt(2.5), stre(1), spd(0.5), jmp(0.5), diq(2) | Di / 0.57 |
| **defensePerimeter** | hgt(0.5), stre(0.5), spd(2), jmp(0.5), diq(1) | Dp / 0.61 |
| **blocking** | hgt(2.5), jmp(1.5), diq(0.5) | — |
| **stealing** | **50**(1), spd(1), diq(2) | — |
| **fouling** | **50**(3), hgt(1), diq(−1), spd(−1) | — |

#### Other Composites

| Composite | Components → Weights | Skill Label / Cutoff |
|---|---|---|
| **rebounding** | hgt(2), stre(0.1), jmp(0.1), reb(2), oiq(0.5), diq(0.5) | R / 0.61 |
| **pace** | spd, jmp, dnk, tp, drb, pss (all weight 1) | — |
| **endurance** | **50**(1), endu(1) | — |
| **athleticism** | stre(1), spd(1), jmp(1), hgt(0.75) | A / 0.63 |
| **jumpBall** | hgt(1), jmp(0.25) | — |

**Bold numbers** (e.g., **50**) are literal constants, not ratings — they act as baseline floors.

---

## Layer 3: Team Composites and Synergy

### Team Composite Ratings (per possession)

Updated every possession. For each of the six composites the engine queries at the team level (`dribbling`, `passing`, `rebounding`, `defense`, `defensePerimeter`, `blocking`):

```
C_k = (1/5) × Σ_{p on court}  c_pk × fatigue(energy_p) × perfFactor × foulFactor
      + synergyFactor × S_k
```

Source: `index.ts` → `updateTeamCompositeRatings()`

Where:
- **synergyFactor** = 0.1 (hardcoded default)
- **perfFactor** = `1 − 0.2 × tanh(pointDiff / 60)` — dampens ratings in blowouts (garbage-time mean reversion)
- **foulFactor** — for defensive composites only: 0.9 if at foul limit, 0.75 if over

Synergy `S` is added to composites as follows:
- `dribbling += synergyFactor × synergy.off`
- `passing += synergyFactor × synergy.off`
- `rebounding += synergyFactor × synergy.reb`
- `defense += synergyFactor × synergy.def`
- `defensePerimeter += synergyFactor × synergy.def`
- `blocking += synergyFactor × synergy.def`

### Fatigue Function

```
fatigue(energy) =
    energy + 0.016  (capped at 1.0)
    
    In late-game situations:
    result = (energy + factor) / (1 + factor)
    where factor = 6 − (time_remaining / 60)   [normal games]
                  = 2                            [Elam ending]
```

Energy depletes with playing time and recovers on the bench (+0.016 per possession).

### Synergy (the nonlinear interaction term)

Source: `index.ts` → `updateSynergy()`

This is the only *combinatorial* term — player value is **not** additive because of synergy.

#### Step 1: Count fractional skills on court

For each of 8 skills, sum a sigmoid over the 5 on-court players:

```
sigmoid(x, a, b) = 1 / (1 + e^(−a × (x − b)))

skillsCount["3"]  = Σ sigmoid(c_shootingThreePointer, 15, 0.59)
skillsCount["A"]  = Σ sigmoid(c_athleticism,           15, 0.63)
skillsCount["B"]  = Σ sigmoid(c_dribbling,             15, 0.68)
skillsCount["Di"] = Σ sigmoid(c_defenseInterior,       15, 0.57)
skillsCount["Dp"] = Σ sigmoid(c_defensePerimeter,      15, 0.61)
skillsCount["Po"] = Σ sigmoid(c_shootingLowPost,       15, 0.61)
skillsCount["Ps"] = Σ sigmoid(c_passing,               15, 0.63)
skillsCount["R"]  = Σ sigmoid(c_rebounding,            15, 0.61)
```

Each sigmoid outputs ~0 or ~1 per player (steep with a=15), so skillsCount ∈ [0, 5].

#### Step 2: Offensive synergy

```
off  = 5 × sigmoid(count_3, 3, 2)                                    # shooting (0–5)
off += 3 × sigmoid(count_B, 15, 0.75) + sigmoid(count_B, 5, 1.75)    # ball handling
off += 3 × sigmoid(count_Ps, 15, 0.75) + sigmoid(count_Ps, 5, 1.75) + sigmoid(count_Ps, 5, 2.75)  # passing
off += sigmoid(count_Po, 15, 0.75)                                    # post play
off += sigmoid(count_A, 15, 1.75) + sigmoid(count_A, 5, 2.75)        # athleticism
off /= 17

perimFactor = clamp(sqrt(1 + count_B + count_Ps + count_3) − 1,  0,  2) / 2
off *= (0.5 + 0.5 × perimFactor)
```

**Key insight:** Offensive synergy is *multiplied* by a perimeter factor that rewards having multiple ball-handlers, passers, and shooters. A team of five non-shooters is severely punished.

#### Step 3: Defensive synergy

```
def  = sigmoid(count_Dp, 15, 0.75)                                   # perimeter D
def += 2 × sigmoid(count_Di, 15, 0.75)                               # interior D (weighted 2×)
def += sigmoid(count_A, 5, 2) + sigmoid(count_A, 5, 3.25)            # athleticism
def /= 6
```

#### Step 4: Rebounding synergy

```
reb  = sigmoid(count_R, 15, 0.75) + sigmoid(count_R, 5, 1.75)
reb /= 4
```

---

## Layer 4: Per-Possession Outcome Probabilities

All probabilities are **ratios of offense to defense team composites**. Absolute rating inflation cancels; only differentials matter.

Notation: `O` = offense team, `D` = defense team. `C_x^T` = team T's composite for rating x. All global tuning factors (`turnoverFactor`, `stealFactor`, etc.) default to 1.0.

Source: `index.ts` → `probTov()`, `probStl()`, `probBlk()`, `probAst()`, `doReb()`, `getShotInfo()`, `doShot()`

### Turnover Probability

```
P(turnover) = clamp( 0.14 × C_defense^D  /  (0.5 × (C_dribbling^O + C_passing^O)) )
```

### Steal Probability (given a turnover occurred)

```
P(steal | TO) = clamp( 0.45 × C_defensePerimeter^D  /  (0.5 × (C_dribbling^O + C_passing^O)) )
```

### Block Probability (per shot attempt)

```
P(block) = 0.2 × (C_blocking^D)²
```

### Assist Probability

```
P(assist) = 0.6 × (2 + C_passing^O) / (2 + C_defense^D)
```

If assisted, the made shot gets +0.025 to make probability.

### Shooter Selection

The player who takes the shot is selected proportionally:

```
P(player p shoots) ∝ (c_p_usage × fatigue(energy_p))^1.25
```

With a floor of 5% of total to prevent anyone from being completely frozen out.

### Shot Type Selection

The shooter's composites determine what type of shot is attempted.

#### Three-pointer tendency

The engine applies a two-stage piecewise rescaling to `shootingThreePointer`:

```
s = c_shootingThreePointer

# Stage 1: compress high end (0.55–1.0 → 0.55–0.85)
if s > 0.55:  s = 0.55 + (s − 0.55) × (0.3 / 0.45)

# Stage 2: compress low end (0–0.35 → 0–0.1, 0.35–0.45 → 0.1–0.45)
if s < 0.35:      s2 = s × (0.1 / 0.35)
elif s < 0.45:    s2 = 0.1 + (s − 0.35) × (0.35 / 0.1)
else:             s2 = s

P(attempt 3-pointer) = 0.67 × s2 × threePointTendencyFactor
```

Late-game situations force three-pointers when trailing by 3–10 points in the 4th quarter.

#### Two-pointer type selection

Among non-three-point shots, the type is chosen by comparing three independent random draws:

```
r_midRange = 0.8 × U(0,1) × c_shootingMidRange
r_atRim    = U(0,1) × (c_shootingAtRim + synergyFactor × (synOff^O − synDef^D))
r_lowPost  = U(0,1) × (c_shootingLowPost + synergyFactor × (synOff^O − synDef^D))

type = argmax(r_midRange, r_atRim, r_lowPost)
```

Offensive and defensive synergy shift the atRim and lowPost draws, making easy interior shots more or less likely depending on lineup fit.

### Shot Make Probability

Base make probability by shot type (before defense):

| Type | Formula | P(miss & foul) | P(and-one) |
|---|---|---|---|
| At rim | `0.41 × c_shootingAtRim + 0.54` | 0.37 | 0.25 |
| Low post | `0.32 × c_shootingLowPost + 0.34` | 0.33 | 0.15 |
| Mid-range | `0.32 × c_shootingMidRange + 0.42` | 0.07 | 0.05 |
| Three-pointer | `0.30 × c_3pt_scaled + 0.36` | 0.02 | 0.01 |

#### Defense and synergy adjustment (applied after base, if not blocked)

```
foulFactor = 0.65 × (c_drawingFouls / 0.5)² × foulRateFactor

probMissAndFoul *= foulFactor
probAndOne *= foulFactor

probMake = (probMake_base
            − 0.25 × C_defense^D
            + synergyFactor × (synOff^O − synDef^D)
           ) × fatigue(energy)
```

If the shot was rushed (< 2 seconds remaining, short possession):
```
probMake *= sqrt(possessionLength / 8)
```

### Rebounding

```
P(defensive rebound) = 0.75 × (2 + C_rebounding^D) / (orbFactor × (2 + C_rebounding^O))
P(out of bounds) = 0.10
P(offensive rebound) = 1 − P(drb) − P(oob)
```

An offensive rebound extends the possession (the offense gets another shot attempt).

### Free Throws

On a shooting foul:
- 2-point foul → 2 free throws
- 3-point foul → 3 free throws
- And-one → 1 free throw

Make probability per FT = `c_shootingFT` (the player's shootingFT composite).

### Possession Flow Summary

```
Possession start:
├─ Ball in backcourt → advance clock (1–5 sec)
│  ├─ P(turnover) → check steal → end possession
│  └─ Continue to frontcourt
├─ Select shooter (∝ usage^1.25)
├─ Determine shot timing (clock advances)
├─ (If frontcourt start) another P(turnover) check
├─ Determine shot type (3pt / atRim / lowPost / midRange)
├─ P(block) → blocked → rebound
├─ P(make) → score → end possession
│  └─ P(and-one) → +1 FT
├─ P(miss & foul) → 2 or 3 FTs → end possession
└─ Miss → rebound
   ├─ Defensive rebound → end possession
   └─ Offensive rebound → new shot attempt (loop)
```

---

## Home Court Advantage

Source: `index.ts` → `homeCourtAdvantage()`

Default home court advantage = 1% (configurable via `homeCourtAdvantage` game attribute).

```
homeCourtModifier = homeCourtFactor × clamp(1 + homeCourtAdvantage/100, 0.01, ∞)

For team 0 (home): all compositeRatings *= homeCourtModifier
For team 1 (away): all compositeRatings /= homeCourtModifier

Exception: "turnovers" and "fouling" (negative ratings) get the inverse adjustment.
Exception: "endurance" is never modified.
```

---

## Player Overall Rating (ovr)

Source: `src/worker/core/player/ovr.basketball.ts`

A linear regression fit to predict player value from base ratings:

```
r = 48.5
    + 0.159 × (hgt − 47.5)     ← HIGHEST: height
    + 0.159 × (diq − 46.7)     ← HIGHEST: defensive IQ
    + 0.133 × (oiq − 46.8)     ← offensive IQ
    + 0.123 × (spd − 50.8)     ← speed
    + 0.0777 × (stre − 50.2)
    + 0.0726 × (tp − 47.1)     ← 3pt shooting
    + 0.0632 × (endu − 39.9)
    + 0.062 × (pss − 51.3)
    + 0.059 × (drb − 54.8)
    + 0.051 × (jmp − 48.7)
    + 0.0286 × (dnk − 49.5)
    + 0.0202 × (ft − 47.0)
    + 0.0126 × (ins − 42.4)
    + 0.01 × (fg − 47.0)       ← LOWEST
    + 0.01 × (reb − 51.4)      ← LOWEST
```

A piecewise "fudge factor" is then applied to maintain historical scale:

```
if r >= 68:      fudge = 8
elif r >= 50:    fudge = 4 + (r − 50) × (4/18)
elif r >= 42:    fudge = −5 + (r − 42) × (9/8)
elif r >= 31:    fudge = −5 − (42 − r) × (5/11)
else:            fudge = −10

ovr = clamp(round(r + fudge), 0, 100)
```

### Marginal Rating Values (coefficient ranking)

| Rating | Coefficient | Relative Value |
|---|---|---|
| hgt | 0.159 | 16× |
| diq | 0.159 | 16× |
| oiq | 0.133 | 13× |
| spd | 0.123 | 12× |
| stre | 0.078 | 8× |
| tp | 0.073 | 7× |
| endu | 0.063 | 6× |
| pss | 0.062 | 6× |
| drb | 0.059 | 6× |
| jmp | 0.051 | 5× |
| dnk | 0.029 | 3× |
| ft | 0.020 | 2× |
| ins | 0.013 | 1.3× |
| fg | 0.010 | 1× (baseline) |
| reb | 0.010 | 1× (baseline) |

**A point of hgt or diq is worth 16× a point of fg or reb.**

---

## Team Overall Rating → Predicted Margin of Victory

Source: `src/worker/core/team/ovr.basketball.ts`

This is the game's own closed-form model of team strength. It is a regression fit to simulated season data.

### Formula

Sort roster by player ovr descending: `v[0] ≥ v[1] ≥ ... ≥ v[9]` (top 10 players only).

```
predictedMOV = −k + a × Σ_{i=0}^{9}  e^(b×i) × v[i]
```

| Mode | a | b | k | Interpretation |
|---|---|---|---|---|
| **Regular season** | 0.3334 | −0.1609 | 102.98 | Gentler decay; depth matters |
| **Playoffs** | 0.6388 | −0.2245 | 157.43 | Steeper decay; stars dominate |

### Exponential Weights by Rotation Rank

| Rank | Regular Season Weight | Playoff Weight |
|---|---|---|
| 0 (best) | 0.3334 | 0.6388 |
| 1 | 0.2839 | 0.5090 |
| 2 | 0.2418 | 0.4055 |
| 3 | 0.2060 | 0.3231 |
| 4 | 0.1754 | 0.2574 |
| 5 | 0.1494 | 0.2050 |
| 6 | 0.1273 | 0.1634 |
| 7 | 0.1084 | 0.1302 |
| 8 | 0.0923 | 0.1037 |
| 9 | 0.0786 | 0.0826 |

### Converting to 0–100 Scale

```
rawOVR = (predictedMOV × 50 / 15) + 50
if playoffs: rawOVR −= 40
teamOVR = round(rawOVR)
```

### Strategic Implications

- Player value decays **exponentially** with rotation rank. The best player is worth ~17% more than the 2nd, who is ~17% more than the 3rd, etc.
- In **playoffs**, the top coefficient nearly **doubles** (0.33→0.64) and decay steepens (−0.16→−0.22). Concentrating talent in 1–2 stars beats spreading across depth.
- Players 11–15 contribute **nothing** to team ovr.
- A "win the title" optimizer and a "win 82 games" optimizer have **different optima**.

---

## Positions

```
POSITIONS = ["PG", "G", "SG", "GF", "SF", "F", "PF", "FC", "C"]
```

Positions are descriptive labels, not mechanically enforced — the simulation engine does not restrict which positions can play together. All interactions are through composite ratings and synergy.

---

## Global Tuning Factors

These are game settings (default = 1.0 unless noted) that scale various probabilities:

| Factor | Default | Affects |
|---|---|---|
| `turnoverFactor` | 1.0 | P(turnover) multiplier |
| `stealFactor` | 1.0 | P(steal) multiplier |
| `blockFactor` | 1.0 | P(block) multiplier |
| `assistFactor` | 1.0 | P(assist) multiplier |
| `orbFactor` | 1.0 | Offensive rebound rate (denominator) |
| `foulRateFactor` | 1.0 | Foul frequency multiplier |
| `twoPointAccuracyFactor` | 1.0 | 2PT make probability multiplier |
| `threePointAccuracyFactor` | 1.0 | 3PT make probability multiplier |
| `threePointTendencyFactor` | 1.0 | 3PT attempt frequency multiplier |
| `pace` | 100 | Game pace (possessions per game) |
| `homeCourtAdvantage` | 1 | Home court boost (%) |
| `numPlayersOnCourt` | 5 | Players per side |

---

## Foul Trouble

Source: `index.ts` → `getFoulTroubleLimit()`, `getFoulTroubleFactor()`

The engine tracks personal fouls and reduces player effectiveness as they approach the foul-out threshold:

- **Foul limit** (trigger for caution): scales with game progress; roughly `foulsNeededToFoulOut − 2` by late game
- At foul limit: defensive composites × 0.9, fouling composite × 0.5
- Over foul limit: defensive composites × 0.75, fouling composite × 0.25
- 1 below limit: fouling composite × 0.8

This affects both team composite ratings and individual player selection weights for defensive/fouling actions.

---

## Substitution Logic

Source: `index.ts` → `updatePlayersOnCourt()`

Players are substituted based on a threshold check each possession:

```
if fatigue(energy) > 0.728 and a replacement is eligible:
    substitute the player with the best available by compositeRating × fatigue
```

The engine prefers to play higher-ovr players more minutes but will rest them when energy drops below effective levels. Endurance (the `endu` rating) determines how fast energy depletes.

---

## Key Derived Insights for a Solver

### What the ovr formula misses that the simulation captures

1. **Synergy is nonlinear and combinatorial.** The perimFactor punishes lineups without perimeter skills (shooting + passing + dribbling). Fit matters — five high-ovr non-shooters underperform five slightly-lower-ovr players with diverse skills.

2. **Usage collision.** The engine selects shooters ∝ usage^1.25. Three ball-dominant stars split touches; if any of them are inefficient at their shot type, the team wastes possessions. This isn't captured by ovr at all.

3. **Defense is a team composite, not an individual stat.** The `−0.25 × C_defense^D` penalty on every shot applies the *team average* defense composite. One weak defender drags the whole lineup.

4. **Offensive rebounds are a geometric multiplier.** Each miss has a chance of extending the possession. A lineup with high rebounding gets more second chances, multiplying offensive efficiency by roughly `1 / (1 − P_miss × P_orb)`.

5. **Fatigue gates minutes.** High-endurance players can sustain peak performance longer. A star with 40 endurance vs. 70 endurance may only play 28 vs. 36 minutes, which the per-game impact model doesn't see but the sim captures over a full game.

### The closed-form objective for a GM solver

```
Maximize:  J = E[predictedMOV_playoff]
Subject to: predictedMOV_regular ≥ threshold (enough to make playoffs / secure seeding)
            salary cap constraints
            roster size constraints (10–15 players)
            trade/draft/FA action space

With the correction that J should be computed using the synergy-aware
on-court efficiency model, not the pure ovr-sum team.ovr proxy.
```

The `team.ovr` proxy (exponentially-weighted sum of sorted player ovrs) is cheap and nearly analytic — use it for fast pruning. Then use the full per-possession model (or the actual GameSim) as the reward signal to correct for synergy, usage, and fatigue.
