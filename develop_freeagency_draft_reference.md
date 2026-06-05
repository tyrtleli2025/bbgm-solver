# ZenGM Basketball — Development, Free Agency & Draft Reference

> Extracted from [zengm-games/zengm](https://github.com/zengm-games/zengm).
> Source files: `src/worker/core/player/develop.ts`, `developSeason.basketball.ts`, `potEstimator.ts`, `genContract.ts`, `value.ts`, `moodComponents.ts`, `moodInfo.ts`, `genRatings.basketball.ts`, `bonus.ts`, `src/worker/core/freeAgents/*`, `src/worker/core/draft/*`, `src/common/budgetLevels.ts`, `src/common/random.ts`

This document covers the three missing subsystems needed for a horizon-aware solver: how players age, how contracts and free-agency work, and how draft classes are generated.

---

## Part 1 — Player Development / Aging

This is the most important subsystem for a solver. It determines how every player's 15 base ratings change each season, which in turn controls ovr, pot, and value trajectories.

### 1.1 The Development Pipeline

Source: `player/develop.ts`

Each offseason (or when generating prospects), `develop(player, years)` is called:

```
for i in 0..years:
    if not ratings.locked:
        developSeason(ratings, age, coachingLevel)
    age += 1

ratings.ovr = ovr(ratings)              # recalculate overall
ratings.pot = monteCarloPot(ratings, age)  # recalculate potential
ratings.pos = pos(ratings)               # recalculate best position
ratings.skills = skills(ratings)         # recalculate skill badges
```

### 1.2 Per-Season Rating Changes (`developSeason.basketball.ts`)

Each rating (except `hgt`) changes by:

```
new_rating = clamp(old_rating + clamp((baseChange + ageModifier) × uniform(0.4, 1.4),
                                       changeLimits[0], changeLimits[1]),
                   0, 100)
```

Where `baseChange` and `ageModifier` are age-dependent, and each rating has its own formula.

#### Base Change (applies to ALL ratings)

Source: `calcBaseChange(age, coachingLevel)`

| Age | Base Value | Noise (added to base) |
|---|---|---|
| ≤21 | +2 | `clamp(realGauss(0, 5), -4, +20)` |
| 22–25 | +1 | `clamp(realGauss(0, 5), -4, +10)` |
| 26–27 | 0 | `clamp(realGauss(0, 3), -2, +4)` |
| 28–29 | −1 | `clamp(realGauss(0, 3), -2, +4)` |
| 30–31 | −2 | `clamp(realGauss(0, 3), -2, +4)` |
| 32–34 | −3 | `clamp(realGauss(0, 3), -2, +4)` |
| 35–40 | −4 | `clamp(realGauss(0, 3), -2, +4)` |
| 41–43 | −5 | `clamp(realGauss(0, 3), -2, +4)` |
| 44+ | −6 | `clamp(realGauss(0, 3), -2, +4)` |

Then the coaching effect is applied:

```
baseChange *= 1 + (baseChange > 0 ? 1 : -1) × coachingEffect(coachingLevel)
```

Where `coachingEffect(level) = 0.09 × levelToEffect(level)`.

At the default coaching level (34), `levelToEffect(34) ≈ 0`, so coaching has roughly zero effect. At max level (100), the effect is about `0.09 × 1.1 ≈ 0.099`, boosting positive development by ~10% and slowing decline by ~10%.

**Key insight for projections:** Young players (≤21) have mean base change of +2 with high variance (σ=5 Gaussian, clipped to [-4, +20]). This means a young player's per-rating annual change is roughly `uniform(0.4, 1.4) × (2 + noise)`, which can range from about −0.8 to +30.8 before per-rating modifiers. The upside is enormous but rare; the expected path is moderate growth.

#### Per-Rating Age Modifiers

Each rating has an `ageModifier(age)` added to `baseChange` before the uniform multiplier. These modifiers shift certain ratings up or down *on top of* the base:

**Shooting ratings** (`ins`, `ft`, `fg`, `tp`):

| Age | ageModifier | Net effect (base + modifier) |
|---|---|---|
| ≤27 | 0 | Same as base |
| 28–29 | +0.5 | Partially reverses base decline |
| 30–31 | +1.5 | Almost cancels the −2 base |
| 32+ | +2.0 | Almost cancels the −3/−4 base |

**Interpretation:** Shooting skill barely declines with age. The ageModifier counteracts the base decline, so shooters stay effective deep into their careers.

**IQ ratings** (`oiq`, `diq`):

| Age | ageModifier | Net effect |
|---|---|---|
| ≤21 | +4 | Young players gain IQ fast (+6 base) |
| 22–23 | +3 | Still gaining fast (+4 base) |
| 24–27 | 0 | Neutral |
| 28–29 | +0.5 | Partially reverses decline |
| 30–31 | +1.5 | Almost cancels decline |
| 32+ | +2.0 | Almost cancels decline |

**Interpretation:** IQ grows fast when young and barely declines when old — it's the most durable rating category. Combined with its high coefficient in the ovr formula (0.159 for diq), this is crucial.

**IQ change limits** are also age-dependent:

| Age | [min, max] change per season |
|---|---|
| 19 | [−3, +32] |
| 20 | [−3, +27] |
| 21 | [−3, +22] |
| 22 | [−3, +17] |
| 23 | [−3, +12] |
| 24+ | [−3, +9] |

**Speed** (`spd`):

| Age | ageModifier |
|---|---|
| ≤27 | 0 |
| 28–30 | −2 (accelerated decline) |
| 31–35 | −3 |
| 36–40 | −4 |
| 41+ | −8 |

Change limits: [−12, +2]. Speed declines sharply after 27 and is hard-capped from growing much.

**Jumping** (`jmp`):

| Age | ageModifier |
|---|---|
| ≤26 | 0 |
| 27–30 | −3 (sharp decline) |
| 31–35 | −4 |
| 36–40 | −5 |
| 41+ | −10 |

Change limits: [−12, +2]. The most athleticism-punishing aging curve.

**Endurance** (`endu`):

| Age | ageModifier |
|---|---|
| ≤23 | `uniform(0, 9)` — random large boost |
| 24–30 | 0 |
| 31–35 | −2 |
| 36–40 | −4 |
| 41+ | −8 |

Change limits: [−11, +19]. Young players gain endurance rapidly, then it's stable until early 30s.

**Strength** (`stre`):

| Age | ageModifier |
|---|---|
| All ages | 0 |

Change limits: [−∞, +∞]. Strength follows *only* the base change — no extra aging penalty, no protection. It's purely driven by the age-based base change.

**Dribbling, Passing, Rebounding** (`drb`, `pss`, `reb`):

These share the same ageModifier as shooting (partially protects against decline after 27), but with tighter change limits of [−2, +5]. They improve slowly and decline slowly.

**Dunking** (`dnk`):

Same as shooting for ages ≤27 (ageModifier = 0), but for ages 28+ the modifier is only +0.5 (less protection than shooting). Change limits: [−3, +13].

**Height** (`hgt`):

Height does NOT go through the normal formula. Instead:

```
if age ≤ 21:
    if random() > 0.99 and age ≤ 20 and hgt ≤ 99: hgt += 1
    if random() > 0.999 and hgt ≤ 99: hgt += 1
```

~1% chance of +1 height at age ≤20, ~0.1% chance at age 21. After 21, height never changes. Since `hgt` has the highest ovr coefficient (0.159), this is a minor lottery for young prospects.

### 1.3 Development Summary Table (Expected Annual Change)

For a solver doing deterministic projections, here are approximate **expected** rating changes per year (ignoring noise, using midpoint of uniform(0.4, 1.4) = 0.9):

| Rating | Age 19–21 | Age 22–25 | Age 26–27 | Age 28–29 | Age 30–31 | Age 32–34 | Age 35+ |
|---|---|---|---|---|---|---|---|
| **Shooting** (ins,ft,fg,tp) | +1.8 | +0.9 | 0 | −0.45 | −0.45 | −0.9 | −1.8 |
| **IQ** (oiq, diq) | +5.4 | +3.6 | 0 | −0.45 | −0.45 | −0.9 | −1.8 |
| **Speed** (spd) | +1.8 | +0.9 | 0 | −2.7 | −3.6 | −4.5 | −7.2 |
| **Jumping** (jmp) | +1.8 | +0.9 | 0 | −3.6 | −4.5 | −4.5 | −8.1 |
| **Endurance** (endu) | +6.3* | +0.9 | 0 | −0.9 | −1.8 | −3.6 | −7.2 |
| **Strength** (stre) | +1.8 | +0.9 | 0 | −0.9 | −1.8 | −2.7 | −3.6 |
| **Skill** (drb, pss, reb) | +1.8 | +0.9 | 0 | −0.45 | −0.45 | −0.9 | −1.8 |

*Endurance at ≤23 gets an additional `uniform(0,9)` ageModifier on top of base.

**Strategic implications:**
- Players peak at about age 26–27 (base change hits 0)
- Athletic ratings (spd, jmp) decline earliest and hardest — athletic players fall off a cliff after 30
- Skill/shooting/IQ players hold value much longer — the "crafty veteran" archetype is real in the engine
- IQ grows fastest for young players and declines slowest — high-diq/oiq prospects are the safest bets

### 1.4 Potential Estimation

Source: `potEstimator.ts` (basketball-specific fast path)

```
pot = 72.314 + (-2.331 × age) + (0.833 × ovr) + randInt(-2, 2)
```

If `age ≥ 29`, pot simply equals current ovr.

This is a linear approximation of the Monte Carlo method (which simulates development 20 times to age 29 and takes the 75th percentile peak). The fast estimator is used when there are many teams (≥ TOO_MANY_TEAMS_TOO_SLOW).

**Example:** A 19-year-old with ovr 45: `pot = 72.3 − 44.3 + 37.5 ≈ 65.5`
A 25-year-old with ovr 60: `pot = 72.3 − 58.3 + 50.0 ≈ 64.0`

### 1.5 Coaching Effect on Development

Source: `budgetLevels.ts`

```
coachingEffect(level) = 0.09 × levelToEffect(level)

levelToEffect(level):
    x = (3 × (level - 1)) / 99 - 1         # maps level 1→100 to x -1→+2
    if x < 0:  return 1.1 × x               # linear below 0
    else:      return 1.1 × tanh(x)          # saturating above 0
```

| Coaching Level | Effect on Development |
|---|---|
| 1 (minimum) | −9.9% (worse development) |
| 34 (default) | ~0% |
| 67 (high) | +6.5% |
| 100 (maximum) | +9.0% |

The effect multiplies the base change: positive base changes are amplified, negative base changes are amplified in the opposite direction. So good coaching helps young players develop faster AND slows aging decline slightly.

### 1.6 Scout Fuzz (Rating Uncertainty)

Source: `genFuzz.ts`, `budgetLevels.ts`

Each player gets a `fuzz` value when generated:

```
fuzz = clamp(gauss(0, stddev), -cutoff, cutoff)

stddev = 2 - levelToEffect(scoutingLevel)    # range: 1 to 3
cutoff = round((1 - levelToEffect(scoutingLevel)) × 3.5 + 1)  # range: 1 to 8
```

| Scouting Level | Fuzz StdDev | Fuzz Cutoff |
|---|---|---|
| 1 (minimum) | ~3.1 | ~8 |
| 34 (default) | ~2.0 | ~4.5 |
| 100 (maximum) | ~1.0 | ~1 |

When ratings are displayed or used for value calculation with `fuzz=true`, the displayed rating is `rating + fuzz`. Fuzz decreases over time: each year a prospect's draft class advances, their fuzz is divided by `√2`.

**Implication for drafting:** With low scouting, the prospect you think is OVR 55 might actually be OVR 47 or OVR 63. High scouting investment reduces this uncertainty.

---

## Part 2 — Free Agency & Contract Demand

### 2.1 Base Contract Generation

Source: `player/genContract.ts`

The starting point for any contract is:

```
factor = (salaryCapType === "hard") ? 1.6 : 2
factor *= 1.7   # basketball multiplier

amount = ((p.value / 100) - 0.47) × factor × (maxContract - minContract) + minContract
```

With default settings (soft cap, minContract=1200, maxContract=50000):

```
amount = ((p.value / 100) - 0.47) × 3.4 × 48800 + 1200
       = (p.value × 1659.2) - 77883.6 + 1200
```

| Player Value | Raw Contract (thousands/yr) |
|---|---|
| 30 | min (1,200) |
| 40 | min (1,200) |
| 47 | 1,200 (breakpoint) |
| 50 | 6,174 |
| 60 | 22,766 |
| 70 | 39,358 |
| 75+ | 50,000 (max) |

When `randomizeAmount=true` (default for free agents), the amount is multiplied by `clamp(realGauss(1, 0.1), 0, 2)`, adding roughly ±10% noise.

### 2.2 Contract Length (Expiration)

Source: `normalizeContractDemands.ts → getExpiration()`

```
years = 1 + 0.001629 × age² − 0.003661 × (age × ovr) + 0.002178 × ovr²
years = round(years)
years = clamp(years, minContractLength, maxContractLength)
```

With default settings (min=1, max=5):

| Age/OVR | 40 ovr | 50 ovr | 60 ovr | 70 ovr | 80 ovr |
|---|---|---|---|---|---|
| Age 22 | 2 | 2 | 3 | 4 | 5 |
| Age 25 | 1 | 2 | 3 | 4 | 5 |
| Age 28 | 1 | 2 | 3 | 3 | 5 |
| Age 31 | 1 | 2 | 2 | 3 | 4 |
| Age 34 | 2 | 2 | 2 | 3 | 3 |

Young good players get long contracts; old or bad players get short ones.

### 2.3 The Auction System (Basketball)

Source: `normalizeContractDemands.ts`

For basketball, contract amounts are NOT simply generated from the formula above. Instead, an iterative auction runs for **60 rounds** to set market-clearing prices:

1. All free agents (and expiring-contract players) start with their genContract amount
2. Each round, teams bid on players:
   - Teams are shuffled randomly
   - Each team has cap space = `salaryCap - payroll`
   - Team picks players via softmax weighted by `value² × TEMP` (TEMP=0.35 for basketball, PARAM=7.5)
   - A player receiving 0 bids decreases demands; a player receiving 2+ bids increases demands
3. Learning rate decays: `offset = 0.5 × (1 / (1 + round/60))⁴`
4. After 60 rounds, contract amounts have converged to market-clearing levels

**Implication:** In a competitive market (many teams with cap space), star player contracts get bid up toward the max. In a depressed market (teams over the cap), contracts settle lower. The genContract formula is just a starting point.

### 2.4 Contract Demands Decrease Over Time

Source: `freeAgents/decreaseDemands.ts`

Each day during the regular season, unsigned free agents lower their asking price:

```
baseAmount = 50 × sqrt(maxContract / 20000)
           ≈ 50 × sqrt(2.5) ≈ 79 (with default maxContract=50000)

# During regular season, scale by games ratio
factor = 82 / numGames   # (82 is hardcoded, not from settings)

daily_decrease = max(baseAmount × factor, baseAmount)
```

With default settings, a free agent's demands drop by roughly **$79K per day** during the regular season. Over 82 games, that's up to ~$6.5M of total decline. Combined with increasing `numDaysFreeAgent`, this makes mid-season pickups much cheaper than offseason signings.

Additionally, mid-season free agents get short contracts:
- If contract < 1.34 × minContract: contract expires this season
- Otherwise: expires next season

### 2.5 Player Mood System

Source: `player/moodComponents.ts`, `player/moodInfo.ts`

Mood determines (a) whether a player will sign with a team, and (b) a contract amount surcharge. Each component ranges roughly −2 to +2 unless otherwise noted.

#### Mood Components

| Component | Range | Formula |
|---|---|---|
| **Market Size** | [−2, +2] | `−2 + 4 × (numTeams − popRank) / (numTeams − 1)` |
| **Facilities** | [−2, +2] | `2 × levelToEffect(facilitiesLevel)` |
| **Team Performance** | [−∞, +2] | Based on win%, with +0.15 for title. Negative values ×2 for basketball |
| **Hype** | [−2, +2] | `−2 + 4 × hype` (hype is 0–1) |
| **Loyalty** | [0, +∞] | `numSeasonsWithTeam / 8`, plus +2 if re-signing with current team |
| **Trades** | [−∞, 0] | Penalty for teams that traded away many players |
| **Playing Time** | [−∞, +2] | `10 × (playerMinutesFraction − expectedForValueBin)` |
| **Rookie Contract** | [0, +∞] | +8 if on a rookie contract or undrafted |
| **Relatives** | [0, +∞] | +2 per relative on the team |

#### Mood Traits (personality modifiers)

| Trait Code | Name | Effect |
|---|---|---|
| `F` | Fame-seeking | ×2.5 to marketSize, hype, playingTime |
| `L` | Loyal | ×0.5 to marketSize; ×2.5 to loyalty, trades |
| `$` | Money-motivated | ×1.5 to facilities; ×0.5 to marketSize, teamPerformance |
| `W` | Winner | ×0.5 to marketSize, playingTime; ×2.5 to teamPerformance |

#### Difficulty Modulation (user teams)

For user-controlled teams at difficulty > 0:
- Positive components are divided by `1 + difficulty`
- Negative components are multiplied by `1 + difficulty`

This makes free agents harder to attract on higher difficulties.

#### Willingness to Sign

```
sumComponents = Σ all mood components
sumAndStuff = sumComponents - 0.5
             + clamp(numDaysFreeAgent, 0, 30) / 3      # time pressure
             - (valueDiff > 0 ? sqrt(valueDiff) : valueDiff)  # stars are pickier

valueDiff = (p.value - 65) / 2   # basketball threshold

# For AI teams, extra resistance:
if not user_team: sumAndStuff -= 3

probWilling = 1 / (1 + exp(-0.7 × sumAndStuff))
willing = random() < probWilling
```

Players on rookie contracts or in expansion drafts always sign (probWilling = 1).

**Re-signing bonus:** When a player's contract expires and they're considering re-signing with their current team, they get +2 loyalty. For user teams, `valueDiff` is capped at 4 (so your own star isn't punished as much for being good). These bonuses make it significantly easier to re-sign your own players than to poach free agents.

#### Mood's Effect on Contract Amount

```
# Bad mood increases asking price by up to 50%
if not rookieContract and amount > minContract:
    amount *= clamp(1 + (0.5 × (-sumComponents)) / 10, 1.0, 1.5)
```

A player with very negative mood toward your team (sumComponents = −10) asks for 1.5× the normal price. A player with positive mood asks the normal price (no discount below 1.0×).

### 2.6 AI Re-signing Logic

Source: `phase/newPhaseResignPlayers.ts`

When the re-signing phase begins, AI teams decide which expiring players to keep:

1. `normalizeContractDemands` runs the 60-round auction to set market prices
2. For each expiring player on an AI team:
   a. Check if the player is `willing` to re-sign (via mood)
   b. Use `ValueChangeCalculator.evaluate` with `pidsRemove=[player]` to compute `dv` — would the team be worse without this player?
   c. If `dv < 0` (team is worse without them), the player is worth keeping
   d. Skip some low-value min-contract players randomly (50% chance)
   e. If all checks pass, re-sign at the auction-determined contract

**Hard cap additional check:** payroll + new contract must not exceed salary cap.

### 2.7 AI Free Agent Signing

Source: `freeAgents/autoSign.ts`, `freeAgents/getBest.ts`

During free agency, each day:
1. Teams are shuffled randomly
2. Each team has a probability of **skipping** its turn: 90% if rebuilding, 75% if contending
3. The team that doesn't skip evaluates all available FAs:
   - For basketball: sorted by player `value` (since `DRAFT_BY_TEAM_OVR` is false for basketball)
   - Must fit under salary cap
   - Won't sign min-contract players unless roster is 2+ below max size
4. Signs the best affordable player

**Key implication:** AI teams are quite passive in free agency — they skip 75–90% of daily opportunities. This means good free agents often linger, especially mid-season. A solver can exploit this by timing free agent acquisitions.

---

## Part 3 — Draft Class Generation

### 3.1 Draft Class Size

Source: `draft/genPlayersWithoutSaving.ts`

```
normalNumPlayers = round(numDraftRounds × numActiveTeams × 7/6)
```

With defaults (2 rounds, 30 teams): `normalNumPlayers = round(2 × 30 × 7/6) = 70` players.

If `forceRetireAge` or `forceRetireSeasons` are set, the class may be enlarged to keep rosters full.

### 3.2 Individual Player Rating Generation

Source: `player/genRatings.basketball.ts`

#### Step 1: Height

Height in inches is drawn from a custom CDF (cumulative distribution):

| Height | Probability Mass | Cumulative |
|---|---|---|
| 72" (6'0") | 2.9% | 4.2% |
| 73" (6'1") | 4.1% | 8.3% |
| 74" (6'2") | 4.3% | 12.7% |
| 75" (6'3") | 7.0% | 19.7% |
| 76" (6'4") | 7.0% | 26.7% |
| 77" (6'5") | 7.1% | 33.8% |
| 78" (6'6") | 8.2% | 41.9% |
| 79" (6'7") | 10.2% | 52.2% |
| 80" (6'8") | 10.2% | 62.4% |
| 81" (6'9") | 11.6% | 73.9% |
| 82" (6'10") | 9.2% | 83.2% |
| 83" (6'11") | 8.4% | 91.6% |
| 84" (7'0") | 5.2% | 96.7% |
| 85" (7'1") | 1.7% | 98.5% |
| 86" (7'2") | 0.7% | 99.2% |
| 87"+ | <1% | 100% |

Then converted to `hgt` rating (0–100):

```
hgt = clamp((100 × (heightInInches - 66)) / (93 - 66), 0, 100)
```

So 5'6" (66") → hgt 0, 6'6" (78") → hgt 44, 7'0" (84") → hgt 67, 7'9" (93") → hgt 100.

Wingspan adds `randInt(-1, +1)` to the height used for `hgt` (so a 6'8" player could get hgt for a 6'7" or 6'9").

#### Step 2: Player Type (point, wing, big)

Based on height rating:

| Height | Point | Wing | Big |
|---|---|---|---|
| hgt ≥ 59 (≥6'10") | 1% | 4% | 95% |
| hgt ≤ 33 (≤6'3") | 90% | 10% | 0% |
| Middle (6'4"–6'9") | 3% | 67% | 30% |

#### Step 3: Base Ratings

All players start from these baseline values:

```
stre:37  spd:40  jmp:40  endu:17  ins:27  dnk:27
ft:32    fg:32   tp:32   oiq:22   diq:22
drb:37   pss:37  reb:37
```

#### Step 4: Type-Specific Multipliers

| Rating | Point | Wing | Big |
|---|---|---|---|
| stre | 1.0 | 1.0 | 1.2 |
| spd | 1.65 | 1.4 | 1.0 |
| jmp | 1.65 | 1.4 | 1.0 |
| endu | 1.4 | 1.0 | 1.0 |
| ins | 1.0 | 1.0 | 1.6 |
| dnk | 1.0 | 1.5 | 1.5 |
| ft | 1.4 | 1.2 | 0.8 |
| fg | 1.4 | 1.2 | 0.8 |
| tp | 1.4 | 1.2 | 0.8 |
| oiq | 1.2 | 1.0 | 1.0 |
| diq | 1.0 | 1.0 | 1.2 |
| drb | 1.5 | 1.2 | 1.0 |
| pss | 1.5 | 1.0 | 1.0 |
| reb | 1.0 | 1.0 | 1.4 |

#### Step 5: Correlated Random Factors

Four independent random factors add correlation within rating groups:

```
factorAthleticism = clamp(realGauss(1, 0.2), 0.2, 1.2)   # stre, spd, jmp, endu, dnk
factorShooting    = clamp(realGauss(1, 0.2), 0.2, 1.2)   # ft, fg, tp
factorSkill       = clamp(realGauss(1, 0.2), 0.2, 1.2)   # oiq, diq, drb, pss, reb
factorIns         = clamp(realGauss(1, 0.2), 0.2, 1.2)   # ins (inside scoring only)
```

#### Step 6: Final Rating Computation

```
for each rating key:
    rating = clamp(factor × typeFactor × realGauss(baseRating, 3), 0, 100)
```

Where `factor` is the relevant group factor, and `typeFactor` is from the type table.

**Example — a "wing" prospect:**
- `tp` base = 32, typeFactor = 1.2, factor = factorShooting
- If factorShooting = 1.1 (slightly above average): `tp ≈ clamp(1.1 × 1.2 × realGauss(32, 3), 0, 100) ≈ realGauss(42.2, 4.0)`
- So tp likely lands in [34, 50] for a typical wing

### 3.3 Special "Bonus" Players

Source: `draft/genPlayersWithoutSaving.ts`, `player/bonus.ts`

Each draft class has a small chance of producing a "special" player:

```
numSpecialChances = round((4/70) × numPlayers)  # ~4 for a 70-player class

for each of the top numSpecialChances prospects:
    if random() < 1/numSpecialChances:     # on average, exactly 1 special player per class
        for each rating: rating += randInt(0, 10)
        recalculate ovr/pot
```

This creates the occasional generational talent — a player who is a standard deviation or more above the rest of the class.

### 3.4 Age Distribution in Draft Classes

Source: `genPlayersWithoutSaving.ts`

With default `draftAges = [19, 22]` (minMaxAgeDiff = 3):

1. All prospects are generated at age 19 (the minimum draft age)
2. Each year, the highest-potential ~50% (`fractionPerYear = 0.5`) declare for the draft
3. The remaining 50% stay in "college" and develop one more season
4. After all years, remaining players all declare

This means the final draft class has a mix:
- Some 19-year-olds (the best early-declarers)
- Some 20-year-olds (solid but not top-tier)
- Some 21-year-olds (stayed another year)
- Some 22-year-olds (everyone remaining)

Sorting is by `pot + randInt(-50, +50)`, so the "stay vs. declare" decision has significant randomness — some high-pot players stay in college, and some mediocre ones declare early.

**Strategic implication:** Younger draft picks have more development runway. A 19-year-old pick gets 8+ years of improvement before the age-27 peak; a 22-year-old gets 5. The extra years of positive base change (especially the +4/+3 IQ bonus for ≤23) make younger prospects significantly more valuable on a horizon basis.

### 3.5 Rookie Contract Scale

Source: `draft/getRookieSalaries.ts`

```
firstPickSalary = max(maxContract × draftPickAutoContractPercent/100, minContract)
                = max(50000 × 0.25, 1200) = 12,500

excessSalary = firstPickSalary - minContract = 11,300
```

The salary curve has two slopes:
1. **High slope** (first ~10 picks): uses up half of `excessSalary`
2. **Low slope** (rest of round 1 through `draftPickAutoContractRounds`): uses up the other half

| Pick # | Approximate Salary |
|---|---|
| 1 | 12,500 |
| 5 | ~9,850 |
| 10 | ~6,620 |
| 15 | ~5,240 |
| 20 | ~3,860 |
| 30 | ~1,360 |
| 31+ (round 2) | 1,200 (minimum) |

Rookie contract lengths (default `rookieContractLengths = [3, 2]`):
- Round 1: 3 years
- Round 2: 2 years

**This is the cost-control lever that makes draft picks so valuable.** A first-round pick earning $6K on a 3-year deal who develops into a $30K-value player is saving $24K/yr in cap space — that's a massive surplus value that the solver should exploit.

### 3.6 Draft Prospect Fuzz Reduction Over Time

Source: `phase/newPhaseResignPlayers.ts`

Each year during the re-signing phase, draft prospects in future classes have their fuzz reduced:

```
# Prospects 1 year away:
fuzz /= sqrt(2)

# Prospects 2 years away:
fuzz /= sqrt(2)
```

So a prospect 3 years out has their original fuzz. Two years out: `fuzz/√2`. One year out: `fuzz/2`. By draft day, the fuzz has been halved — you have a much clearer picture of who the player really is.

### 3.7 Draft Lottery

Default type: `nba2027` (the NBA's 2027 lottery reform).

Default lottery chances (for 14 non-playoff teams, drawing 4 lottery picks):

```
[140, 140, 140, 125, 105, 90, 75, 60, 45, 30, 20, 15, 10, 5]
```

The bottom 3 teams have equal chances (14% each). After the lottery picks are drawn, remaining teams pick in reverse order of record.

---

## Part 4 — Solver Integration Notes

### 4.1 Projecting Player Trajectories

To project a player's ratings N seasons forward deterministically (for value function V):

```python
def project_ratings(ratings, current_age, years, coaching_level=34):
    projected = copy(ratings)
    for y in range(years):
        age = current_age + y + 1
        base = base_change_expected(age)    # from table in 1.2
        base *= 1 + coaching_effect(coaching_level)  # small adjustment
        
        for rating in ALL_RATINGS:
            modifier = age_modifier(rating, age)
            change = (base + modifier) * 0.9  # E[uniform(0.4,1.4)] = 0.9
            change = clamp(change, limits[rating][0], limits[rating][1])
            projected[rating] = clamp(projected[rating] + change, 0, 100)
    
    return projected
```

Use this to compute future ovr/pot/value for every player on every roster, then feed into the team.ovr formula from the engine reference.

### 4.2 Valuing a Draft Pick Before the Draft

A pick at slot `s` produces a random player from the generation distribution (section 3.2–3.4). To value a pick:

1. Sample many synthetic prospects from the generation model for slot `s`
2. Project each forward through the development model
3. Compute the discounted future V contribution of each, including the rookie-contract surplus value
4. Take the expected value (or a risk-adjusted measure)

A simpler heuristic: use the `estValues` table from the trade AI reference (which maps pick slot → z-scored value), but adjust for the cost-controlled contract savings.

### 4.3 Predicting Re-signing Cost

To predict what a player will demand next offseason:

1. Compute their projected `value` at season end (using development model)
2. Feed into `genContract` formula → base amount
3. Apply mood modifier: `amount × clamp(1 + 0.05 × (-sumComponents), 1, 1.5)`
4. The 60-round auction may shift this ±20%, but genContract is a good central estimate

For your own team, the +2 loyalty bonus and capped valueDiff make re-signing cheaper than the market rate by roughly 10–30%.

### 4.4 Key Asymmetries to Exploit

1. **Rookie contract surplus:** A mid-first-round pick on $5K who develops into a $30K player creates $25K/yr of cap savings. This is the single largest value-creation mechanism.

2. **AI passivity in free agency:** AI teams skip 75–90% of daily signing opportunities. If a player's demands have decreased enough, you can snap up bargains.

3. **Shooting/IQ durability:** Players strong in shooting and IQ hold value much longer than athletic players. When trading for "win now" pieces, prefer skilled veterans over athletic ones — their decline is gentler.

4. **Young IQ growth:** The +4/+3 IQ age modifier for ages ≤21/≤23 means young players with decent base IQ can gain 15–20 points of diq/oiq in 3–4 years. Since diq has a 0.159 ovr coefficient, that translates to a huge ovr jump. Prospect IQ ratings are the best predictor of future value.

5. **Coaching ROI:** Coaching effect at max level is only ~9% on development. It's worth having, but facilities and scouting may offer more solver-relevant advantages (mood for FA signing, fuzz reduction for drafting).
