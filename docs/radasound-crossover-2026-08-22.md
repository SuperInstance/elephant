# RADASOUND — the iceberg's tier-2 crossover

**Filed 2026-08-22 · elephant repo (this doc only; Scrapcraft main untouched — convergence lane owns it).**
Sea legs meet the room-field: the boat's sensor dials crossed with Scrapcraft's
party arbitration, yard ecosystem, and teacher transcript. Reference module
below is TypeScript-in-a-doc, not game code.

---

## TL;DR

Three mappings, one shared function — `readField(frames) → { warmth, κ, dials }` —
ported from `field.py` so both sides call the *same* first-order aggregate over
their own local frames. **Verdict: the port is a now-build (elephant-side, this
doc); game wiring is season-two, riding the game-lay spine; anything that ships
kid-derived signals off-device is a beautiful no.** The Switch Test downgrade
(reader-delta → mean-shift) is the caution that keeps every claim first-order.

---

## 1. The mapping

### 1.1 RadarCoherence → party arbitration coherence (`PartyNudger`)

The boat's `RadarCoherenceDial` reads the *distribution of boats*: tight
cluster = on fish, scattered = searching, plus a trend bonus. Scrapcraft's
`PartyNudger` already produces the analogous distribution every tick: each
companion's candidate topics with `weight × jitter` scores. The port reads the
*spread of those scores* — companions converging on one pull (clustered) vs.
kibitzing in all directions (scattered).

- **Game gains:** one honest number — "is the crew pulling together?" It can
  gate the 30% objection probability (scattered party argues more), and it
  gives the teacher console a coherence line without inventing new telemetry.
- **Boat keeps:** `kinematics()` (three-frame track association, accel,
  spread-rate) — meaningless for topics, and it's the part of the dial the
  boat is still hardening. The trope deduction ("together = on fish") never
  crosses either; the game just reads tight or loose.
- **Must NOT cross:** the dial's input is **game-authored companion state**
  (code the game already holds locally), not child input. No candidate-score
  streams off-device.

### 1.2 SounderBiomass → yard density / bot population

`SounderBiomassDial` reads the density under the keel: recent-5 mean plus
trend, "a look, a texture, felt through experience." Scrapcraft's yard has the
same shape: active bots, mining rate, crash/repair cadence — the ecosystem's
biomass. Mean + trend over a session = is the yard thickening.

- **Game gains:** an ambient "the yard is alive" dial (drives DayNight /
  NightShift flavor), and a *context* column for teachers — engagement shape,
  explicitly **not** learning evidence. The Logbook stays the evidence.
- **Boat keeps:** the JEPA attention nudge toward the water column
  (`nudge.py`'s sounder→prior wiring) — that's the boat's perception loop.
- **Must NOT cross:** kid play-pattern streams. Biomass is computed from
  local counters, rendered locally, exported only as a scalar in the existing
  consented class export if the convergence lane opts in. No live monitoring
  channel is created by RADARSOUND.

### 1.3 warmth / κ → Logbook analytics (teacher transcript)

`field.py`'s `RoomField.warmth()` and `concentration()` over the dial bank,
run on-device over the room of companion banter + game events. The Logbook's
`transcript()` gains a per-day footer: `yard day 12 — warmth +0.31, κ 1.8,
14 concepts met`. Teacher reads the shape of the day next to the memory
entries.

- **Game gains:** a two-number session summary in an already-sacred,
  append-only artifact. Cheap, legible, non-judgmental.
- **Boat keeps:** the vMF machinery, registered corpora, acclimation /
  charisma inversion — unvalidated weight for a kids' game. Only the
  pragmatic v0 crosses (normalized dial vector, weighted warmth, κ =
  ‖centered field‖·2).
- **Must NOT cross:** kid free text (Spark questions) as dial input. Dials
  read game-authored state; exported entries are game-authored memory
  phrases plus dial scalars. **The kid's own words never leave the device.**

### The boundary, spelled out once

1. All dial computation is on-device, offline, dependency-free.
2. Dial inputs: game-authored companion/event state and scalar counters.
   Never raw child text, voice, or keystroke timing.
3. Exports ride *existing* consented surfaces only (Logbook transcript /
   class CSV). RADARSOUND opens no new channel, adds no PII, phones nothing.
4. The boat keeps every raw feed; only the *number* ever traveled, and here
   even the number stays home. ClassRoom's COPPA posture (no email, no PII)
   is the floor, not the ceiling.

---

## 2. The Scrapcraft-side API sketch (reference, in-doc only)

```ts
// roomfield.ts — RADARSOUND port of elephant field.py + sensors.py (pragmatic v0).
// Pure, offline, zero deps. Computes on-device; exports nothing on its own.
// Rendered dials: captain-console style, teacher-visible only.

export type DialName = 'mood' | 'volume' | 'earnestness' | 'cynicism'
  | 'joke_landing' | 'panic' | 'presence' | 'radar_coherence' | 'yard_biomass';
export const DIAL_NAMES: DialName[] = ['mood','volume','earnestness','cynicism',
  'joke_landing','panic','presence','radar_coherence','yard_biomass'];

/** A local frame. `data` holds scalars ONLY — never child text. */
export interface Frame { ts: number; sensor: string; data: number[] }

const mean = (xs: number[]) => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;
const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Party "radar": spread of companion top-candidate scores across ticks.
 *  +1 = crew converging (on fish) · −1 = scattered (searching). */
export function radarCoherence(frames: Frame[], n = 3): number {
  const f = frames.filter(x => x.sensor === 'radar').slice(-n);
  if (!f.length) return 0;
  const spreads = f.map(fr => { const m = mean(fr.data);
    return mean(fr.data.map(s => Math.abs(s - m))); });
  const base = 1 - Math.min(1, mean(spreads) / 0.5);
  const trend = spreads.length >= 2 ? spreads[spreads.length-1] - spreads[0] : 0;
  return clip(base * 2 - 1 - 0.3 * trend, -1, 1);   // closing = tighter
}

/** Yard "sounder": biomass = activity density, recent mean + trend. [0..1] */
export function yardBiomass(frames: Frame[], n = 5): number {
  const f = frames.filter(x => x.sensor === 'sounder').slice(-n).map(x => x.data[0]);
  if (!f.length) return 0;
  const base = clip(mean(f), 0, 1);
  const trend = f.length >= 2 ? f[f.length-1] - f[0] : 0;
  return clip(base + 0.2 * clip(trend, -1, 1), 0, 1);
}

export class RoomField {
  constructor(public readings: Partial<Record<DialName, number>>) {}
  vector(): number[] { return DIAL_NAMES.map(n => this.readings[n] ?? 0); }
  /** field.py weights: panic/cynicism cold, presence/earnestness warm. */
  warmth(): number {
    const r = this.readings, c2 = (x = 0.5) => (x - 0.5) * 2;
    return clip(0.30 * (r.mood ?? 0) + 0.15 * (r.joke_landing ?? 0)
      + 0.10 * c2(r.earnestness) + 0.10 * c2(r.presence) + 0.10 * c2(r.volume)
      - 0.15 * (r.cynicism ?? 0.5) - 0.10 * (r.panic ?? 0), -1, 1);
  }
  /** κ = ‖centered field‖·2 — tight room vs many-ways room. */
  concentration(): number {
    const v = this.vector().map(x => x - 0.5);
    return 2 * Math.sqrt(v.reduce((a, b) => a + b * b, 0));
  }
  distance(other: RoomField): number {
    const a = this.vector(), b = other.vector();
    const na = Math.hypot(...a) || 1, nb = Math.hypot(...b) || 1;
    return Math.hypot(...a.map((x, i) => x / na - b[i] / nb));
  }
}

/** THE shared function — both sides call this over their own local frames.
 *  First-order aggregate only (Switch-Test honest): reads the step,
 *  never claims the change-of-reading. */
export function readField(frames: Frame[]): RoomField {
  return new RoomField({
    radar_coherence: radarCoherence(frames),
    yard_biomass: yardBiomass(frames),
    // mood/volume/… filled by the game's own local event->dial adapters
  });
}

/** Logbook footer line — the only output shape RADARSOUND adds. */
export function dayFooter(f: RoomField): string {
  return `warmth ${f.warmth() >= 0 ? '+' : ''}${f.warmth().toFixed(2)}, `
       + `κ ${f.concentration().toFixed(1)}`;
}
```

---

## 3. The boat-side payoff — hundred boats made concrete

Scrapcraft's tested patterns flow *back* to F/V EILEEN:

- **Quests-as-declarative-data** (`schema.js`: typed objectives, validated,
  zero-DOM) → **watch templates as JSON**: a watch is a quest chain (log
  check, course change, haul watch), boat-authored, validated by the same
  discipline the elephant harness already enforces.
- **The Logbook pattern** (append-only, dated memory entries in the actor's
  own voice, `transcript()` plain-text export) → **the watchstanding log**:
  every completed evolution appends one memory entry ("2115 — crossed the
  bar, flood tide, no wood — with Wesley"), and `transcript()` produces the
  coast-guard-ready text the skipper hands over. Sacred like a paper
  logbook: never rewritten.
- **The anti-nag contract** (one nudge per topic, global cooldown, suppressed
  mid-flow, crash suppression) → **watch-alarm hygiene**: the same four rules
  are literally good alarm discipline on a working deck. A wheelhouse
  companion that nudges like Rivet nags like a good mate, not a fire alarm.

That's the hundred-boats doctrine concretized: every boat ships the same
schema, every watch writes the same append-only logbook, every skipper
exports the same transcript — "the scoreboard makes the learning visible"
becomes "the logbook makes the watchstanding visible."

---

## 4. Honest verdict

**Now-build (elephant side):** this doc + the reference module above. Pure
port of code that already works (the pragmatic v0 survived every audit; vMF
gate 1 closed 2026-08-19). Zero game-repo changes, zero privacy surface,
reversible.

**Season-two (game side, convergence lane's call):** wiring `readField()` into
the captain console and Logbook footer. It's data-only (~a day) *but* the
game-lay spine (16–23 dev-days) must land first — RADARSOUND dials decorate a
transcript that only matters once the spine makes the transcript chapter-
shaped. Wire it in the same PR-train as the spine, not before.

**Beautiful no:** cloud warmth aggregation, live classroom monitoring, any
fleet-of-kids analytics. COPPA floor plus the boat's own doctrine ("the dial
runs on the boat, shares only its NUMBER, never the feed") — and here we keep
even the number home.

**The Switch Test caution (cited where it bites):** the second-order
reader-delta was downgraded to "mean-shift, baseline-relative delta — reads
the step, not the change-of-reading" (drift-reader missed its own detection
0.467 vs 0.80; a static baseline beat it on localization, r 0.816/0.800 vs
0.435/0.467). So RADARSOUND ships **first-order aggregates only**: warmth, κ,
coherence, biomass over a window. No "the game senses the learner drifting"
claims — that instrument was tested and it isn't licensed yet. If season two
wants second-order readings, they enter through the wave-4 registration door
(pre-stated branches, sealed corpora), never through a kids' game UI.

---

## Provenance

Read (read-only): `elephant/sensors.py`, `field.py`, `nudge.py`,
`dials/__init__.py`, `docs/wave4-registration-draft-2026-08-22.md`; MEMORY.md
elephant/nurse/Switch-Test sections; Scrapcraft `src/companion/nudge.js`
(PartyNudger), `party.js`, `src/quests/{Logbook,LogbookPanel,schema}.js`,
`src/ClassRoom.js`, `teacher.html`, and `docs/GAME-LAY.md` (branch `game-lay`,
c075512). Written: this document only. No game code touched; no frozen,
sealed, or registered elephant artifact modified.
