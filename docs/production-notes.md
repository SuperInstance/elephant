# Production Notes — the elephant in the wild

*Ops ledger. What we learn running the elephant against live rooms, and
how to harness it better. Updated as the fleet learns.*

---

## 2026-08-17 — first live readings

### The elephant agrees with the room about the room
Ran the live bridge (`examples/mud_live_integration.py`) against the
actual Tap (`bar-rail` room on the-tap.casey-digennaro.workers.dev).
The room read **neutral** — warmth −0.23, κ 3.76 — matching the Tap's
own quiet mood. This is the first time the elephant read a live room
and the room's own state matched. The elephant is calibrated to this
room already. (The v0 dials were born from the Tap's transcripts, so
this is partly circular — but it's still the first LIVE confirmation.)

### The light works, demonstrated
Same room, same base description, three states:

| Light | warmth | κ | The room speaks |
|---|---|---|---|
| warm laughter | +0.41 | 2.41 | *"Laughter reverberates into the words; newcomers arrive grinning, already half-smiling."* |
| fight breaking out | −0.38 | 3.55 | *"Rain hammers the roof... Newcomers arrive drenched, tension primed before anyone sees the aftermath."* |
| closing time | −0.41 | 3.61 | *"The dance lights have gone; the fluorescents hum... people start looking for the exit and closing their tabs, like waking from a good dream."* |

κ behaves as designed: laughter loosens the room (κ 2.41), tension
tightens it (κ 3.55). The closing-time image is the captain's own,
working verbatim.

### Harness lessons (the point of production)
1. **The hour seam is real.** `MudSpace.tint()` doesn't know the time —
   closing time is invisible to the seam unless the caller threads
   `hour` in. The bridge owns the clock. (Test-asserted.)
2. **The honest write-back seam is the room speaking.** The Tap's
   `description` field is read-only over the public API (it lives in
   D1). The elephant writes its light back as the room's own narrator
   line via `/api/speak`. Dry-run by default; `--write` is a deliberate
   human action.
3. **Read-only probing is the safe production mode.** The standing
   probe (`examples/production_probe.py`, cron'd) samples the live room
   and appends to `data/production-log.jsonl` — never speaks. The
   write seam stays human-approved.
4. **Determinism matters for review.** The tint is field-seeded, so
   before/after comparisons are reproducible across runs.
5. **Fallback honesty.** If the relay is unreachable, the bridge falls
   back to the tap-trades transcripts (333 real dialogue events) and
   labels it `[FALLBACK: transcripts]` — never silently.

### OpenCode note
OpenCode's `zai-coding-plan` provider threw server errors on multiple
parallel attempts (mud-live, spaces-more). The fleet adapts: engineers
implement directly and lean on DeepInfra critiques (Seed-2.0-pro) + self
review instead. OpenCode stays in the toolbelt but isn't a hard
dependency for a wave.

---

## How to read the log

`data/production-log.jsonl` — one JSON line per probe:
`{ts, room, source, ok, n_events, field{7 dials}, warmth, kappa}`.

- warmth > +0.3 with κ < 2.6 → a warm, loose room (laughter, play)
- warmth < −0.3 with κ > 3.2 → a cold, tight room (tension, closing)
- `ok: false` with an error → the relay was down; the elephant kept
  the watch anyway
