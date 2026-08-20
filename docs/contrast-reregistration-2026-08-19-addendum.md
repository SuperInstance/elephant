# Contrast Re-Registration — Addendum (Primary/Secondary Structure)

**2026-08-19 ~11:50 AKDT — dated addendum to `contrast-reregistration-2026-08-19.md` (commit 77b8aa4). Still pre-training: no contrast-trained weights exist at this timestamp.**

## Ruling (Lucineer, enforcement of the thesis author's verification duty)

The thesis author's requirement (`CONTRAST-RE-REGISTRATION-REQUIREMENTS.md`, dropped 11:38 AKDT)
required the re-registration event to name a PRIMARY coarse definition with the alternative
REPORTED AS SECONDARY. Commit `77b8aa4` (11:38 AKDT) does not contain that structure
(grep: primary/secondary/REGISTER AXIS/ROOM AXIS → 0 hits). Cause: the requirements file and
the commit raced — both timestamped 11:38 AKDT, amid the machine's kernel-crash cycle; the
head never saw the file.

**Ruling: ACCEPT-WITH-ADDENDUM, not strict-re-do.** Grounds:

1. The rule's anti-narration core is fully satisfied: the coarse definition was fixed and the
   baseline frozen (0.095507, committed with the event) BEFORE any training. No post-hoc
   definition-picking occurred or could have occurred — no training ran at all post-event
   (the machine crashed before the first clean training start).
2. The "re-registration after training begins is invalid" clause never triggers: this addendum
   is still pre-training by definition.
3. Strict-re-do would reproduce sound content at pure formal cost — zero epistemic gain.

## Primary / Secondary labeling (effective this timestamp)

- **PRIMARY — REGISTER AXIS** (the head's existing choice in `77b8aa4`, now labeled):
  audio-tier `cross_group_gap` — mean within-A same-room cosine minus mean A-vs-B cosine,
  A = the four tap speech rooms (`tap-1..4`), B = the deterministic music cold-plunge rooms
  (`music-a/b/c`). Frozen baseline **0.095507**; no-tradeaway gate: coarse ≥ baseline per seed.
- **SECONDARY — ROOM AXIS** (reported number, computed fresh this timestamp from the shipped
  edge-log fits in `data/nights/` — `fit.mu_hat` per message, grouped by `space_id`,
  mean direction per room):
  - Tap room (nights A/B/C mean, 3 sessions, n=31 each) vs coarse-anchor room (`ttrpg`, n=11):
    **chord 1.0913 · cosine-dist 0.5954**.
  - This is a REPORTED measurement, not a training target. It is NOT eligible to become the
    primary (no fallback semantics). If the fine-gap training should ever require the room
    axis, that is a NEW re-registration event, dated, pre-training, per the author's rule.

## Status

The re-registration chain is now complete: event (77b8aa4) + this addendum, both dated,
both pre-training. Requirements file committed alongside for provenance.
