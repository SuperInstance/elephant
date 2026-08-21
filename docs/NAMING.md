# Naming Doctrine — elephant v1 and the next species

*Captain's ruling, 2026-08-21. This file is the authority on what this repo is
called, why, and what the name means.*

---

## The ruling

1. **This repo keeps the name `elephant` for v1.** The name stays. v1 is what
   it is — perfected for what it is — and it is called `elephant`. No rename,
   no rebrand, no "JEPA-as-the-name" sleight of hand. (The repo used to be
   `SuperInstance/elephant`; other fleet repos got renamed in the org-wide
   master→main pass, but the elephant's name was never in question. This
   ruling makes that explicit and final.)
2. **The NEXT major species earns a new name.** When a successor project
   grows out of this line — a genuinely new animal, not an iteration of this
   one — it gets its own name at birth. Names are earned by new species, not
   re-decorated onto old ones. `elephant` will not be stretched to cover
   something that is no longer the elephant.
3. **The elephant-in-the-room metaphor is the correct framing for JEPA.**
   The elephant is the thing that cannot be put into words but that everyone
   in the room feels. That is what this system measures: the room's ambient
   field — mood, volume, earnestness, cynicism, joke-landing, panic — the
   temperature of *being in the room*. You don't notice it until you walk
   into a different room, and then it's a very different elephant.

## What "JEPA" means in this repo (honest label, honest stub)

- **`JEPA` is the v1 roadmap label, not an implementation claim.** It names
  the direction — the learned sense the hand-crafted dials are the stand-in
  for — not a thing that exists yet.
- `elephant/jepa.py` is a **stub**: the aspirational learned backbone (EMA +
  stop-gradient + VICReg, the same skeleton as fleet-jepa-midi). It is the
  promise, not the product. The hand-crafted dials in `dials/` are the real,
  working v0 senses — keyword heuristics, honestly labeled.
- **The stub keeps its name.** It is not renamed, not deleted, not dressed up
  as an implementation. Calling it `jepa.py` while being honest that it is
  aspirational is the point: the name marks the direction of travel.
- The learned side that *does* exist is `elephant/learned.py` (supervised
  distillation from the hand-crafted dials, with an optional JEPA-shaped
  self-supervised pretraining stage) — see `docs/dial-training-v1.md`.

## Related docs

- `docs/jepa-is-the-elephant.md` (2026-08-17) — the perceptual reframing:
  the elephant as a field over a room, acclimation vs charisma, contrast as
  the training signal. The metaphor's full statement.
- `README.md` — the one-line pointer to this file, under the title.

---

*The name is a promise about the animal, not a label on the crate. This crate
holds the elephant — v1, perfected for what it is. The next species will name
itself.*
