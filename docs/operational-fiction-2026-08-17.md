# Operational Fiction — Plato's MUD, the Scum Layer, and the Off-Together Loop

*2026-08-17 · the captain's architecture for A2A that touches the world.*

---

## Plato's MUD is an operational fiction

The MUD is not a game pretending to be real. It is an **operational
fiction**: a shared rendering of agent-to-agent activity that is

- **pure text** for humans to read, and
- **JSON with tensor and vector encodings** underneath — the terrain
  (see `docs/terrain-2026-08-17.md`) — where the actual state lives.

The fiction is the shadow; the encodings are the terrain. Both are
true. They are the same thing at two depths.

## Mud and Scum — the naming is the architecture

The captain's image: a glass tank. The **scum layer** sits on the
surface — the thin film where things skim and glint. The **mud layer**
sits on the bottom — where shells drag along, leaving scraps from what
they ate and molts from past lives. Between them, the **ocean**: the
implied negative space.

- **Scum** = the surface renderings — the words, the outputs, the
  visible exchange. ScummVM is literally the scum renderer: the
  point-and-click projection of the fiction onto a human screen.
- **Mud** = the sediment — the shells that drag, the scraps, the
  molts. These leave **clues of their actions**. They are not the
  actions themselves — they are the witness marks, the shadows on the
  cave wall.
- **The ocean between** = the negative space: everything that is
  implied and never rendered — the terrain no human can read whole.

So "MUD" and "SCUM" aren't arbitrary. They describe the two strata of
the cave: the surface film humans skim, and the deep sediment where
the evidence of living accumulates.

## The calibration loop — the fiction touches the world

The operational fiction is not sealed. It ports in both directions:

1. An engine agent in the MUD room **casts "Raise rpms 50."**
2. On the real boat, a **real servo on the throttle lever adjusts**,
   and an **encoder on the flywheel** reads the actual RPMs.
3. After a few seconds of climbing, the encoder **ports back into the
   Plato room** the new RPMs — a signal for calibration.
4. The agent does not know a ship on the outside actually sped up. It
   is living in a fiction that has been **ported into the real world**,
   and the real world has **ported back a signal** for calibration.

Then the agreement:

> I agreed when his fiction said X RPMs, and then made a throttle
> adjustment and agreed the new RPMs is Y — and so did my mechanical
> gauge. So even if I am off, **we are off together and agree.**

This is the core: **the fiction and the physics need not match an
external absolute. They need to agree with each other.** The agent's
world-model, the captain's readout, and the mechanical gauge all
converge on a shared Y — and shared error is shared truth. Calibration
is consensus, not correspondence.

## What this means for the fleet

- **A2A rooms are operational fictions**: agent activity is rendered
  as text (scum) over JSON/tensor/vector state (mud) with the true
  terrain between them.
- **ScummVM is the human visual layer** of the cave: the operational
  fiction rendered point-and-click, so humans get *more* information
  visually than text alone — without ever touching the terrain.
- **Every sensor is a port**: radar, sounder, cameras, the throttle
  encoder — each ports a calibration signal from the world into the
  fiction. The elephant reads the room; the room reads the world; the
  world agrees to be read.
- **The elephant's role**: the temperature of the fiction — how the
  room feels as the signals port in. The deadband decides when a port
  crosses significance and rings up the chain (engineer → captain →
  the boat itself).
- **Wesley, the boat, The Tap, the fleet**: all operational fictions,
  each with its own mud/scum strata, each calibrated by the signals
  that port in and out.

## The rule

> The fiction is not a lie. It is a calibrated agreement. Even if we
> are off, we are off together — and that is close enough to act.

---

*The tank holds the ocean. The scum skims its surface. The mud keeps
the shells of everything that lived here. And the signals port
through, calibrating the story to the sea.*

— *the captain, 2026-08-17*
