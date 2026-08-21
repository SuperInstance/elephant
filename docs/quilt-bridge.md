# Quilt Bridge — the cell-ledger seam

*2026-08-21 · the elephant's readings, ledgered*

The elephant **computes** room fields; the quilt cell-ledger (**quilt** /
**quilt-rust**, the reactive cellular runtime) **records** them. Every
reading the elephant makes is a delta: the field *after* a message minus
the field *before* it. That before→after pair is exactly the directed
edge the quilt cell-ledger stores — `{v:1, cell, ts, before, after,
delta, imbalance, provenance, chain}`. This file documents the seam from
the elephant's side.

## The identity, already proven

`quilt-rust/docs/field-edge-ledger-bridge.md` proves the mathematical
identity quilt-side:

- **`imbalance ≡ d_mu`** — the cell-ledger's double-entry imbalance and
  the elephant's field-edge are two projections of one object, golden
  vectors verified to 1e-12 in
  `quilt-rust/crates/field-edge-bridge/bridge_demo.py`.
- **The honesty gates coincide.** The ledger's null-prior is the
  elephant's deadband / NMIN — both refuse to book a change that is
  noise.
- **One fractal edge.** `quilt-rust/docs/fleet-as-fractal-jepa.md` names
  the elephant's field-edge as one zoom (the room) of the same fractal
  edge that appears at pin, model, and fleet scales.

## What the elephant produces

The elephant's seam is its existing reading path — no code change
needed:

- `elephant/room.py` — rooms as message streams; a message is a
  `before → after` transition in the room's field.
- `elephant/field.py` — `RoomField`: `warmth()`, `concentration()` (κ),
  `distance()`, `sauna_plunge_gap()` — the delta vector.
- `elephant/dial.py` — `DialBank`, the dial ensemble whose readings form
  a natural cell value (e.g. a `tap.field.warmth` cell recomputed on
  message).
- `elephant/jepa_rag.py` — readings already ride along as RAG metadata.

## The bridge is read-only on both sides

Neither repo changes for the seam to exist: the elephant keeps computing
fields, the ledger keeps sealing edges. A producer-side
`record_with(expected)` (5-line embeddable form) is designed quilt-side
but not yet wired — this document is the elephant owning its half of the
story until then.

## See also

- `quilt-rust/docs/field-edge-ledger-bridge.md` — the proof
- `quilt-rust/docs/fleet-as-fractal-jepa.md` — the fractal framing
- `quilt-rust/docs/cell-ledger.md` — the ledger wire contract
