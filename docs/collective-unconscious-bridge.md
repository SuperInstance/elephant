# Collective Unconscious Bridge — the shared memory seam

*2026-08-17 · the elephant's other half*

The elephant **computes** feelings; the collective-unconscious
(**collective-unconscious** repo, TypeScript/Cloudflare) **stores and
retrieves** them. This file documents the seam from the elephant's
side. It is read-only: the elephant's code does not change for the
bridge — its `jepa_rag.py` moments already ARE the contract.

## The two-sided cross-pollination

- **Elephant side (Python).** `elephant/jepa_rag.py` builds moments:
  `moment_from_text`, `moment_from_room`, `moments_from_markdown`.
  Each moment carries `text` (the shadow), `readings` (the 9-dial JEPA
  reading — what the room FELT), `ts` (the time stamp), `space_id`
  (the space stamp), and `meta`.
- **Collective-unconscious side (TypeScript).** `src/readingsIndex.ts`
  ingests those moments and retrieves them by feeling: `queryByText`,
  `queryByReadings` (cosine or range constraints), `queryByField` (the
  perfume query), `queryByTime`, `queryBySpace`, `queryCombined`.

The bridge is the **moments JSON contract** — see
`collective-unconscious/docs/moments-json-contract.md` for the full
spec; the seam script is
`collective-unconscious/scripts/momentsToJson.ts`.

## What the elephant produces

```python
from elephant.jepa_rag import moments_from_markdown, moment_from_room, moment_from_text

# A whole conversation -> one moment (dials feel density, ripple, panic)
moment = moment_from_room(room, space_id="galley", ts=1724000000.0)

# A markdown transcript -> chunked moments, each with its own reading
moments = moments_from_markdown("tap-trades-2026-08-16.md", space_id="the-tap")
```

Dump them to JSON and hand them to the seam:

```python
import json
with open("moments.json", "w") as f:
    json.dump({"moments": moments}, f, indent=2)
```

```bash
# on the collective-unconscious side
npx tsx scripts/momentsToJson.ts --in moments.json --out enriched.json
```

## The contract, in one breath

| field      | type                   | what it is                        |
|------------|------------------------|-----------------------------------|
| `text`     | string                 | the shadow — the witness words    |
| `readings` | dict of dial → float   | the JEPA reading (the 9 dials)    |
| `ts`       | number (epoch seconds) | the time stamp                    |
| `space_id` | string                 | the space stamp                   |
| `meta`     | dict (optional)        | anything else worth riding along  |

The dial order is the vector layout both sides share:
`mood, volume, earnestness, cynicism, joke_landing, panic, presence,
model_vs_code, vision` (the elephant's `JEPA_DIAL_NAMES`).

## Honesty rules

- The TS side **never computes readings from text** — it only stores
  what the elephant computed. A moment without `readings` ingests as a
  zero vector with a loud warning.
- Every retrieved hit carries its `readings` and dial-order
  `readingVector` — the citizen rides along on every hit.
- The elephant's `readings` dict is the source of truth; the seam maps
  `space_id` → `spaceId`, derives `readingVector` in dial order, and
  derives ids from `meta.source` + `meta.chunk` when no `id` is given.

## Why the elephant does not need to change

The bridge is deliberately one JSON document in each direction's own
shape. The elephant's moments already are the contract; the
collective-unconscious already knows how to read them. The two repos
stay independent — the seam is the only thing that touches both, and it
touches them only through JSON.

---

*Computed by the elephant, stored by the collective. The feelings are
first-class citizens on both sides of the bridge.*

— *the elephant's memory engineer, 2026-08-17*
