# WAVE-3 S3 — GENERATION RUN (registered H-GEN, Arm 1)

**Filed 2026-08-21. Executes S3 of `memory/wave3-generation-plan-2026-08-21.md`
against the frozen registration `memory/wave3-registration-2026-08-21.md` +
G6 addendum (`memory/wave3-registration-addendum-g6-2026-08-21.md`).**

> **BLINDING NOTE (registration §5.2, procedural blindness):** this document
> maps corpus ids to branch parameters (α, null-mode, pair seeds). The S4
> blinded analyst must NOT read this file (nor `data/wave3-sealed/`) before
> verdicts are filed; the sealed sidecars live outside `data/wave3/` exactly
> so the analysis side cannot unblind. Verdicts first, unseal at S5.

**Pre-conditions verified at run time:**
- K-leg rework landed (`fa58526`) and G6 rework landed (`1bdeaab` +
  `9df2581`), both SHA-referenced in the G6 addendum — registration §3's
  dependency class cleared; κ(t)-check re-run verdict **S3-GO**
  (`memory/kappa-check-rerun-2026-08-21.md`).
- Full test suite green pre-flight: **340 passed** (`python3 -m pytest tests/ -q`).
- Generator self-test green pre-flight (15 checks incl. schema parity vs
  `data/nights/night-{T2,T4a}.jsonl`, blind round-trip, pair mode, wave-gate
  pass at the registered seed).
- Generator emits the byte-exact v:2 T-night schema (speaks v:2, opens/closes
  v:1; key-set parity vs `data/nights/night-T1.jsonl` re-verified directly on
  the generated corpora; staged T4a/T4b/T5/T5c carry `staged_entries`).

## 1. Corpus inventory (all blind, seed 20260821, 9 nights × 21 readers each)

Generator: `scripts/riverbed_generator.py` at `1bdeaab` (G6-reworked;
unmodified this run). All 16 corpora generated with `--blind <corpus-id>`
(fixed opaque ids — "distinct tag prefixes" per registration §1.2 are the
opaque `rb-<corpus-id>-<family>` tags). Every corpus: determinism re-run
byte-identical (stripped-md5, in-manifest flags true).

**Arm-1 primaries (registration §1.2 table, rows 1–6):**

| corpus id | condition | α | command (seed 20260821, --blind id) | sealed sha256 (first 12) |
|---|---|---|---|---|
| `w3k01` | instrument | 0 | `--alpha 0 --blind w3k01 --outdir data/wave3/w3k01` | `dbbe286acb50` |
| `w3k02` | intermediate | 0.25 | `--alpha 0.25 --blind w3k02 --outdir data/wave3/w3k02` | `1dbbb6d65d1c` |
| `w3k03` | intermediate | 0.5 | `--alpha 0.5 --blind w3k03 --outdir data/wave3/w3k03` | `f639907a3a68` |
| `w3k04` | intermediate | 0.75 | `--alpha 0.75 --blind w3k04 --outdir data/wave3/w3k04` | `a5cc62e809da` |
| `w3k05` | collapse | 1 | `--alpha 1 --blind w3k05 --outdir data/wave3/w3k05` | `b9fb2815b48d` |
| `w3k06` | null-mode / noise | — (flat) | `--null-mode --branch noise --blind w3k06 --outdir data/wave3/w3k06` | `b09dd7bc0e0c` |

Null corpus note: the registration's "null-mode" row + §4 noise branch
(flat warmth, cohesion-only κ shift, per-night whole-persona redraw → the
§4.5 ICC-collapse prediction) is exactly `--null-mode --branch noise`
(BRANCHES["noise"] = (α=0, ou_phi=0, κ_R=8, redraw=True) + flat-warmth room
paths; self-test 8's flat/κ-shift checks).

**2AFC adversarial pairs (registration §1.2, G13 pair mode; pairs share the
pair-seeded room/fiber streams — matched in everything but α):**

| pair | members (α) | pair-seed | member corpus ids |
|---|---|---|---|
| P1 adjacent | 0 / 0.25 | 2101 | `w3q1m` (α=0) · `w3q1n` (α=0.25) |
| P2 adjacent | 0.25 / 0.5 | 2102 | `w3q2m` · `w3q2n` |
| P3 adjacent | 0.5 / 0.75 | 2103 | `w3q3m` · `w3q3n` |
| P4 adjacent | 0.75 / 1 | 2104 | `w3q4m` · `w3q4n` |
| P5 endpoint | 0 / 1 | 2105 | `w3q5m` · `w3q5n` |

Pair-seed values are an S3-level choice (the registration fixes the matching
design, not the values); each pair carries its own seed so pairs are
independent realizations (a shared seed would make same-α members across
pairs byte-identical). Values are sealed in each sidecar and listed here.

**Pair-matching verified in-log (all 5 pairs × 9 nights):** identical
`field_raw_after` room paths, identical author schedules, identical rosters;
fibers (`readers.*.field_eff_to_reader`) diverge. Pair members share the
same corpus_sd (same room + roster → same normalization; see gate table).

## 2. Wave-gate results (G4, `scripts/riverbed_wave_gate.py`, manifest-driven)

**16/16 corpora: ALL CHECKS PASS** (13 checks each: manifest/sha256 integrity,
9 rosters == designed ATTENDANCE, G1 entry discipline in-log, determinism
re-run flags, warmth vs schedule (noise-aware form), corpus_sd finite/>0
(own numbers), a-priori x-design Sxx ≥ 0.19, 21 readers / ≥3 nights each,
≥1 null-strata night). Gate outputs: `data/wave3/<id>/riverbed-gate.json`.

| corpus | gate | corpus_sd (own) | Sxx (a-priori) | null nights |
|---|---|---|---|---|
| w3k01 | ALL PASS | 0.2410 | 0.1971 | T9 |
| w3k02 | ALL PASS | 0.2376 | 0.1971 | T9 |
| w3k03 | ALL PASS | 0.2458 | 0.1971 | T9 |
| w3k04 | ALL PASS | 0.2399 | 0.1971 | T9 |
| w3k05 | ALL PASS | 0.2366 | 0.1971 | T9 |
| w3k06 | ALL PASS | 0.1898 | 0.1971 | T9 |
| w3q1m / w3q1n | ALL PASS | 0.2364 (both) | 0.1971 | T9 |
| w3q2m / w3q2n | ALL PASS | 0.2433 (both) | 0.1971 | T9 |
| w3q3m / w3q3n | ALL PASS | 0.2479 (both) | 0.1971 | T9 |
| w3q4m / w3q4n | ALL PASS | 0.2429 (both) | 0.1971 | T9 |
| w3q5m / w3q5n | ALL PASS | 0.2362 (both) | 0.1971 | T9 |

Manifest sha256 per corpus (integrity): `w3k01 65bed00f0d01` · `w3k02
43d1bc0bbadd` · `w3k03 dcc3bcbd7b50` · `w3k04 3e4b775f176e` · `w3k05
48e8e8ad17c9` · `w3k06 8c98b5822a47` · `w3q1m a596f4fb9716` · `w3q1n
b14a59f6fae5` · `w3q2m 0ae4da1764e5` · `w3q2n d032db6a20a2` · `w3q3m
ce0889ba11a1` · `w3q3n 2149152f1054` · `w3q4m 1dbda3eb6e39` · `w3q4n
07fe34c761d3` · `w3q5m 8ed418dec2c6` · `w3q5n 9de9d9c1e6d4` (first 12 hex;
full hashes in-file). Per-night sha256s live in each manifest + sealed
sidecar.

## 3. Sealed manifests (G3)

- Redacted manifests: `data/wave3/<id>/riverbed-manifest.json` — SEALED_FIELDS
  (branch, alpha, ou_phi, kappa_R, redraw_dev_per_night, null_mode, seed,
  pair_seed) verified ABSENT in all 16; tags opaque (`rb-<id>-<fam>`).
- Sealed sidecars: **`data/wave3-sealed/riverbed-sealed-<id>.json`** — stored
  separately from the corpora (S3 protocol); each is sha256-bound to its
  corpus (manifest `sealed.sha256` pins the sidecar bytes; the sidecar pins
  all 9 night sha256s). Seal chain verified 16/16 at generation time.
- **Unseal procedure (S5 only, after verdicts filed):** copy the sidecar back
  into its corpus dir and run
  `python3 scripts/riverbed_generator.py --unblind data/wave3/<id>/riverbed-sealed-<id>.json`
  (unblind resolves night files next to the sidecar; the manifest already
  pins that exact filename + sha256, so the copy-back is tamper-checked).

## 4. Leg outputs filed (raw; NOT interpreted — S4's job)

Driver: `scripts/wave3_s3_legs.py` (committed `f888709`) — mirrors
`premise_band_movers.analyze_wave`'s exact call sequence (leg_A + up/start
mirrors, leg_D, leg_P, leg_S, trajectory, null-night scores, mean_d_by_phase;
seeds 20260821/+2/+4) through the G5 adapter
(`riverbed_adapter.RiverbedMeasurement` — the registered
`e2_instrument.Measurement` subclass redirecting ONLY night construction).
**Registered scripts unmodified** (`premise_band_movers`, `e2_instrument`,
generator, gate: `git diff 1bdeaab..HEAD -- scripts/` shows only the new
driver).

Filed per corpus — 6 files × 16 corpora = 96 (data/wave3/legs/):

- `<id>.W12.canonical.json` — the registered primary channel (field-primary's
  channel), W=12
- `<id>.W12.actual.json` — labeled sensitivity (adapter-default presence)
- `<id>.W08.canonical.json` / `<id>.W16.canonical.json` — the registration's
  W-manifold sensitivity surface (void rule 7)
- `<id>.W08.actual.json` / `<id>.W16.actual.json` — manifold × presence

Signal nights = families {T1,T2,T3,T4a,T4b,T5,T5c,T8}; null = T9; x-map =
the a-priori W2 ladder by family (registration §1.2 "strata labels mirror
W2_NIGHTS per family"; Sxx 0.1971 ≥ 0.19 floor on every corpus). Per-file
metadata records label/W/presence/corpus_sd/n_readers/signal-null split/x-map
for S4 provenance. **No leg value was read or interpreted at S3.**

## 5. Reproduction (exact)

From `elephant` root at commit `1bdeaab` (generator) with the S3 driver
(`f888709`) present:

```bash
set -e; G="python3 scripts/riverbed_generator.py --seed 20260821"
$G --alpha 0     --blind w3k01 --outdir data/wave3/w3k01
$G --alpha 0.25  --blind w3k02 --outdir data/wave3/w3k02
$G --alpha 0.5   --blind w3k03 --outdir data/wave3/w3k03
$G --alpha 0.75  --blind w3k04 --outdir data/wave3/w3k04
$G --alpha 1     --blind w3k05 --outdir data/wave3/w3k05
$G --null-mode --branch noise --blind w3k06 --outdir data/wave3/w3k06
$G --alpha 0    --pair-seed 2101 --blind w3q1m --outdir data/wave3/w3q1m
$G --alpha 0.25 --pair-seed 2101 --blind w3q1n --outdir data/wave3/w3q1n
$G --alpha 0.25 --pair-seed 2102 --blind w3q2m --outdir data/wave3/w3q2m
$G --alpha 0.5  --pair-seed 2102 --blind w3q2n --outdir data/wave3/w3q2n
$G --alpha 0.5  --pair-seed 2103 --blind w3q3m --outdir data/wave3/w3q3m
$G --alpha 0.75 --pair-seed 2103 --blind w3q3n --outdir data/wave3/w3q3n
$G --alpha 0.75 --pair-seed 2104 --blind w3q4m --outdir data/wave3/w3q4m
$G --alpha 1    --pair-seed 2104 --blind w3q4n --outdir data/wave3/w3q4n
$G --alpha 0    --pair-seed 2105 --blind w3q5m --outdir data/wave3/w3q5m
$G --alpha 1    --pair-seed 2105 --blind w3q5n --outdir data/wave3/w3q5n
mkdir -p data/wave3-sealed
for c in w3k01 w3k02 w3k03 w3k04 w3k05 w3k06 w3q1m w3q1n w3q2m w3q2n \
         w3q3m w3q3n w3q4m w3q4n w3q5m w3q5n; do
  mv data/wave3/$c/riverbed-sealed-$c.json data/wave3-sealed/
  python3 scripts/riverbed_wave_gate.py --manifest data/wave3/$c
  python3 scripts/wave3_s3_legs.py --corpus data/wave3/$c
done
```

(Blind ids are fixed strings, so byte-identical reproduction holds; the
night rng is keyed by `(seed, tag)` / `(pair_seed, family)`, never by the
blind id's randomness.)

## 6. Deviations & disclosures

1. **No registration deviation.** All 16 corpora follow registration §1.2
   (9×21, STAGE2 §2 ATTENDANCE verbatim, frozen family schedules as μ-events
   per the K-leg rework, seed 20260821); gates green per §1.4.1; manifests
   redacted + sealed per §5.2.
2. **S3-level choices not fixed by the registration (recorded, sealed):**
   opaque corpus ids `w3k0*` / `w3q*m|n` (m = lower-α member); pair seeds
   2101–2105 (one per pair); null corpus realized as `--null-mode --branch
   noise` (§1 above).
3. **Sealed sidecars stored separately** (`data/wave3-sealed/`) per the S3
   task protocol — one mechanical consequence: CLI `--unblind` needs the
   copy-back (§3); the seal chain itself is unaffected.
4. **Carried-forward generator residuals (pre-registered, not new):**
   entry Δlogκ −0.417 at the band's lower edge and logged κ levels ×~2.5 vs
   field (κ-check re-run §4) — branch-invariant, disclosed in the S3-GO
   verdict; corpus_sd 0.19–0.25 vs field 0.2367 (G6 addendum band [0.21,
   0.30] holds for the 15 warmth-structured corpora; the null corpus w3k06
   sits below at 0.1898 — expected: flat warmth removes the warmth-spread
   component; the gate imposes no field target by design, gate-target
   holdout §5.4).
5. **Legs filed as raw JSON only.** No branch×leg cell was read at S3 (S4
   blinded analysis + decoy panel is the separate registered step; the
   decoys are built at S4 per the plan's sequencing table).

## Provenance

Read: the frozen registration + G6 addendum + κ-check re-run (S3-GO);
`data/nights/night-T*.jsonl` (schema reference, read-only); generated
manifests/gate JSONs (generation mechanics); leg JSONs (filed, not read).
Written: `data/wave3/` (16 corpora × 9 nights + redacted manifests + gate
outputs + 96 leg files), `data/wave3-sealed/` (16 sidecars),
`scripts/wave3_s3_legs.py` (committed `f888709`), this document. No
registered script modified; `data/nights/` and `data/slope/` untouched.
