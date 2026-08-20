# E5 — Identity-Propagation Null: The Class-Residual Decomposition, Corrected and Tested

**Dated: 2026-08-20.** Sharpens the E2 side-by-side's E5 bullet ("class-residual ratio 0.4366 vs population 0.6088 — most of the baseline spread is archetype structure") into a measured, tested claim — and **corrects a bug in the filed number** that makes the claim substantially *stronger* than filed. Everything here runs on the committed E2 field corpus and instrument (`scripts/e2_instrument.py` via `scripts/e2_field.py`), canonical presence, 15 readers, 9 primary nights, corpus_sd = 0.2367. Script: `scripts/e5_identity_propagation.py` (numpy-only, CPU, read-only; seeds 20260820; writes `data/e5/e5-identity-results.json`). No git commit made.

## 0. The verdict paragraph

**"Most of the baseline spread is archetype structure" is robust — and it was an understatement.** The exact, convention-free decomposition puts **93–96% of baseline variance between archetypes** (informative cells 0.9436, bootstrap 95% CI [0.8783, 0.9943]; permutation against exchangeable archetype labels: **p = 0.0001**), not the ~59% the filed ratio pair implied. The **corrected** class-residual ratio is **0.1342** (95% CI [0.0303, 0.1942]) — the filed 0.4366 was inflated by an in-place-mutation bug in the instrument's class-residualization (demonstrated below; direction: *understating* class structure). What survives archetype-conditioning — the class-independent idiosyncratic component the premise actually needs — is real but **small**: spread 0.10–0.13 corpus-sd vs population 0.4556, ~5.6% of baseline variance, though still person-stable (residual ICC 0.6599 [0.396, 0.775]). The doctrine's fixtures assumed class-independent baselines; in the field, "reader idiosyncrasy" as measured by E2's numerator is **~94% archetype identity propagation**. E5's null is confirmed in its strong form.

## 1. Correction first: the filed 0.3267/0.4366 is buggy

Reproduction attempt: population numbers reproduce **exactly** (E-cont 0.4556/0.6088; E-seg 0.5128/0.6853 — both bit-for-bit against `data/e2/e2-field-results.json`). The class-residual spread does **not**: a clean group-centering gives **0.1005** (ratio **0.1342**) where the instrument files 0.3267.

Cause, in `e2_instrument.py::spread_seg(class_residual=True)`:

```python
for r in present:
    gm = np.mean([vecs[x] for x in groups[self.arch[r]]], axis=0)
    vecs[r] = vecs[r] - gm        # <-- mutates vecs while group means are still read
```

Group means for later members are computed over a **mix of original and already-residualized vectors**, so multi-member groups are never actually centered. Demonstration (run in-script, cell S1/warm): the instrument's loop leaves the critic-group residual mean at |max| = 0.2000 where correct centering gives 1.4e-17; the cell's residual spread reads 0.2984 under the bug vs 0.0945 clean. The filed number reproduces only under the bug.

**Direction of the error:** inflated residual spread ⇒ understated archetype structure. The side-by-side's E5 conclusion survives its own instrument bug — conservative in the right direction. Affected artifacts: `data/e2/e2-field-results.json` (`class_residual` block), E2 REPORT §class-conditional, side-by-side §6.6/§8 ("0.4366 vs 0.6088"). No other E2 quantity touches the buggy path (population spreads, ICC, drift, sensitivities all reproduce exactly). The instrument file itself was **not modified** — fixing a committed registered instrument is a committee decision; the correction is computed independently in `e5_identity_propagation.py`.

A second, lesser correction: the filed comparison mismatched estimators (E-seg class-residual 0.4366 vs E-cont population 0.6088). Like-for-like pairs below.

## 2. The exact decomposition

Per-(cell, dial) one-way ANOVA on cell-median baselines, exact identity SS_btw + SS_wit = SS_tot asserted numerically. Group structure: critic n=5 (critic, barkeep, singer, fiddler, cartographer), poet n=3, essayist n=2, captain n=2, singleton archetypes writer/engineer/drifter. Two conventions bracket the truth, because on nights A/D/D-cold every attendee is a singleton archetype (those cells are trivially 100% between, and contribute exact zeros to any class-residual spread — the filed statistic was deflated by this too):

| decomposition | between-archetype share | perm p |
|---|---|---|
| all 18 signal cells (upper bracket) | **0.9568** | 0.0001 |
| 12 informative cells (≥1 multi-member group; honest test) | **0.9436** | **0.0001** |
| E-cont global baselines (15 readers, like-for-like with the primary) | **0.9325** | 0.0001 |

Informative-cell inventory: S1 (critic×4), S2 (captain, critic, poet pairs), S3 (critic, poet, essayist pairs), S4a (critic×3), S4b (poet×3, critic×2) — the S-family nights carry all the within-archetype contrasts.

Like-for-like residualization (clean):

| pair | population | class-residual | retained |
|---|---|---|---|
| E-seg spread | 0.5128 (ratio 0.6853) | **0.1005** (ratio **0.1342**) | 0.196 of spread |
| E-cont spread | 0.4556 (ratio 0.6088) | **0.1184** (ratio **0.1582**) | 0.260 of spread |

Within-archetype spreads of global baselines (corpus-sd): critic 0.1197, poet 0.2516, essayist 0.1004, captain 0.0464 — against population 0.4556.

## 3. Bootstrap and permutation

Bootstrap over readers (B=2000, seed 20260820; instrument convention, dedup within cells):

- **class-residual ratio (E-seg, clean): 0.1342, 95% CI [0.0303, 0.1942]** — entirely below the 0.3 kill line, and non-overlapping with the population E-seg ratio CI [0.4174, 1.0340].
- **between-share (informative cells): 0.9436, 95% CI [0.8783, 0.9943]** — entirely above 0.5.
- retained-spread share: 0.1717 [0.0433, 0.2671].

Permutation (10,000 shuffles of archetype labels, fixed group-size multiset, seed 20260820), null = archetype exchangeable across readers: between-share p = 0.0001 (informative, all-cells, and E-cont all agree); spread-reduction p = 0.0001. The class structure is not an accident of labeling.

**Drift side (exploratory):** between-archetype share of per-reader drift = 0.2945, **p = 0.666** — archetype does *not* structure drift. Identity propagation shapes the static baseline, not the dynamic response — consistent with E3's common-response dominance (~7:1) and with the premise's own failure mode living in the numerator, not the denominator.

## 4. Sensitivities

| variant | between-share (informative) | perm p | note |
|---|---|---|---|
| primary (canonical) | 0.9436 | 0.0001 | — |
| S5 null cells included | 0.9488 | 0.0001 | 14 informative cells |
| actual-presence instrument | 0.9176 | 0.0002 | class-residual ratio 0.0974 |
| barkeep excluded | 0.9506 | 0.0002 | ratio 0.1488 (14 readers) |
| singleton archetypes out (n=12, 4 groups) | 0.9199 | 0.0001 | within-archetype E-cont spread 0.1335 |

The share never leaves [0.92, 0.96] under any treatment, cell set, or reader set; no p exceeds 0.0002. The propagation-only row kills the obvious objection (that singletons manufacture the structure): with writer/engineer/drifter gone entirely, 92% of the remaining baseline variance is still between the four seeded archetypes.

## 5. What survives conditioning: the residual-ICC

After removing (night × archetype) means, the 12 multi-member-archetype readers still show person-stable baseline structure: **residual ICC = 0.6599, 95% CI [0.3960, 0.7751]** (vs filed population ICC 0.7714 [0.667, 0.810]). So the premise's stability half survives *within* archetype — a real, stable, class-independent idiosyncratic component exists — but at 0.10–0.13 corpus-sd it is roughly **a quarter of the population spread** the premise's fixtures imputed to it. Caveats: n=12, groups of 2–5, several readers with only 2 nights, archetype means estimated from the same data (mild over-removal, conservative for idiosyncrasy).

## 6. Verdict for E5 and the premise

1. **The claim is robust, and stronger than filed.** ~93–96% of E2's baseline-spread variance is between-archetype (exact ANOVA), p = 0.0001 against label exchangeability, CI [0.88, 0.99] on the honest bracket, stable across five sensitivity variants.
2. **The premise's class-independent-baseline assumption is quantitatively wrong as a field description.** The doctrine's fixtures seeded reader baselines as iid idiosyncrasy; the field's "idiosyncratic" spread is ~94% archetype identity propagation. The premise's numerator (0.4556 corpus-sd of "idiosyncratic" baseline structure) should be re-read as mostly *class* structure — which also reframes the side-by-side's numerator-replication across E2/E3: what replicated across instruments is archetype-structured baseline separation, not individual idiosyncrasy.
3. **The corrected class-residual ratio lands below the kill band**: 0.1342 [0.0303, 0.1942], entirely under 0.3. Conditioned on archetype, the premise's own ratio test — the one E2 left INDETERMINATE at 0.6088 — fails decisively. (The registered verdict belongs to the unconditional 15-reader run and stands; this is the E5 conditional reading, which is the null's whole point.)
4. **What E5 does not kill:** a small, stable, genuinely class-independent idiosyncratic component (residual ICC 0.66) and the drift side (no archetype structure, p = 0.666). Identity propagates into *who you are* (baseline), not *how you move* (drift).
5. **Bookkeeping:** the corrected number (0.1342) strengthens the side-by-side's E5 bullet; the filed 0.4366 should be marked superseded in the E2 report and side-by-side (erratum: instrument bug, direction conservative). The correction also matters prospectively for the registered slope regression (H-reader≡room): archetype-conditioned baselines are the right object on its reader side, or its slope will partly measure archetype geometry rather than reader-vs-room alignment.

## Honesty

- The 8 seeded readers were **generated from archetype parameters by construction** (documented procedure in `e2-personas.json`), so some class structure is expected mechanically. The finding is not that seeding "leaked" but that the *field population's measured spread* — the quantity the premise's fixtures modeled as iid — is class-conditional to ~94%, which is what the doctrine's class-independent assumption gets wrong.
- 3 of 15 readers (writer, engineer, drifter) are singleton archetypes and carry zero within-archetype information; A/D/D-cold cells are structurally all-between. The bracket pair [0.9436, 0.9568] and the propagation-only row (0.9199) bound and survive this.
- Permutation p-values at 0.0001 are floor-limited by 10⁴ shuffles (true p ≤ 1.0001e-4).
- The residual-ICC is post hoc (first computed here), n=12, with noisy 2-night readers inside; treat it as a bound-flavored estimate, not a registered quantity.
- Bootstrap resamples readers-level, matching the instrument's convention; archetype-cluster bootstrap (7 clusters) would be wider — with p = 0.0001 and share CI [0.88, 0.99], the conclusion does not hinge on it.

## Reproduce

```
cd /home/eileen/projects/elephant
python3 scripts/e5_identity_propagation.py     # ~3 min, numpy-only, CPU
```

Outputs: console (reproduction + bug demo + decomposition + bootstrap + permutation + sensitivities + verdict inputs) and `data/e5/e5-identity-results.json`. The population-side reproduction doubles as a guard: any drift in the corpus or instrument shows up as a mismatch against the filed 0.4556/0.5128/0.6853/0.6088 (all reproduce exactly, 2026-08-20).
