# SILENCE TEST — 2026-08-20

**The registered frame-level falsifier, run.** Registration: `zeroclaw-dissertation/research/topic.md` ~line 48 (hermes idea 1, adopted as a test): *rerun the classification suite with inter-event silence durations as the ONLY features; if scores match without any edge content, the apparatus has been measuring the clock.* Advisor constraint honored: the silence feature's SOURCE is specified below, STEP=60 auto-clock sessions are excluded as silence sources, and the teeth of every candidate source were **measured, not assumed**.

Runner: `scripts/silence_test.py` (numpy + `scipy.special` Bessel ratio, same as the shipped vMF estimator; CPU; deterministic — zero-init logistic regression, fixed nearest-centroid). Raw numbers: `SILENCE-TEST-2026-08-20.json`. No git operations.

---

## 1. Silence source specification (the registration constraint)

| Source | Verdict | Evidence (measured) |
|---|---|---|
| `roomd-field-log.jsonl` (real wall-clock, 2.006s poll) | **teeth FAIL — excluded** | Arrivals derived from per-room message-count increments: `doctor-canary` = a +3-message generator tick every ~900.0s (8/12 gaps within poll-jitter of the median — near-constant silence is trivial by the advisor's own exclusion logic); `the-bridge` had **0 arrivals in 178.6 min**. |
| `production-log.jsonl` (real wall-clock) | **teeth FAIL — excluded** | Single room (`bar-rail`); gaps = the 30-min measurement poll cadence (median 1800.4s) — the apparatus's own sampling clock, not conversation pacing; single class ⇒ classification impossible. |
| S-series nights | **excluded by rule** | `clock_mode: auto60` (as are all nights — no session timestamp is used anywhere in this test). |
| **Text-length reading-time proxy (PRIMARY)** | **teeth CONFIRMED** | Inter-event silence before message *i* modeled as `τ_i = 0.20s + 0.005s·len_i` (a message takes as long to arrive as its text takes to read). Affine in `len` ⇒ "silence-only" ≡ "length-only": length is **not** one of the 7 dials — exactly the non-field structure the falsifier must rule in or out. Teeth: SEG1 lens 76.0±12.2 vs SEG2 61.5±17.1 chars (Welch d = 0.98); len sequences differ A vs H. |

The tap/radio-theater transcripts carry **no timestamps** (`room_from_markdown` assigns `ts = index`), so the "real-timed transcripts" branch of the registration could not run on them; the proxy branch is the registered fallback and is what ran.

**Corpus:** nights A (=B=C, byte-identical len/dial sequences by construction, used ONCE per `condition_eval.py` convention) + night-H (independent transition geometry, different len sequence). D/D′ excluded per suite convention. Strata SEG1 = seq 0–19, SEG2 = seq 20–39 (byte-identical convention for H). Windows W=4 (the suite's condition window), stride 1; W=8 sensitivity run. **Anti-rigging rule:** silence features are duration statistics ONLY — no cumulative/absolute time, because window position encodes "SEG2 comes later," which *is* the clock.

**Arms:** (a) CONTENT = the room field as shipped — window mean+sd of the 7 logged `field_eff_after` dials, and the stronger `field+fit` variant adding the vMF window fit (unit μ̂, log κ — the quantities behind the dial-tier fine gap); (b) SILENCE = [mean, sd, min, max] of τ per window, nothing else; (c) SILENCE+POSITION — rigged on purpose, diagnostic only. Classifiers: logistic regression (the pinned one) and nearest-centroid (variance-robust at this n). Eval: leave-one-night-out (LONO, primary), within-night message-disjoint half-split (positional-stationarity check).

---

## 2. L1 — condition classification (SEG1 warm-earnest vs SEG2 cynical-banter)

The fine condition edge — the dial-tier flagship (vMF μ̂ chord 1.229, reproduced here from `summary.json`: **1.2285**).

**Pooled LONO (n=68 windows):**

| Arm | Classifier | Acc | AUC | p vs chance |
|---|---|---|---|---|
| content (dials) | logreg | 0.603 | 0.641 | 1.1e-01 (ns) |
| content (dials) | centroid | 0.765 | 0.790 | **1.4e-05** |
| content (field+fit) | logreg *(pinned by the rule)* | 0.603 | 0.647 | 1.1e-01 (ns) |
| content (field+fit) | centroid | 0.603 | 0.794 | 1.1e-01 (ns) |
| **silence only** | logreg | **0.897** | **0.906** | **7.4e-12** |
| **silence only** | centroid | **0.882** | **0.908** | **5.8e-11** |
| silence+position (RIGGED) | logreg | 0.956 | 0.978 | diagnostic |

W=8 sensitivity: silence **0.923** (logreg) / **0.942** (centroid); content best 0.788 (centroid dials), 0.558 (logreg). Silence-only exceeds content-only under **every** classifier × window-size combination. Per-direction LONO: silence transfers both ways (train A→H 0.882 / AUC 0.851; train H→A 0.912 / AUC 0.983); content transfers asymmetrically (0.706 / AUC 0.706 one way; 0.500 acc but AUC 0.882 the other — signal present, threshold miscalibrated).

**Mechanism (Pearson r of message length vs dial, pooled A+H):** volume **−0.298**, cynicism **−0.298**, joke_landing −0.219, mood +0.061, panic −0.000, presence −0.078. The length confound is concentrated in precisely the banter dials that carry the SEG1→SEG2 contrast — cynical banter *is* shorter messages, and the cynicism dial partly reads length.

**L1 reading: the falsifier FIRED in the clock direction.** The registered CLOCK condition — silence-only ≥ content-only − 0.10 with content discriminating — is met under the centroid classifier (content 0.765, silence 0.882 ≥ 0.665). Under the rule's pinned classifier (pooled LONO logreg) content fails its own discrimination gate (0.603, p=0.114) while silence-only discriminates at p=7.4e-12 — a third outcome shape the pre-registered branches did not anticipate, reported under the declared honesty clause rather than absorbed: **direct clock-direction evidence, stronger than "match."** At the condition level, the apparatus does not demonstrate an object beyond the length/pacing clock.

## 3. L2 — night-identity classification (A vs H)

Silence has genuine teeth here (len sequences differ A vs H). With only two nights, leave-one-night-out training is single-class — degenerate by construction (the encoder-tier held-out failure's own shape) — so the eval is the message-disjoint half-split (train first-half windows of both nights, test second halves; n=28):

| Arm | Classifier | Acc | AUC | p |
|---|---|---|---|---|
| content (field+fit) | centroid | **0.893** | **0.944** | **2.7e-05** |
| content (field+fit) | logreg | 0.679 | 0.714 | 8.7e-02 |
| silence only | logreg | 0.393 | 0.408 | ns |
| silence only | centroid | 0.357 | 0.393 | ns |

**L2 reading: the falsifier did NOT fire — object direction.** The dial field separates the two nights' composition geometry (flip vs ramp) at 0.893 while silence-only collapses *below* chance. Where the frame's room-identity claims actually live, the object survives the clock test.

## 4. Verdict

**SPLIT VERDICT — the falsifier drew a line inside the frame's claims:**

- **Condition grain (SEG1→SEG2, the fine edge): CLOCK-CONFOUNDED.** Silence-only (length proxy) classifies the condition better than the room field does under every tested setting (0.88–0.94 vs 0.56–0.79). The flagship fine gap does not demonstrate an object beyond message length/pacing, and the mechanism is identified: the cynicism/volume dials correlate −0.30 with length. Per the registration's own sentence: *at this level, the apparatus may be measuring the clock.*
- **Identity grain (night/room identity): OBJECT.** Content discriminates (0.893, p=2.7e-05) while silence-only collapses below chance (0.36–0.39). The frame has an object here that the length clock cannot supply.

Under the pre-registered decision rule taken literally (pooled LONO logreg), the L1 gate outcome is the third shape — content failed its own gate while silence-only discriminates far above it — which is clock-direction evidence at least as strong as the registered CLOCK branch; the L2 result is the registered OBJECT branch. The dissertation's honest sentence: **the fine condition edge is length-confounded; room identity is not; and the silence test is now the instrument that keeps saying so.**

## 5. Limitations (stated, not buried)

1. **Two independent nights** (A=B=C identical by construction). Windows overlap within strata (stride 1), so binomial p-values overstate independence; the LONO split is the actual protection. Direction of results is stable across W, classifiers, and both LONO directions.
2. **One clock model.** The proxy is affine in message length — this tests whether condition signal is separable from length structure, the specific registered proxy. Other clock models (burstiness, turn-taking rhythm) untested; silence-as-data remains a booked, logged-but-unread edge-log feature.
3. **The real-timed branch could not run on the tap corpus** (no timestamps) and the two live logs failed the teeth audit (near-constant generator/poll gaps — numbers above). The condition-level result therefore rests on the nights corpus, whose content arm *is* the shipped logged field (`field_eff_after` + vMF fits) — the strongest available instance of the test, not the ideal one. Computing shipped dials for the tap evenings and rerunning is the cheapest upgrade.
4. **Night-H is pro-premise by design** (ramp pulls era means together), which handicaps L1 content on H; per-direction numbers are reported so this is visible, and content trained-on-A still reaches AUC 0.88 tested-on-H — signal present, but never separating from the length clock.
5. The within-night half-split (positional stationarity) mildly favors content (0.750 logreg, p=0.0125, vs silence 0.607–0.679 ns) — the length structure itself drifts within strata. Reported for completeness; it does not rescue the L1 condition claim.

## 6. What this buys the dissertation

- The hermes "no invariant" alternative is now **partially quantified**: at condition grain it is live (the clock explains at least as much), at identity grain it is beaten.
- The dial-tier fine gap (1.229) must henceforth be reported **alongside its length confound** (silence-only ≥ content-only), or defended against it (e.g., length-matched conditions in E3) — the honest upgrade is registered here for the first time.
- The cynicism/volume length correlation (−0.30) is a concrete dial-design finding: banter dials partially read message length, which is a standardization target (length-normalized dials), not a reason to hide the number.
