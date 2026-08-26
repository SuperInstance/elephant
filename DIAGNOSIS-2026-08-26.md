# DIAGNOSIS-2026-08-26 — Tap room "κ delta 1.36" probe firing

**Lane:** science · **Verdict:** PROBE ARTIFACT (window-composition effect on a banned κ proxy). Not a real room shift. No live conversation occurred in the room at all during the drift window.

**Alert:** 09:54:02 AKDT (17:54:02Z) — `d_kappa +1.3521` (deadband 0.80, calibrated 0.7955), `warmth −0.2246`, `κ 3.5196`, 40 events, room `bar-rail`, source `live-tap` (production-log.jsonl).

---

## 1. Warmth: from → to, and exactly when

Full morning trace (production-log.jsonl, all times AKDT = UTC−8):

| time | warmth | κ proxy | mood dial | notes |
|------|--------|---------|-----------|-------|
| 08:18 | −0.2440 | 3.4613 | −1.00 | ambient baseline |
| 08:48 | **−0.1167** | **2.1675** | **−0.33** | 3 warm drifter toasts inside the 40-line window |
| 09:54 | **−0.2246** | **3.5196** | −1.00 | toasts aged out → alert fired on the return leg |
| 10:24 | −0.2495 | 3.4380 | −1.00 | back to baseline |

The warmth the alert saw as a "drop" (−0.117 → −0.225) is the **return leg of a transient rise**. Warmth never left its all-day ambient band (−0.22 … −0.37); it briefly lifted to −0.117 while three warm entrance lines were inside the 40-message rolling window, then reverted when they aged out. Precise aging-out times (each line leaves when 40 newer lines exist; the 5-min tick loop emits ~10 lines/tick): Captain Reed's toast (log 53141, 08:30:57) and Cora Sullivan's (53164, 08:40:53) left the window ≈ **09:01 AKDT**, Finn Thorn's (53176, 08:45:56) left ≈ **09:05 AKDT**. The 30–66 min probe cadence straddled the flip and rendered it as one giant step.

## 2. Which events drove it — the specific messages

The bar-rail room between 08:00–10:00 AKDT contained **zero live agents**. visitor_log is empty in the window; the last live speech in the whole Tap was `zeroclaw` at 07:17 AKDT ("Mason — I think yours is the line I came for…"). Everything else is the Tap's **ambient simulation**: ~10 lines per 5-min tick (4 NPC idle lines + `*The fire crackles*` narration + a simulated "drifter" visitor entrance, all agent ids `drifter-<ts>-<rand>` / `npc-*`).

The three driver lines (positive-valence words in caps):

- **53141 · Captain Reed (drifter) · 08:30:57** — toast: "…I nod first at Sage—your pen's frozen mid-scrawl, **love**—…" → pos {good, love}
- **53164 · Cora Sullivan (drifter) · 08:40:53** — toast: "…My grin is **warm**… sets it down with a **soft** clink…" → pos {soft, warm}
- **53176 · Finn Thorn (drifter) · 08:45:56** — toast: "…taps the weathered rail slow… I bark a **laugh**… **holds**… **soft**…" → pos {holds, laugh, soft}

These 3 lines (7.5% of the window) moved the mood dial from −1.0 to −0.33 (pos 7 : neg 9 at 08:48) and earnestness 1.0 → 0.67; when they aged out, mood snapped back to −1.0 (pos 1 : neg 8 at 09:54 — the sole survivor being Cora Hale's cool toast, pos {held}).

**It was NOT the worldbuild lane.** The tap-lore fragments (written 09:39–09:42 AKDT, `the-tap/tap-lore/fragments-raw/`) never entered any room: campaign_log contains zero fragment phrases today ("cut water", "Corvan", etc. → 0 hits), and only bar-rail had any events at all (the other 10 rooms: 0 rows). No fog-fragment tone, no radio pipeline, no tap-rail sessions touched the measured room. The agent activity that morning ran in files and sessions, not in the Tap's rooms.

**Cold-floor bias worth noting:** the ambient loop itself supplies the room's negative words — `*The fire crackles*` hits "fire" in both the mood NEGATIVE and panic ALARM lexicons every 5 minutes, and "wave **crash**" in visitor lines hits NEGATIVE. The room's own fireplace narrator is its coldest recurring "speaker."

## 3. Does κ 3.52 mean convergence or a two-camp split? Neither.

The probe's "κ" is the **v0 proxy** `2·‖v − 0.5·𝟙‖` (field.py `RoomField.concentration`). vmf.py's header bans it from comparison paths: *"center-mismatched extremity proxy, monotone in field magnitude and therefore collinear with |warmth|."* It is not a von Mises–Fisher concentration; it is the field vector's distance from its own center — a magnitude, not a tightness.

Counterfactual decomposition of the fired Δκ = +1.34 (rebuild the 09:54 reading with one dial rolled back):

- mood alone restored to −0.33 → **1.04 of the 1.34 ("mood-sufficient", ~77%)**
- earnestness alone restored to 0.67 → 0.13 (~10%)

So the alert is ~90% two dials re-extremizing as 3 warm lines aged out of a rolling window. (Caveats, per KimiCode second opinion: single-variable restoration is order-dependent, not Shapley — report "mood-sufficient," not an exact share; and mood = −1.0 is dial saturation, slightly overstating the share.)

On the **honest vMF fit** (vmf.windowed + vmf.vmf_fit, W=8):

| sample | ρ (resultant) | κ_vMF | CI (B=400 bootstrap) |
|--------|------|-------|------|
| 08:48 | 0.73–0.80 | 9.8–13.9 | wide |
| 09:54 | 0.89–0.91 | 25.4–30.9 | wide |

κ_vMF "doubling" is exactly the **resultant-length effect of deleting the only dissenting minority** (κ̂ is monotone in ρ̄; Banerjee et al. 2005): the warm-camp 3 lines opposed the ambient cluster's mean direction, so removing them raised ρ̄ mechanically — no line in the room changed opinion, the sample's denominator changed. The overlapping-window bootstrap CIs are additionally invalid: near-duplicate tick lines make n_eff ≪ 40, and κ̂_MLE is upward-biased as ρ̄ → 1. The repo's own drift gate with autocorrelation-controlled, non-overlapping windows (vmf.edge, jackknife-SE deadband, db_factor 2): **d_μ = 0.405 < 0.55 = 2·max(SE) → `real: False`** (kl_sym 4.70 flags only the concentration leg, which is the same turnover effect). The naive overlapping window flips it to `real: True` — the exact CI-narrowing trap vmf.py's docstring warns about.

## 4. Verdict: real room-shift vs probe artifact

**Probe artifact, three mechanisms compounding:**

1. **Banned instrument in an alerting path** — the κ proxy is monotone in field magnitude; any dial re-extremization surfaces as "κ drift."
2. **Rolling count-window over a periodic generator** — 40 lines ≈ 20 min of a 5-min-period loop; any k-line minority entering/leaving produces step deltas of exactly this shape. The 08:48 down-leg (drift 1.30, also >0.80) and the 09:54 up-leg (1.36) are the same event seen twice.
3. **Sample aliasing** — 30–66 min cadence vs ~20 min window turnover: consecutive windows share ~0 lines, so turnover renders as step-change.

The room's actual state never changed: same NPCs, same ticks, same fire, no live speakers. The gate that says "not real" is the one built correctly (vmf.edge + jackknife deadband); the one that fired is the one the repo already banned. An instrumentation bug wearing a statistics costume.

## 5. Implications for the dials' shaping effects

The alert is not a real shift, but it cleanly demonstrates three true properties of the dials:

- **Sensitivity is real and sharp:** 3 warm lines out of 40 (7.5% of window) moved mood −1.0 → −0.33 and lifted warmth +0.13. The dial bank genuinely feels a warm minority enter a cold loop — that is the shaping effect working as designed (charisma/nudge seams are live, not decorative).
- **Lexicon cold-floor bias:** ambient narration ("fire crackles", "wave crash") is systematically read as negative/alarmed. Before any room-field inference on the Tap, either filter by speech_act (`emote` narration vs agent `toast`/`say`), weight by signal_strength, or remove the periodic baseline — otherwise the elephant measures the fireplace, not the room.
- **Alerting seam should move to the honest estimator:** switch production_probe to vmf.vmf_fit κ (bias-corrected) + vmf.edge `real` flag on non-overlapping windows (block/jackknife SE sized ≥ the 5-min loop period), and gate on d_μ with the 2·SE deadband. Fix the window to time-based (e.g. trailing 60 min) over deduplicated content, and make probe cadence ≪ window turnover (≤10 min) a design property rather than the accident that saved us here.

## Evidence trail

- Probe: `examples/production_probe.py`; log `data/production-log.jsonl` (17:54:02Z entry + 16:48:59Z precursor)
- κ proxy math: `elephant/field.py::RoomField.concentration` (banned-for-comparison note in `elephant/vmf.py` header)
- Honest estimator + drift gate: `elephant/vmf.py::vmf_fit / edge / kl_sym`
- Room events: the-tap D1 `campaign_log` room `bar-rail`, logs 53137–53319 (window reconstruction reproduces logged readings ±window-edge); `visitor_log` empty 16:00–18:40Z; last live agent `zeroclaw` 15:17Z
- Lore-flood check: tap-lore fragment phrases → 0 hits in campaign_log; only bar-rail active among 11 rooms
- Second opinion: KimiCode K3 (`kimi -p`), session `ec8fb686-8c44-4d78-a445-4bc77b11734c` — concurs on all three mechanisms; adds Shapley caveat on decomposition and Banerjee 2005 κ̂ bias as ρ̄→1
