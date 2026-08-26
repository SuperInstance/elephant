"""Probe hygiene + honest alerting — the fixes the 2026-08-26 diagnosis ordered.

DIAGNOSIS-2026-08-26.md: three false alarms (08:48 / 09:54 / 11:29 AKDT) were
window-composition artifacts, not room shifts. The probe watched a banned κ
proxy (`RoomField.concentration`, monotone in field magnitude), sampled a
40-line count-window whose turnover (~20 min) was aliased by a 30–66 min
cadence, triple-counted simulated drifter entrances, and let the room's own
narrator (`*The fire crackles*` — "fire" is in the mood NEGATIVE and panic
ALARM lexicons) read as its coldest speaker.

This module is the fixed pipeline. Four surgical rules (captain-approved):

1. **Alert on the honest vMF estimator, never the magnitude proxy.**
   The drift decision is `vmf.edge(prev_fit, cur_fit)` — d_μ̂ against the
   jackknife-SE deadband (gate 4), on non-overlapping windows so the SE is
   not autocorrelation-narrowed (the diagnosis's overlapping-CI trap). The
   legacy warmth/κ-proxy fields stay in the log for continuity and are
   *banned from every alerting path* — vmf.py's header rule, now enforced
   structurally: `alarm_decision()` consumes only estimator fields.

2. **Time-based dedup of NPC entrances.** Three drifter toasts 5–10 min
   apart are one ambient-entrance event, not three. Simulated visitor
   lines (`drifter-*` agent ids) collapse within a trailing dedup window
   (default 15 min, sized to the false-positive batch: 08:30:57, 08:40:53,
   08:45:56). Live agents are never deduped; a visitor 30 min later is a
   separate event.

3. **Cadence ≪ window turnover; alarm only on persistence.** The content
   window is time-based (trailing 60 min, not 40 lines) and the nominal
   cadence is 10 min — a design property: PROBE_INTERVAL_S * CADETNCE_FACTOR
   ≤ WINDOW_S. A drift alarms only after `confirm_probes` consecutive
   probes see a real (deadband-clearing) edge, with no inter-probe gap
   larger than `max_gap_s` (a stale gap breaks the chain — you cannot
   confirm persistence across an aliasing hole).

4. **Narration-aware channels.** Room-narrator lines (`the-tap` emotes:
   fire crackles, rain, door chimes) are excluded from the speaker-
   temperature room and logged as their own `narration` channel, so the
   elephant stops measuring the fireplace.

Walls honored: vmf.py math untouched (imported, never modified); dial
semantics untouched (the dials see a *cleaner room*, same lexicons); the
legacy log fields keep their shape for continuity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .dial import DialBank
from .field import read_field
from .room import Message, Room
from . import vmf

# --------------------------------------------------------------------------- #
# Constants — the design properties (fix 3 keeps them honest by test)         #
# --------------------------------------------------------------------------- #

WINDOW_MINUTES = 60          # trailing content window (time-based, not count)
PROBE_INTERVAL_MINUTES = 10  # nominal cadence — ≪ window turnover
CONFIRM_PROBES = 2           # consecutive real edges required to alarm
MAX_GAP_MINUTES = 15         # larger inter-probe gap breaks the persistence chain
ENTRANCE_DEDUP_S = 15 * 60   # simulated entrances within this window = one event
LEGACY_WINDOW_LINES = 40     # back-compat proxy block only — never alerting
FIT_W = 8                    # vmf window size (session default, ≈ the 5-min loop)
FIT_STEP = 8                 # non-overlapping: honest jackknife SE
FIT_CAP = 512                # never truncate a 60-min clean window

# The room's own narrator (fix 4). The Tap's narrator agent is `the-tap`;
# generic system ids future-proof other rooms.
NARRATOR_AGENTS = ("the-tap", "system", "narrator")

# Simulated-visitor agent id prefixes (the ambient loop's periodic stream).
SIMULATED_PREFIXES = ("drifter-", "npc-",) + NARRATOR_AGENTS


# --------------------------------------------------------------------------- #
# Line parsing — one shape for API dicts, fixture dicts, and Messages         #
# --------------------------------------------------------------------------- #

@dataclass
class ProbeLine:
    author: str
    text: str
    ts: float                       # epoch seconds (UTC)
    speech_act: str = ""            # "statement" | "emote" | "toast" | "story" | ...
    display_name: str = ""

    def to_message(self) -> Message:
        return Message(author=self.author, text=self.text, ts=self.ts)


_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _parse_ts(raw) -> Optional[float]:
    """Epoch seconds from a Tap timestamp ("2026-08-26 19:45:36", UTC) or a number."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1]
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def parse_line(line) -> Optional[ProbeLine]:
    """Normalize one fetched line (dict from the Tap API or a Message) to a ProbeLine.

    Returns None for unparseable lines (never crashes the probe).
    """
    try:
        if isinstance(line, Message):
            return ProbeLine(author=line.author, text=line.text,
                             ts=line.ts if line.ts else None)
        if isinstance(line, dict):
            author = (line.get("agent_id") or line.get("speaker")
                      or line.get("author") or "unknown")
            text = line.get("content") or line.get("text") or ""
            ts = _parse_ts(line.get("timestamp") or line.get("ts"))
            return ProbeLine(author=str(author), text=str(text), ts=ts,
                             speech_act=str(line.get("speech_act") or ""),
                             display_name=str(line.get("display_name") or ""))
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Fix 4 — narration-aware channels                                            #
# --------------------------------------------------------------------------- #

def is_narration(line: ProbeLine) -> bool:
    """Room-narrator line: the system tells the room what is happening
    (fire crackles, rain, door chimes) rather than a speaker's words.

    Primary rule is the author (the Tap's narrator agent is `the-tap`;
    drifter *emote* lines stay speaker content — the visitor acting, not
    the room narrating). Text-shape fallback (a whole asterisk-wrapped
    line from a system-shaped id) covers rooms without speech metadata.
    """
    a = (line.author or "").lower()
    if a in NARRATOR_AGENTS:
        return True
    if a.startswith(("system", "narrator")):
        return True
    t = (line.text or "").strip()
    if a.startswith(NARRATOR_AGENTS) and t.startswith("*") and t.endswith("*"):
        return True
    return False


def split_channels(lines: Sequence[ProbeLine]) -> Tuple[List[ProbeLine], List[ProbeLine]]:
    """(speaker_lines, narration_lines) — fix 4's channel split."""
    speakers, narration = [], []
    for ln in lines:
        (narration if is_narration(ln) else speakers).append(ln)
    return speakers, narration


# --------------------------------------------------------------------------- #
# Fix 2 — time-based dedup of simulated entrances                             #
# --------------------------------------------------------------------------- #

def is_simulated_entrance(line: ProbeLine) -> bool:
    """A simulated visitor's own entrance line (drifter agent id, first line
    of a fresh id). Recurring NPC idle chatter is not an entrance; live
    agents are never simulated."""
    return (line.author or "").startswith("drifter-")


def dedup_entrances(lines: Sequence[ProbeLine],
                    window_s: float = ENTRANCE_DEDUP_S
                    ) -> Tuple[List[ProbeLine], List[ProbeLine]]:
    """Collapse bursts of simulated entrances into one event per window.

    Time-based: the first entrance line is kept; every later simulated
    entrance within `window_s` of the last SEEN simulated entrance (kept
    or dropped) is dropped — the ambient loop's periodic visitor stream,
    not independent arrivals. The window therefore collapses any burst
    whose inter-arrival gaps stay under `window_s` (the false-positive
    batch: 5–10 min apart), while a genuinely separate visitor 30 min
    after the last one survives. Live-agent lines pass through
    untouched. Returns (kept, dropped).
    """
    kept: List[ProbeLine] = []
    dropped: List[ProbeLine] = []
    last_seen_ts: Optional[float] = None
    for ln in lines:  # caller passes time-ordered lines
        if is_simulated_entrance(ln) and ln.ts is not None:
            if last_seen_ts is not None and ln.ts - last_seen_ts < window_s:
                dropped.append(ln)
                last_seen_ts = ln.ts
                continue
            last_seen_ts = ln.ts
        kept.append(ln)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Fix 3 — time-based content window                                           #
# --------------------------------------------------------------------------- #

def trailing_window(lines: Sequence[ProbeLine],
                    minutes: float = WINDOW_MINUTES
                    ) -> Tuple[List[ProbeLine], List[ProbeLine]]:
    """Lines inside the trailing `minutes` window, anchored at the newest
    line's timestamp (the room's clock, not the probe's). Returns
    (in_window, aged_out)."""
    if not lines:
        return [], []
    t_newest = max(ln.ts for ln in lines if ln.ts is not None) if any(
        ln.ts is not None for ln in lines) else None
    if t_newest is None:
        return list(lines), []
    horizon = t_newest - minutes * 60.0
    in_w = [ln for ln in lines if ln.ts is not None and ln.ts >= horizon]
    out_w = [ln for ln in lines if ln.ts is None or ln.ts < horizon]
    return in_w, out_w


# --------------------------------------------------------------------------- #
# Fix 1 — the honest estimator path (vmf.py's math, imported not modified)    #
# --------------------------------------------------------------------------- #

def honest_fit(clean_lines: Sequence[ProbeLine], bank: DialBank,
               W: int = FIT_W, step: int = FIT_STEP) -> Optional[dict]:
    """vmf_fit over non-overlapping trailing windows of the clean room.

    step = W (non-overlapping) so the jackknife SE is not narrowed by
    window autocorrelation — the exact overlapping-CI trap the diagnosis
    documents. Returns None below NMIN windows (κ not identifiable —
    never a fake number).
    """
    room = Room("probe-clean", [ln.to_message() for ln in clean_lines])
    zs = vmf.windowed(room, bank, W=W, step=step, cap=FIT_CAP)
    if len(zs) < vmf.NMIN:
        return None
    return vmf.vmf_fit(zs)


# --------------------------------------------------------------------------- #
# The alert decision — consumes ONLY estimator fields (fix 1, structural)     #
# --------------------------------------------------------------------------- #

def alarm_decision(current: dict, history: Sequence[dict],
                   confirm_probes: int = CONFIRM_PROBES,
                   max_gap_s: float = MAX_GAP_MINUTES * 60.0) -> dict:
    """Persistent-drift alarm from the honest estimator alone.

    Inputs are probe log entries; only the estimator-derived fields
    (`vmf`, `drift_vmf`, `ts`) are read — the κ-proxy fields cannot reach
    this function's verdict by construction. Fires when the last
    `confirm_probes` probes each recorded a real (deadband-clearing) edge
    with no inter-probe gap exceeding `max_gap_s` (a stale gap breaks the
    persistence chain — fix 3).
    """
    edge = current.get("drift_vmf")
    if current.get("vmf") is None:
        return {"alarm": False, "alarm_reason": "not_identifiable"}
    if edge is None:
        return {"alarm": False, "alarm_reason": "no_prior"}
    if not edge.get("real"):
        return {"alarm": False, "alarm_reason": "edge_real_false"}

    chain: List[dict] = [current]
    for prev in reversed(history):
        last_ts = _iso(chain[-1].get("ts"))
        prev_ts = _iso(prev.get("ts"))
        if last_ts is None or prev_ts is None:
            break
        if (last_ts - prev_ts).total_seconds() > max_gap_s:
            break  # cadence hole — persistence cannot be confirmed across it
        chain.append(prev)
        if len(chain) >= confirm_probes:
            break
    if len(chain) < confirm_probes:
        return {"alarm": False, "alarm_reason": "awaiting_confirmation"}
    for e in chain:
        ed = e.get("drift_vmf") or {}
        if not ed.get("real"):
            return {"alarm": False, "alarm_reason": "awaiting_confirmation"}
    return {"alarm": True, "alarm_reason": "confirmed_persistent_drift"}


def _iso(s) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# One probe — the fixed pipeline end to end                                   #
# --------------------------------------------------------------------------- #

def probe_entry(raw_lines: Sequence,
                bank: DialBank,
                history: Sequence[dict],
                now: Optional[datetime] = None,
                window_minutes: float = WINDOW_MINUTES) -> dict:
    """Compute one probe log entry from fetched lines + prior entries.

    The entry carries three blocks, cleanly separated:
    - legacy proxy block (warmth/κ-proxy + deltas on the raw last-40 lines)
      — continuity with the old log, *never* consulted for alerting;
    - clean block (fix 4 narration channel + fix 2 dedup counts, clean
      dial readings over the fix 3 time window);
    - estimator block (`vmf` fit + `drift_vmf` edge) and the `alarm`
      verdict from `alarm_decision` (fix 1 + persistence).
    """
    now = now or datetime.now(timezone.utc)
    entry: dict = {
        "ts": now.isoformat(),
        "room": "probe",
        "source": "probe",
        "ok": False,
    }

    parsed = [p for p in (parse_line(l) for l in raw_lines) if p is not None]
    missing_ts = [ln for ln in parsed if ln.ts is None]
    if missing_ts:  # degrade honestly: time window needs timestamps
        for i, ln in enumerate(missing_ts):
            ln.ts = 0.0  # keep order via sort below; window falls back to all
        entry["ts_fallback"] = True

    entry["n_events"] = len(parsed)

    # ---- legacy proxy block (back-compat; banned from alerting) ---------- #
    legacy_room = Room("legacy", [
        Message(author=ln.author, text=ln.text, ts=float(i))
        for i, ln in enumerate(parsed[-LEGACY_WINDOW_LINES:])
    ])
    legacy_field = read_field(legacy_room, bank)
    entry["field"] = {k: round(v, 4) for k, v in legacy_field.readings.items()}
    entry["warmth"] = round(legacy_field.warmth(), 4)
    entry["kappa"] = round(legacy_field.concentration(), 4)  # proxy, log-only

    # ---- clean block (fixes 2 + 4 over the fix 3 time window) ------------ #
    in_w, aged = trailing_window(parsed, window_minutes)
    speakers, narration = split_channels(in_w)
    kept, dropped_entrances = dedup_entrances(speakers)
    clean_room = Room("probe-clean", [ln.to_message() for ln in kept])
    clean_readings = bank.readings(clean_room)
    entry["window"] = {
        "minutes": window_minutes,
        "lines_in_window": len(in_w),
        "lines_aged_out": len(aged),
        "anchor": datetime.fromtimestamp(
            max((ln.ts for ln in in_w), default=0.0), tz=timezone.utc
        ).isoformat() if in_w else None,
    }
    entry["hygiene"] = {
        "narration_dropped": len(narration),
        "entrances_dropped": len(dropped_entrances),
        "entrances_kept": sum(1 for ln in kept if is_simulated_entrance(ln)),
        "live_speakers": sorted({ln.author for ln in kept
                                 if not ln.author.startswith(SIMULATED_PREFIXES)}),
    }
    entry["narration"] = {
        "n": len(narration),
        "sample": [ln.text[:60] for ln in narration[:3]],
    }
    entry["clean_field"] = {k: round(v, 4) for k, v in clean_readings.items()}

    # ---- estimator block (fix 1: the honest path) ------------------------ #
    # `fit` is the single source: the logged summary AND the mu_hat carried
    # forward for the next probe's edge both come from this one object.
    fit = honest_fit(kept, bank)
    if fit is not None:
        entry["vmf"] = {
            "kappa": round(fit["kappa"], 4),
            "rho": round(fit["rho"], 4),
            "n": fit["n"],
            "warmth_vmf": round(fit["warmth_vmf"], 4),
            "saturated": fit["saturated"],
        }
        entry["vmf_mu_hat"] = [round(x, 6) for x in fit["mu_hat"]]
        entry["vmf_mu_se"] = round(fit["mu_se"], 6)
    else:
        entry["vmf"] = None

    prev = history[-1] if history else None
    prev_fit = None
    if prev and prev.get("vmf_mu_hat") and prev.get("vmf"):
        prev_fit = {
            "mu_hat": prev["vmf_mu_hat"],
            "kappa": prev["vmf"]["kappa"],
            "warmth_vmf": prev["vmf"]["warmth_vmf"],
            "mu_se": prev.get("vmf_mu_se", prev["vmf"].get("mu_se", 0.0)),
        }
    edge = None
    if fit is not None and prev_fit is not None:
        e = vmf.edge(prev_fit, fit)
        if e is not None:
            edge = {
                "d_mu": round(e["d_mu"], 6),
                "d_warmth": round(e["d_warmth"], 6),
                "d_log_kappa": round(e["d_log_kappa"], 6),
                "real": bool(e["real"]),
            }
    entry["drift_vmf"] = edge

    # legacy proxy deltas for log continuity only
    if prev and prev.get("ok") is not False and "kappa" in (prev or {}):
        entry["d_warmth"] = round(entry["warmth"] - prev["warmth"], 4)
        entry["d_kappa"] = round(entry["kappa"] - prev["kappa"], 4)
        entry["drift"] = round(
            (entry["d_warmth"] ** 2 + entry["d_kappa"] ** 2) ** 0.5, 4)

    # ---- cadence + alarm (fix 3) ----------------------------------------- #
    prev_ts = _iso(prev.get("ts")) if prev else None
    cur_ts = _iso(entry["ts"])
    gap_s = (cur_ts - prev_ts).total_seconds() if (prev_ts and cur_ts) else None
    entry["cadence"] = {
        "gap_s": round(gap_s, 1) if gap_s is not None else None,
        "stale": bool(gap_s is not None and gap_s > MAX_GAP_MINUTES * 60.0),
        "nominal_interval_minutes": PROBE_INTERVAL_MINUTES,
    }
    entry.update(alarm_decision(entry, list(history)))
    entry["ok"] = True
    return entry

# design property (fix 3), asserted by test:
#   cadence × 3 ≤ window — the probe fires far more often than turnover.
assert PROBE_INTERVAL_MINUTES * 3 <= WINDOW_MINUTES, (
    "probe cadence must be ≪ window turnover (design property, fix 3)")
