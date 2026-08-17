#!/usr/bin/env python3
"""MUD live integration — the elephant's light in a real room.

The captain's promise: *"people will be drawn into our MUD systems with
tools like Elephant."* This is the live seam that makes it real. It
drives a real MUD room's description with the Room-Elephant, so the room's
own text changes with its vibe:

    * warm laughter  -> joyful adjectives, laughter reverberating into the words
    * a fight        -> storms outside, newcomers described as DRENCHED
    * closing time   -> disco off, fluorescents on, people drift toward the exit

The pipeline is the elephant's own (`elephant/space.py` + `elephant/mud.py`):

        room events ──▶ MudSpace ──▶ RoomField ──▶ tint_description ──▶ the
                       (ingest)      (read)          (the light)         room's
                                                                         description

WHAT THIS BRIDGE TALKS TO
-------------------------
The live target is **The Tap** — the fleet's text-rendered tavern MUD on
Cloudflare (https://the-tap.casey-digennaro.workers.dev). Its worker relay
exposes a plain JSON API (no keys needed for reads):

    GET  /api/rooms                          list rooms + base descriptions
    GET  /api/room/{room_id}/state           room state (description + mood)
    GET  /api/conversation/{room_id}?limit=N recent conversation lines
    POST /api/speak {room_id, speaker, text} write a line into the live room

The room `description` itself lives in the Tap's D1 `rooms` table and is
*read-only* over the public API today. So the elephant writes the light back
through the one seam that is both live and honest: it speaks the tinted
description INTO the room as the room's own narrator line (`POST /api/speak`).
In `--dry-run` (the default) it prints the would-be description plus the exact
seam it would use, and never touches the live room.

WHEN THE LIVE ROOM IS UNREACHABLE
---------------------------------
If there is no network or the relay is down, the bridge falls back to the
repo's own Tap transcripts (`ai-writings/tap-trades/2026-08-16/*.md`) as the
room feed — clearly labelled `[FALLBACK: transcripts]`.

Run it:

    python3 examples/mud_live_integration.py                 # live + 3 states, dry-run
    python3 examples/mud_live_integration.py --write         # actually speak into the room
    python3 examples/mud_live_integration.py --room bridge-table
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib import error as urlerr
from urllib import request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.field import RoomField
from elephant.mud import classify, tint_description
from elephant.room import Message
from elephant.space import AdapterRegistry, MudSpace

TAP_BASE_URL = os.environ.get(
    "TAP_BASE_URL", "https://the-tap.casey-digennaro.workers.dev")
DEFAULT_ROOM = "bar-rail"
DEFAULT_LIMIT = 40
DEFAULT_TRANSCRIPTS = "/home/eileen/projects/ai-writings/tap-trades/2026-08-16/*.md"

# The room's plain "before" text (the base description the field mutates).
BASE_DESCRIPTION = (
    "The counter is polished dark wood, well-worn where elbows have rested. "
    "Behind it, rows of bottles catch the light. The air smells of old wood "
    "and conversation."
)

# --------------------------------------------------------------------------- #
# HTTP (stdlib only — no external deps beyond numpy)                          #
# --------------------------------------------------------------------------- #
def _http_json(url: str, timeout: float = 12.0):
    req = request.Request(url, headers={
        "User-Agent": "elephant-mud-bridge/0.1",
        "Accept": "application/json",
    })
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict, timeout: float = 12.0) -> Tuple[dict, int]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "elephant-mud-bridge/0.1",
    })
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")), resp.status


def _hour_from_timestamp(ts: Optional[str]) -> Optional[float]:
    """Hour-of-day from a Tap `timestamp` string ('YYYY-MM-DD HH:MM:SS')."""
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).hour + 0.0
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Feeds — a room feed is: label, base_description, events, hour, writeable    #
# --------------------------------------------------------------------------- #
class TapRoomFeed:
    """The live Tap — real room events pulled over the worker relay."""

    label = "live The Tap"

    def __init__(self, room_id: str = DEFAULT_ROOM, limit: int = DEFAULT_LIMIT,
                 base_url: str = TAP_BASE_URL):
        self.room_id = room_id
        self.limit = limit
        self.base_url = base_url
        self.base_description: str = BASE_DESCRIPTION
        self.events: List[Tuple[str, str]] = []
        self.hour: Optional[float] = None
        self.mood_label: Optional[str] = None
        self._loaded = False

    def load(self) -> "TapRoomFeed":
        state = _http_json(f"{self.base_url}/api/room/{self.room_id}/state")
        conv = _http_json(
            f"{self.base_url}/api/conversation/{self.room_id}?limit={self.limit}")
        self.base_description = state.get("description") or BASE_DESCRIPTION
        mood = state.get("mood") or {}
        self.mood_label = mood.get("label")
        lines = conv.get("lines") or []
        # Sort chronologically by timestamp (the API reverse-orders by id).
        lines.sort(key=lambda ln: ln.get("timestamp") or "")
        self.events = [
            (ln.get("display_name") or ln.get("agent_id") or "[room]",
             ln.get("content") or "")
            for ln in lines
            if (ln.get("content") or "").strip()
        ]
        if lines:
            self.hour = _hour_from_timestamp(lines[-1].get("timestamp"))
        self._loaded = True
        return self

    def writeable(self) -> bool:
        return True

    def write_back(self, tinted_text: str) -> dict:
        """Speak the tinted description into the live room as its narrator."""
        body, status = _http_post_json(f"{self.base_url}/api/speak", {
            "room_id": self.room_id,
            "speaker": "the-room",
            "text": tinted_text,
        })
        return {"status": status, "response": body}


class TranscriptRoomFeed:
    """Fallback: the repo's own Tap transcripts as the room feed.

    Parses dialogue (`**NAME:** text`), stage directions (`*(...)*`), and
    prose paragraphs into room events, so the elephant still has a room to
    read when the live relay is unreachable.
    """

    label = "FALLBACK: repo transcripts"

    _DIALOGUE = re.compile(r"^\*\*([^*]+?):\*\*\s*(.*)$")
    _STAGE = re.compile(r"^\*\((.*)\)\*\s*$")

    def __init__(self, glob_pattern: str = DEFAULT_TRANSCRIPTS,
                 room_name: str = "The Tap (transcripts)"):
        self.glob_pattern = glob_pattern
        self.room_name = room_name
        self.base_description: str = BASE_DESCRIPTION
        self.events: List[Tuple[str, str]] = []
        self.hour: Optional[float] = 22.0
        self._loaded = False

    def load(self) -> "TranscriptRoomFeed":
        files = sorted(Path(self.glob_pattern).parent.glob(
            Path(self.glob_pattern).name))
        for path in files:
            self._parse_file(path)
        self._loaded = True
        return self

    def _parse_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = self._DIALOGUE.match(line)
            if m:
                author = _clean_author(m.group(1))
                self.events.append((author, m.group(2).strip()))
                continue
            s = self._STAGE.match(line)
            if s:
                self.events.append(("[room]", s.group(1).strip()))
                continue
            # Prose paragraph (not a header, rule, or meta line).
            if line.startswith(("#", "---", "*", ">")):
                continue
            if len(line) > 40:
                self.events.append(("[room]", line))

    def writeable(self) -> bool:
        return False

    def write_back(self, tinted_text: str) -> dict:
        return {"status": 0, "response": {"error": "transcript feed is read-only"}}


def _clean_author(raw: str) -> str:
    """'LUCINEER (foreman), setting down...' -> 'LUCINEER'."""
    raw = raw.strip()
    raw = re.sub(r"\([^)]*\)", "", raw)
    raw = raw.split(",")[0].strip()
    return raw or "[room]"


# --------------------------------------------------------------------------- #
# The bridge                                                                #
# --------------------------------------------------------------------------- #
def build_mud_space(name: str, base_description: str,
                    events: Iterable[Tuple[str, str]]) -> MudSpace:
    """Feed room events into a MudSpace; the room description is the tint target."""
    space = MudSpace(name, description=base_description)
    for author, text in events:
        space.chatter(author, text)
    return space


def read_and_tint(space: MudSpace, hour: Optional[float] = None) -> dict:
    """The elephant reads the room and speaks its tinted description.

    NOTE — the hour must be threaded by the bridge, not the seam:
    `MudSpace.tint` calls `tint_description(field, base_text)` with the
    default `hour=None`, so the *only* hour-dependent mode — closing time
    (late + quiet + low warmth) — is invisible to the seam alone. The live
    bridge is the thing that knows what time it is, so it computes the
    tinted text itself and pushes it through `send_back(field, tinted_text)`.
    """
    field = space.read()
    tinted = tint_description(field, space.base_description, hour=hour)
    space.send_back(field, tinted_text=tinted)  # sets space.description
    return {
        "field": field,
        "tinted": tinted,
        "mode": classify(field, hour),
        "warmth": field.warmth(),
        "kappa": field.concentration(),
        "readings": field.readings,
    }


def bridge_room(feed, write: bool = False, hour: Optional[float] = None) -> dict:
    """Run the live bridge over one feed; optionally write the light back.

    Returns a dict with everything the demo/tests need. Dry-run is the
    default; `write=True` actually speaks into the live room.
    """
    space = build_mud_space("The Tap", feed.base_description, feed.events)
    h = hour if hour is not None else feed.hour
    result = read_and_tint(space, hour=h)
    result["feed"] = feed.label
    result["base_description"] = feed.base_description
    result["n_events"] = len(feed.events)
    result["mood_label"] = getattr(feed, "mood_label", None)
    # Clock provenance: the bridge, not the seam, knows what time it is.
    if hour is not None:
        result["hour_source"] = "--hour override"
    elif feed.hour is not None:
        result["hour_source"] = "relay's latest line timestamp"
    else:
        result["hour_source"] = "local clock"

    if feed.writeable() and write:
        result["wrote_back"] = feed.write_back(result["tinted"])
    else:
        # The seam — what the bridge would push (and where), if not dry-run.
        result["wrote_back"] = None
        result["write_seam"] = {
            "endpoint": f"{TAP_BASE_URL}/api/speak",
            "method": "POST",
            "payload": {
                "room_id": getattr(feed, "room_id", DEFAULT_ROOM),
                "speaker": "the-room",
                "text": result["tinted"],
            },
            "note": ("The room `description` is read-only over the public API "
                     "(it lives in the Tap's D1 rooms table); the live write "
                     "seam is the room's narrator speaking into the transcript. "
                     "Pass --write to use it."),
        }
    return result


# --------------------------------------------------------------------------- #
# The three states — the SAME room under three lights                         #
# --------------------------------------------------------------------------- #
def synthetic_scenes() -> List[Tuple[str, List[Tuple[str, str]], float]]:
    """Three Tap-flavoured room feeds that drive the room to three modes.

    Returned as (label, events, hour). These are the *same* room; only the
    events change. They mirror real Tap texture so the dials move honestly.
    """
    warm_laughter = [
        ("lucineer", "I love this place — it's warm and the company's good."),
        ("welder", "haha, to the room then, it heard us before we walked in."),
        ("carpenter", "I'll drink to that, the room just holds. cheers everyone"),
        ("mason", "I talked to it like a horse, it listened. lol"),
        ("shipwright", "the floor holds, the floor remembers. haha 😂"),
        ("composite", "haha, and the dust came off in years. twelve-to-one!"),
        ("welder", "you and me, same trade with different torches. haha"),
        ("[room]", "A ripple of laughter crosses the room, warm and golden."),
    ]
    fight = [
        ("guard", "everyone — NOW. get back from the door!"),
        ("marlo", "what's happening?! there's a breach —"),
        ("pincher", "FIRE in the galley! all hands! run!"),
        ("skip", "help! collision on the aft side! alarm!"),
        ("barnacle", "evacuate! this is not a drill!"),
        ("sally", "the storm is right on top of us, we're sinking!"),
        ("guard", "STAMPEDE at the bar — everyone move, now!"),
        ("[room]", "Chairs scrape. Someone shouts. The door bangs open onto the storm."),
    ]
    closing_time = [
        ("bartender", "last call was ten minutes ago."),
        ("sally", "the street outside is empty."),
        ("bartender", "closing your tab?"),
    ]
    return [
        ("warm laughter", warm_laughter, 21.0),
        ("a fight breaking out", fight, 23.0),
        ("closing time", closing_time, 2.0),
    ]


def demo_three_states(base_description: str = BASE_DESCRIPTION) -> List[dict]:
    """The light itself: one room, three fields, three descriptions."""
    out = []
    for label, events, hour in synthetic_scenes():
        space = build_mud_space("The Tap", base_description, events)
        r = read_and_tint(space, hour=hour)
        r["label"] = label
        r["hour"] = hour
        r["base_description"] = base_description
        r["n_events"] = len(events)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _print_live(feed, result: dict, write: bool) -> None:
    print("=" * 78)
    print(f"LIVE BRIDGE — {result['feed']}")
    print("=" * 78)
    print(f"room:        {getattr(feed, 'room_id', DEFAULT_ROOM)}")
    print(f"events read: {result['n_events']}")
    if result.get("mood_label"):
        print(f"Tap's own mood label: {result['mood_label']}")
    print(f"hour:        {getattr(feed, 'hour', None)}  (source: {result.get('hour_source', 'n/a')})")
    print(f"mode:        {result['mode']}  (warmth {result['warmth']:+.2f}, "
          f"κ {result['kappa']:.2f})")
    print(f"\nBEFORE (base description):\n    {result['base_description']}")
    print(f"\nAFTER (the room speaks):\n    {result['tinted']}")
    if result["wrote_back"]:
        print(f"\nWROTE BACK -> {result['wrote_back']}")
    else:
        seam = result["write_seam"]
        print("\nWRITE SEAM (dry-run — nothing was pushed):")
        print(f"  {seam['method']} {seam['endpoint']}")
        print(f"  payload = {json.dumps(seam['payload'], indent=2)}")
        print(f"  note: {seam['note']}")


def _print_three_states(results: List[dict]) -> None:
    print("\n" + "=" * 78)
    print("THE LIGHT ITSELF — the SAME room under three lights")
    print("=" * 78)
    for r in results:
        print(f"\n[{r['label']}]  (hour={r['hour']:g}, mode={r['mode']}, "
              f"warmth={r['warmth']:+.2f}, κ={r['kappa']:.2f})")
        print(f"  {r['n_events']} events")
        print(f"  BEFORE: {r['base_description']}")
        print(f"  AFTER:  {r['tinted']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room", default=DEFAULT_ROOM, help="Tap room id")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="conversation lines to read")
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS,
                    help="fallback transcript glob")
    ap.add_argument("--hour", type=float, default=None,
                    help="override hour-of-day (for closing-time tint)")
    ap.add_argument("--write", action="store_true",
                    help="actually speak the tint into the live room")
    ap.add_argument("--no-live", action="store_true",
                    help="skip the live room and only run the three-state demo")
    args = ap.parse_args(argv)

    if not args.no_live:
        feed: TapRoomFeed
        try:
            feed = TapRoomFeed(args.room, args.limit).load()
        except (urlerr.URLError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[live The Tap unreachable: {type(exc).__name__}: {exc}]")
            print(f"[falling back to transcripts: {args.transcripts}]\n")
            feed = TranscriptRoomFeed(args.transcripts).load()
        result = bridge_room(feed, write=args.write, hour=args.hour)
        _print_live(feed, result, args.write)

    results = demo_three_states()
    _print_three_states(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
