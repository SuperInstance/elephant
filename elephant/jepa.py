"""JEPA backbone — optional learned encoder for dial time-series.

The dials in `dials/` are hand-crafted v0 senses (the fleet pattern:
hand-crafted first, learned second — same as vibe_matcher -> audio-jepa-v2).
This module is the learned side: a small JEPA that watches a dial's
windowed readings over time and learns to PREDICT the next room-state
embedding, EMA + stop-gradient + VICReg — the same skeleton as
fleet-jepa-midi `src/jepa/`.

Not required to use the repo. Import torch only when you want to grow
the elephant's learned skin.
"""

DIAL_SERIES_HELP = """
Each Dial.series(room, window) returns windowed readings; feed them
here to learn the room's dynamics. The JEPA objective is masked-window
prediction: given past readings, predict the future field embedding —
acclimation is the prediction error shrinking as the room stabilizes.
"""
