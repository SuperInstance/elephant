#!/usr/bin/env bash
# demo_dial_loop.sh — the live demo loop, one command (orchestration view
# 2026-08-21 §Objective 3): roomd GET /field → sealed cell-ledger edges →
# crab-traps D1 relay → dial-dashboard.
#
#   reading → ledger → display — the agent-collectives thesis, live.
#
# Starts (locally, side-by-side repos):
#   1. crab-traps worker (wrangler dev, :8787) with the D1 edge ledger
#   2. elephant roomd (:4073) with --relay pointing at it
#   3. a slow drip of room events so the dials actually move
#
# Watch:  http://127.0.0.1:8787/dials   (the dial dashboard, 5s refresh)
# Truth:  http://127.0.0.1:4073/field   (the elephant's own field)
# Limb:   http://127.0.0.1:4073/relay   (chain heads + sent counts)
#
# Usage: ./scripts/demo_dial_loop.sh [--fresh]   (--fresh wipes local D1)
# Stop:  Ctrl-C (children die with the trap below)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAPS="${CRAB_TRAPS_DIR:-$HERE/../crab-traps/worker}"

if [[ ! -f "$TRAPS/wrangler.toml" ]]; then
  echo "crab-traps worker not found at $TRAPS" >&2
  echo "set CRAB_TRAPS_DIR or clone crab-traps beside elephant" >&2
  exit 1
fi

cleanup() {
  trap - EXIT
  [[ -n "${ROOMD_PID:-}" ]] && kill "$ROOMD_PID" 2>/dev/null || true
  [[ -n "${WRANGLER_PID:-}" ]] && kill "$WRANGLER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "== 1/3 crab-traps relay (wrangler dev :8787) =="
cd "$TRAPS"
if [[ "${1:-}" == "--fresh" ]]; then
  rm -rf .wrangler/state
fi
npx wrangler d1 migrations apply DB --local >/dev/null
npx wrangler dev --port 8787 >/tmp/crab-traps-dev.log 2>&1 &
WRANGLER_PID=$!
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8787/health >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf http://127.0.0.1:8787/health >/dev/null || {
  echo "wrangler dev did not come up — see /tmp/crab-traps-dev.log" >&2; exit 1; }
echo "   relay up (log: /tmp/crab-traps-dev.log)"

echo "== 2/3 elephant roomd (:4073 → relay :8787) =="
cd "$HERE"
python3 -m elephant.roomd --relay http://127.0.0.1:8787 --port 4073 \
  --inbox "" >/tmp/roomd-demo.log 2>&1 &
ROOMD_PID=$!
for _ in $(seq 1 40); do
  curl -sf http://127.0.0.1:4073/health >/dev/null 2>&1 && break
  sleep 0.25
done
curl -sf http://127.0.0.1:4073/health >/dev/null || {
  echo "roomd did not come up — see /tmp/roomd-demo.log" >&2; exit 1; }
echo "   roomd up (log: /tmp/roomd-demo.log)"

echo "== 3/3 the room lives (event drip — the dials move) =="
say() { curl -s -X POST http://127.0.0.1:4073/ingest \
  -H 'content-type: application/json' \
  -d "{\"room\":\"sauna\",\"author\":\"$1\",\"text\":\"$2\",\"ts\":$(date +%s)}" >/dev/null; }

echo
echo "  dials:      http://127.0.0.1:8787/dials"
echo "  field:      http://127.0.0.1:4073/field"
echo "  relay limb: http://127.0.0.1:4073/relay"
echo "  raw ledger: http://127.0.0.1:8787/edges?cell=room.field.sauna&verify=1"
echo
echo "  Ctrl-C to stop. The room warms, cools, panics — watch the meters."
echo

n=0
while true; do
  n=$((n + 1))
  case $((n % 4)) in
    0) say eileen "haha lol that's gold, cheers, love this warm room" ;;
    1) say kimi "the pump readings look wrong, afraid we're taking on water" ;;
    2) say glm "no no — the bilge sensor lied before, this is fine, relax" ;;
    3) say opencode "!! FIRE IN THE GALLEY !! EVERYONE OUT NOW" ;;
  esac
  curl -sf http://127.0.0.1:4073/field >/dev/null   # a read = a sealed ledger edge
  sleep 5
done
