#!/usr/bin/env python3
"""Generate README hero images for the fleet's matured repos (CF FLUX, DI fallback)."""
import base64, json, os, re, subprocess, sys, time, urllib.request, urllib.error

def deepinfra_key():
    txt = open(os.path.expanduser("~/.bashrc")).read()
    m = re.search(r'DEEPINFRA_API_KEY=.*?"?([^"\'\s]+)', txt)
    return m.group(1) if m else os.environ.get("DEEPINFRA_API_KEY", "")

def cf_token():
    try:
        txt = open("/home/eileen/.config/.wrangler/config/default.toml").read()
        m = re.search(r'oauth_token\s*=\s*["\']([^"\']+)', txt)
        return m.group(1) if m else ""
    except Exception:
        return ""

CF_ACCOUNT = "049ff5e84ecf636b53b162cbb580aae6"
CF = cf_token()
DI = deepinfra_key()

JOBS = [
    ("/home/eileen/projects/elephant/assets/images/hero.png",
     "A warm wooden harbor bar at night called The Tap, seen from inside, empty of people but full of presence — a large calm barely-visible elephant-shaped warmth of amber light standing in the room like the air itself, wooden walls that seem to remember every conversation, one glass left on the counter, moonlight through a window, painterly atmospheric, cozy and slightly numinous"),
    ("/home/eileen/projects/study-signal-chain/assets/images/thesis.png",
     "A signal chain as a physical thing — a chain of glowing links running through a dim workshop room, half the links made of typed code glyphs in cold blue, half made of handwritten flowing prose in warm amber, one dial at the end of the chain reading the mix, painterly, atmospheric, industrial-romantic"),
    ("/home/eileen/projects/plato-perception/assets/images/perception.png",
     "A quiet harbor at dawn where every object — buoys, boats, gulls — casts a glowing numbered vector-trail of light into the air, a sensor net reading the world into thin luminous lines and coordinate points, cool blue light, painterly, atmospheric, the world becoming data"),
    ("/home/eileen/projects/plato-prediction/assets/images/prediction.png",
     "The same harbor at night, but the light-lines continue FORWARD of the objects — faint ghost-trails of where each boat will be, one boat's trail bending into a warning amber glow where the prediction flags an anomaly, deep blue night, painterly, atmospheric, the future visible as light"),
    ("/home/eileen/projects/plato-vision-jepa/assets/images/vision.png",
     "A camera's eye-view of a warm room (a bar) dissolving into a constellation of 16 labeled points of light hovering in the air, each point a dimension of the room's state — brightness, motion, occupancy — the constellation shaped like the room itself, amber and teal, painterly, atmospheric"),
    ("/home/eileen/projects/fleet-jepa-midi/assets/images/hero.png",
     "A dark recording studio at night where the music is visible — a pulse of light threading through three layers: a thought-cloud of a thinking mind at the top, a glowing heartbeat line in the middle, a fine rain of sample particles at the bottom, and in the corner of the room a barely-visible warm elephant-shaped presence, amber and deep blue, painterly, atmospheric"),
]

def render_cf(prompt):
    body = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/run/@cf/black-forest-labs/flux-1-schnell",
        data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + CF})
    with urllib.request.urlopen(req, timeout=240) as r:
        data = json.load(r)
    if data.get("success") and data.get("result", {}).get("image"):
        return base64.b64decode(data["result"]["image"])
    return None

def render_di(prompt):
    body = json.dumps({"prompt": prompt, "width": 832, "height": 832,
                       "num_inference_steps": 4, "seed": 777}).encode()
    req = urllib.request.Request(
        "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX-1-schnell",
        data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + DI})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    b64 = data.get("output") or data.get("images")
    if isinstance(b64, list):
        b64 = b64[0] if b64 else None
    if isinstance(b64, str):
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)
    return None

for path, prompt in JOBS:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"skip (exists) {path}"); continue
    ok = False
    for attempt in range(4):
        try:
            img = render_cf(prompt)
            if img and len(img) > 1000:
                open(path, "wb").write(img)
                print(f"OK (CF) {path} {len(img)} bytes"); ok = True; break
        except urllib.error.HTTPError as e:
            print(f"CF HTTP {e.code} — {'sleep' if e.code == 429 else 'fail'}")
            if e.code == 429: time.sleep(8); continue
        except Exception as e:
            print(f"CF ERR {e}")
        try:
            img = render_di(prompt)
            if img and len(img) > 1000:
                open(path, "wb").write(img)
                print(f"OK (DI) {path} {len(img)} bytes"); ok = True; break
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(8); continue
        except Exception as e:
            print(f"DI ERR {e}")
        time.sleep(4)
    if not ok:
        print(f"FAIL {path}")
print("=== done ===")
