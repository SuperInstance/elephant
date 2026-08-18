#!/usr/bin/env python3
"""Hermit-crab OpenRoom hero — tries across many models (CF + DeepInfra)."""
import base64, json, os, re, time, urllib.request, urllib.error

def deepinfra_key():
    txt = open(os.path.expanduser("~/.bashrc")).read()
    for line in txt.splitlines():
        if "DEEPINFRA_API_KEY" in line and "=" in line:
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return val
    return os.environ.get("DEEPINFRA_API_KEY", "")

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
OUT = "/home/eileen/projects/elephant/assets/images/openroom-tries"
os.makedirs(OUT, exist_ok=True)

PROMPT = (
    "A hermit crab that is a cyberpunk mech pilot, its shell a salvaged found-tech room "
    "with warm amber interior light glowing from inside, the crab's two claws extended as "
    "glowing agent-tools reaching into the dark, Titan AE meets Voltron meets Ghost in the "
    "Shell, lived-in worn metal, rivets and heat-stains, cyberpunk warmth, cinematic, "
    "detailed, serious and cool, not cartoonish"
)

def cf_gen(model, prompt, path):
    body = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/run/{model}",
        data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + CF})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    if data.get("success") and data.get("result", {}).get("image"):
        open(path, "wb").write(base64.b64decode(data["result"]["image"]))
        return os.path.getsize(path)
    print(f"  CF {model}: no image: {json.dumps(data)[:120]}")
    return 0

def di_gen(model, prompt, path):
    body = json.dumps({"prompt": prompt, "width": 1024, "height": 1024,
                       "num_inference_steps": 4, "seed": 777}).encode()
    req = urllib.request.Request(
        f"https://api.deepinfra.com/v1/inference/{model}",
        data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + DI})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    b64 = data.get("output") or data.get("images")
    if isinstance(b64, list):
        b64 = b64[0] if b64 else None
    if isinstance(b64, str):
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        open(path, "wb").write(base64.b64decode(b64))
        return os.path.getsize(path)
    print(f"  DI {model}: no image: {json.dumps(data)[:120]}")
    return 0

def gen(label, fn, model, prompt, path):
    for attempt in range(3):
        try:
            sz = fn(model, prompt, path)
            if sz and sz > 1000:
                print(f"OK  {label:28s} {os.path.basename(path)} {sz} bytes")
                return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(8); continue
            print(f"ERR {label}: HTTP {e.code}")
        except Exception as e:
            print(f"ERR {label}: {str(e)[:100]}")
        time.sleep(3)
    print(f"FAIL {label}")
    return False

# Cloudflare models (novel + proven)
cf_models = [
    ("cf-flux1-schnell",  "@cf/black-forest-labs/flux-1-schnell"),
    ("cf-flux2-dev",      "@cf/black-forest-labs/flux-2-dev"),
    ("cf-flux2-klein-9b", "@cf/black-forest-labs/flux-2-klein-9b"),
    ("cf-flux2-klein-4b", "@cf/black-forest-labs/flux-2-klein-4b"),
    ("cf-sdxl-lightning", "@cf/bytedance/stable-diffusion-xl-lightning"),
    ("cf-sdxl-base",      "@cf/stabilityai/stable-diffusion-xl-base-1.0"),
]
# DeepInfra models (novel + proven)
di_models = [
    ("di-flux1-schnell", "black-forest-labs/FLUX-1-schnell"),
    ("di-sdxl-turbo",    "stabilityai/sdxl-turbo"),
    ("di-flux1-dev",     "black-forest-labs/FLUX-1-dev"),
]

for label, model in cf_models:
    gen(label, cf_gen, model, PROMPT, f"{OUT}/{label}.png")
for label, model in di_models:
    gen(label, di_gen, model, PROMPT, f"{OUT}/{label}.png")

print("=== done ===")
for f in sorted(os.listdir(OUT)):
    print(f"  {f} {os.path.getsize(os.path.join(OUT, f))} bytes")
