#!/usr/bin/env python3
"""Quick probe: which DeepInfra models answer for the tap-night cast."""
import json
import re
import time
import urllib.request

KEY = re.search(r'export DEEPINFRA_API_KEY="([^"]+)"',
                open('/home/eileen/.bashrc').read()).group(1)

MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Pro",
    "NousResearch/Hermes-3-Llama-3.1-405B",
    "anthropic/claude-haiku-4-5",
    "ByteDance/Seed-2.0-pro",
    "zai-org/GLM-4.7-Flash",
    "zai-org/glm-4.7-flash",
]


def call(model, prompt, max_tokens=12, timeout=60):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
                return d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = str(e)
            if "engine_overloaded" in err or "busy" in err.lower() or "429" in err:
                time.sleep(3 * (attempt + 1))
                continue
            return f"ERR: {err[:100]}"
    return "ERR: overloaded after retries"


if __name__ == "__main__":
    for m in MODELS:
        out = call(m, "Reply with exactly: ping")
        print(f"{m:45s} -> {out[:60]}")
