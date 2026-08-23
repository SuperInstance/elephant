# Decomposition Risk Audit — `scripts/gen_openroom_tries.py`

**Repo:** `SuperInstance/elephant`  
**Commit:** `e3664ed`  
**Date:** 2026-08-22  
**Auditor:** Hermes (OB1)

## Summary

107-line image-generation orchestrator. Generates openroom hero portraits across Cloudflare + DeepInfra image models. Hard-codes secrets and machine-specific absolute paths. Audit finds **HIGH** risk for both invariants.

## Invariant Findings

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| I-1 | No hardcoded API tokens | **FAIL** | `CF_ACCOUNT` hardcoded line 21; `CF` token loaded from `/home/eileen/.config/.wrangler/config/default.toml` line 15; `DI` loaded from `~/.bashrc` line 6 |
| I-2 | No hardcoded absolute output paths | **FAIL** | `OUT = "/home/eileen/projects/elephant/assets/images/openroom-tries"` line 24 |
| I-3 | Portable location resolution | **FAIL** | Script depends on `/home/eileen/...` paths that will not resolve on `C:\Users\casey\...` or any other host |
| I-4 | Graceful failure on missing credentials | **FAIL** | Both `cf_token()` and `deepinfra_key()` silently return `""` if files/env missing; subsequent HTTP calls will 401 with no clear abort path |

## Additional Risks

- `urllib.request.urlopen(..., timeout=300)` — 5-minute timeout per model; 14 total models => worst case ~70 min runtime.
- No downloaded image integrity verification beyond `sz > 1000`. A partial image will pass.
- Output directory creation via `os.makedirs(OUT, exist_ok=True)` is fine, but relies on hardcoded path.
- Models list includes deprecated `cf-flux1-schnell` and `di-flux1-dev` (may 404 depending on provider state).

## Required Fixes

1. Replace `CF_ACCOUNT` with `os.environ.get("CF_ACCOUNT_ID", "")`
2. Replace hardcoded `OUT` with `os.environ.get("OPENROOM_OUT", "openroom-tries")`
3. Preserve `.bashrc` and wrangler-toml lookup as fallbacks, but **never commit the loaded values**
4. Add early abort if either token resolves to empty string
5. Verify PNG magic bytes after download (`b"\x89PNG\r\n\x1a\n"`)
