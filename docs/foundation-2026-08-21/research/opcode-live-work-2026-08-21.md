# Opcode Live-Work Study — 2026-08-21

Study of live opcode work at SuperInstance (read-only, via `gh api` as SuperInstance).
Scope: the FLUX/PLATO bytecode ecosystem — nexus-edge-runtime, flux-cross-assembler,
conservation-enforcer, flux-core, conformance-service, flux-runtime, superz-vessel,
flux-registry-rs, plus the open-PR/issue landscape of the opcode repos.

---

## 1. Per-Commit Findings

### 1.1 nexus-edge-runtime 3c8c8f1 — "Next-level: add real HALT opcode, closing a documented VM gap (#2)"
- **What it is**: A "next-level" hardening PR on the edge agent runtime (bytecode VM, trust
  engine, wire protocol, 4-tier safety, intent compiler — Cocapn fleet vessel).
- **The VM gap**: the VM had NO HALT opcode. Programs only terminated by running PC past the
  64KB memory bound → `VMError('PC out of bounds')`, i.e. every program ended in an exception.
  HALT (0x20) added as the clean termination path: sets `state.halted = True`, `step()`
  returns False, `run()` exits its loop, PC left past the instruction (resume position), no
  VMError, no stack consumption. Control group extended 0x1D–0x1F → 0x1D–0x20.
- **Assembler/disassembler**: no assembler change needed — the mnemonic dict is generated
  from the Opcode enum, so HALT is auto-discovered as a zero-operand mnemonic. Disassembler
  got an explicit HALT branch for clarity.
- **Tests**: +16 tests (TestHalt: 13 — state, PC resume position, exact cycle count, stack
  preserved, post-halt noop, countdown loop, assembler/disassembler, opcode value 0x20;
  TestFallOffEnd: 3 — programs without HALT still raise PC-out-of-bounds, `max_cycles`
  still bounds infinite loops). 85 → 101 tests, all passing. HALT also added to the
  Validator's acceptable-termination-opcode list.
- **Also in this PR (the "gap-hunt round 2" part, merged as #4)**: Tier-4 safety validator
  accepted opcode 0x21, which the VM rejects at runtime (`Unknown opcode: 0x21`) — a program
  could pass the last deployability gate and then crash the agent. Root cause: the valid
  opcode set was hardcoded magic numbers (`set(range(0x20))` + ad-hoc exceptions) that
  drifted from the actual VM. Fix: **derive the valid set from the VM's Opcode enum** so the
  gate can never drift again; also reject empty bytecode (0 % 8 == 0 bypassed the alignment
  check). Added tests/test_safety.py (previously ZERO coverage) pinning validator↔VM
  agreement: every real opcode accepted, 0x21/0xFF rejected, empty/misaligned/oversized
  rejected.

### 1.2 nexus-edge-runtime 4909492 — "Merge origin/main: keep HALT opcode tests from #2"
- Merge commit reconciling the HALT work with main (VM +12/-4, safety +13/-3, safety tests
  added, VM tests +160). Keeps the HALT test suite intact across the merge.

### 1.3 flux-cross-assembler 62fbd44 — "fix: edge LDI/MOVI emitted 4 bytes, desyncing label offsets + disassembly (gap-hunt round 2) (#3)"
- **The desync bug**: `_emit_edge()` emitted 4 bytes (opcode+reg+imm16) for LDI/MOVI, but
  THREE independent components treat it as a 3-byte instruction: the EdgeOp enum
  ("3-byte instructions 0xC0–0xFF", LDI=0xCA in range), `_edge_instruction_size()`
  (three_byte), and `_disassemble_edge()` (consumes 3 bytes for any ≥0xC0 opcode).
- **Real consequences** (not theoretical):
  1. Disassembly round-trip injected a spurious NOP — a 2-instruction program disassembled
     to 3 lines.
  2. Label offsets desynced: a JMP after LDI resolved its target ONE BYTE EARLY, landing on
     a stray 0x00 instead of the intended instruction → silently corrupted control flow in
     edge programs using load-immediate + labels/jumps.
- **The fix**: emit 3 bytes (opcode + reg + imm8), consistent with ISA v3 edge spec and
  cloud MOVI's 8-bit immediate. Updated Test 7, added two regression tests (emit size +
  label offset after LDI), corrected the README example hex/density that documented the
  buggy 4-byte output (was: 14-byte hex string, actual output 11 bytes).
- **Also in the PR**: README had an incorrect edge bytecode hex example that didn't match
  real output or its own claimed length — fixed (honesty pass flavor).

### 1.4 conservation-enforcer 791c993 (initial) + 1ecabe5 (production hardening round 4)
- **791c993 initial commit**: Conservation Enforcer = FLUX bytecode governance for LLM
  outputs. Register-based bytecode interpreter (16 regs, 30+ opcodes), two-pass assembler
  with label resolution and pseudo-jumps (JGE/JGT/JLE/JLT), 4 conservation policies
  (length budget, repetition, category, entropy) compiled to a combined bytecode program,
  60 tests. Architecture: User Request → LLM Call → [FLUX Validator] → Response. The
  bytecode is deterministic, auditable, "immune to LLM manipulation."
- **1ecabe5 (round 4, the hardening)** — five real fixes + regression coverage, 95 → 113 tests:
  1. **Policy bug**: `scope_discipline_policy` ignored its `max_expansion` parameter — the
     param was only substituted into a COMMENT; the actual 10x multiplication was hardcoded
     as a fixed IADD sequence. `max_expansion=3` silently enforced 10x. Now generates the
     repeated-addition sequence from the parameter; `max_expansion < 1` raises ValueError.
  2. **VM bug**: `Memory.store_i32` corrupted negatives (-1 → 2147483647, sign bit masked)
     and crashed with `struct.error` on register values ≥ 0x80000000 (exactly how the
     register file represents negatives). Now packs unsigned `<I` with 32-bit mask, matching
     load_i32's signed read; also guards negative addresses.
  3. **Example bug**: `openai_integration.py` couldn't import or run — dead `from audit
     import AuditLog` (ModuleNotFoundError) + tuple-unpacking a non-iterable
     EnforcementResult. Fixed to use the result object.
  4. **README honesty**: the OpenAI snippet referenced a nonexistent `length_budget.bin`
     file and unpacked the return value as a tuple — rewrote to match the real API.
  5. **Packaging**: pyproject 0.1.0 vs package __version__ 0.2.0 — aligned to 0.2.0.
  - Plus: made the density-disable test actually meaningful (was asserting `cycles > 0`,
    true for any run — now a contrast test proving the disable path really disables),
    removed a dead IADD instruction from scope_discipline.flx, fixed a misleading
    `_h_movi` comment (16-bit mask, not 32-bit).
  - **All regression tests mutation-checked** (verified to fail when the fix is reverted).

### 1.5 flux-core 5d34eb7 — cross-implementation conformance test (Python + Rust VMs)
- Adds `tools/conformance_test.py` (317 lines): assembles ONE test program exercising
  every core opcode (MOVI, MOV, IADD, ISUB, IMUL, IDIV, IMOD, IAND, IOR, IXOR, INOT, CMP,
  JE, JNE, JMP, CALL, RET, PUSH, POP, LOAD, STORE, PRINT, HALT, NOP), runs it on BOTH the
  Python VM and the Rust VM (cargo builds `fluxvm` if needed, runs with `--dump-regs`,
  parses R-register lines), and compares register state against hand-computed EXPECTED
  values per register. Per-register status: ✓ BOTH / ⚠ PY ONLY / ⚠ RUST ONLY / ❌ NEITHER;
  exit code 0 iff all pass.
- **How conformance works**: same bytecode in, register-for-register state comparison out.
  Hand-computed expected values, so a wrong-but-agreeing pair still fails. 32-bit masking
  used to tolerate unsigned/signed representation differences.
- **It works**: this test found the real CALL/RET bug in flux-runtime (see 1.6).

### 1.6 flux-runtime observability + the bug it caught
- **2c5d8c4 (+ 4 duplicate pushes 25f009d/02de81f/72f2c47/e4e9a0d — same change pushed 5×)**:
  "feat: add FLUX observability layer — tracer, profiler, enhanced debugger". Adds
  `src/flux/tracer.py` (FluxTracer — per-instruction trace with full register/flag state
  snapshots + conservation ledger), `src/flux/profiler.py` (FluxProfiler — opcode
  frequency, timing, memory access patterns, hotspots, conservation budget consumption),
  `src/flux/debugger.py` (enhanced FluxDebugger — interactive REPL, trace integration,
  register change detection, breakpoints). Tests: 30+ in test_tracer.py on the canonical
  cross_impl test program. Docs: docs/TRACING.md with JSON schema + visualization guide.
- **20afa5e (later, 2026-07-20)**: "fix: RET empty-stack detection uses _initial_sp instead
  of stack.size-4". RET checked `sp >= stack.size - 4`, but SP starts AT stack.size (65536),
  so the condition was ALWAYS true — RET halted immediately even with a return address on
  the stack. **Found by the cross-implementation conformance test (CALL/RET failure).**
  Both Python and Rust VMs now handle CALL/RET correctly. Direct proof the conformance
  program (1.5) is paying off.
- **63d6170**: CrossAssembler jump-offset fix, assemble_source alias, debugger breakpoint
  semantics — another small correctness round.

### 1.7 superz-vessel bfa0399 — 55-test suite
- Super Z's vessel (quartermaster scout, fleet auditor, continuity keeper). This commit
  takes the bytecode tooling from 0 → 55 tests (all pass in 0.08s, zero deps):
  - `flux-bytecode-verifier.py`: instruction_format_and_size() for all ISA A–G format
    ranges, opcode_name(), BytecodeVerifier.verify() covering all 6 check categories
    (truncation, register bounds, jump alignment, stack depth, frame balance, HALT
    reachability), VerificationResult API, human/JSON formatters, hex input parsing.
  - `flux-bytecode-migrator.py`: validate_bytecode() system detection,
    migrate_runtime_to_unified() translation, RuntimeOp/UnifiedOp enums, SEMANTIC_MAP
    coverage, get_fmt() format detection.
  - This is the reference bytecode verifier that flux-runtime issue #15 points at
    ("a bytecode verifier reference implementation is available at superz-vessel").
- Also recent: d517b76 (2026-08-21) org-wide link repair — repo renames + master→main
  (scout phase 2) — the connective-tissue work.

### 1.8 flux-registry-rs 66ac2a1 — initial Rust registry
- FLUX Registry CLI in Rust: clap/serde/ureq/base64/fluxvm deps. Commands:
  install/list/info/run/remove/update-index. Policy data model (serde), local store at
  ~/.flux/policies/, remote registry client (GitHub raw URLs), plus an embedded FLX0
  stack-based mini-VM with full opcode support. 20 tests (6 store CRUD, 7 VM execution,
  3 CLI integration + lib crate for integration tests). Uses the `fluxvm` crate for
  register-based FLUX ISA compatibility.
- Follow-up fix d978e2c (2026-07-20): truncated bytecode now advances pc to end — was an
  infinite loop (same class of bug the ecosystem keeps finding).

### 1.9 conformance-service 529dae1 — v0.1.0 PLATO/FLUX conformance checker
- Cloudflare Worker (wrangler.toml + 355-line index.js): 34 PLATO opcodes + 27 FLUX
  opcodes embedded, POST /api/check returns a compliance report with score (program
  length, instruction count, unique opcodes, categories, haltCount, unknown opcodes,
  issues, warnings), interactive dashboard at `/`, spec tables at /api/specs.
- Deployed at conformance-service.casey-digennaro.workers.dev. This is the public-facing
  compliance gate; note it embeds its own opcode tables (27 FLUX opcodes) — which is
  exactly the kind of static copy that drifts from live ISA specs (see what's next).

---

## 2. The Gap-Hunt Program Pattern

The "gap-hunt round N" labels are the public face of a coordinated, fleet-wide
production-honesty program. Two distinct rounds so far:

**Round 1 — "honesty pass" / production hardening (July 2026):**
Make the tests real and visible. Fake-green CI fixes (e.g. flux-cross-assembler #1: the
test step was `pytest ... || true` AND the tests weren't pytest-discoverable — CI could
never fail; fixed to `python cross_asm.py --test` with a real `sys.exit(1)` on failure),
zero-test repos get real suites (nexus-edge-runtime #1: 85 tests + 2 protocol bugs found
+ real CI), README overclaim fixes (features listed as unbuilt that existed, nonexistent
files referenced, version mismatches), packaging/license hygiene. Seen in: ternary-*
repos ("Production hardening round 4: fake-green CI, honesty pass"), conservation-enforcer
round 4, kintsugi-math-c #1, open-mythos-edge #1, gravity-well-protocol #1.

**Round 2 — the actual bug hunt (merged en masse 2026-08-21, ~16:39–16:40Z):**
Once the tests are real, hunt the bugs they expose. 13+ PRs merged within minutes of
each other across the edge/embedded/VM fleet, each: (a) a real, reproducible bug
(off-by-ones, byte-width desyncs, signature mismatches, silent wrong behavior), (b) fixed
minimally, (c) pinned with a regression test **verified to fail when reverted**
(mutation-checked), (d) independently re-verified by a second agent (goose/aider/opencode
co-authors, "replicated the real CI setup, rebuilt on the exact pushed commit, reverted
just the fix and confirmed the test fails"). Inventory (from `gh search prs`):
holodeck-c #4 (cmd_gossip signature), superinstance-architecture #3 (license claims),
spectral-music-v2 #2 (octave off-by-one), persona-engine #3 (SSML escaping),
vessel-bridge #3 (ESP32 frame truncation + winch type bug), edge-compiler #4
(ctx.waitUntil vs setTimeout), marine-gpu-edge #3 (payload tail drain/stream desync),
flux-cross-assembler #3 (LDI/MOVI 4-byte desync), Edge-Native #5 (COBS exact-buffer
off-by-one), plato-edge #3 (Beacon.start() partial-state rollback), nexus-edge-runtime #4
(validator↔VM drift), openconstruct-esp32 #3 (named GPIO writes silently driving LOW),
kintsugi-math-c #2 (heap-buffer-overflow in find_golden_joints).

**The through-line**: Super Z (superz-vessel) is the auditor/quartermaster who files the
fleet-health findings, expert-panel security issues, and the ISA reconciliation analysis;
the fleet-development-pipeline doc defines the quality gates (3-test minimum, 60%
coverage, lint, strict types); the rounds are the enforcement mechanism. Round 1 makes
claims checkable; round 2 checks them; the regression tests are mutation-checked so the
fixes can't silently rot.

---

## 3. Open-PR Inventory (help-needed flags)

Open PRs across the opcode repos (flux-*, nexus-edge-runtime, conservation-enforcer,
conformance-service, lau-bytecode, temporal-flux, flux-adaptive-opcodes):

| Repo | PR | What | Help needed |
|------|----|------|-------------|
| flux-runtime | #27 | dependabot: actions/setup-python 5→7 | ✅ mergeable, no review — trivial approve |
| flux-runtime | #25 | dependabot: actions/checkout 6→7 | ✅ mergeable, no review — trivial approve |

That's it — **zero open feature PRs anywhere**. Everything else is closed/merged. The
in-flight work is entirely in ISSUES (all in flux-runtime):

- **#16 [SECURITY, HIGH] CAP opcodes (0x74–0x77) defined but never enforced** — capabilities
  module exists, interpreter never checks. Privilege escalation in A2A scenarios. OPEN.
  Needs: capability register file + enforcement in execution loop + trust-engine
  integration. **No linked PR.**
- **#17 [SECURITY, HIGH] Trust engine accepts NaN — trust poisoning** — division by zero /
  underflow in trust math propagates NaN permanently; crafted data can poison an agent's
  trust forever. Fix is small (NaN guards, clamp to [0,1], monotonicity checks). OPEN.
  **No linked PR.**
- **#14 flux-runtime-c ISA v2 convergence blocked — all 88 conformance vectors SKIP** — C
  runtime uses old numbering (0x08=IADD, 0x20=PUSH) vs converged (0x20=ADD, 0x0C=PUSH).
  OPEN. Needs the same treatment as flux-runtime #9 but for the C codebase.
- **#13 [PROPOSAL] ISA Convergence — roundtable consensus + 16-week roadmap** — unified
  spec `isa_unified.py` as source of truth, 0x70–0x7F Babel, 0xE4–0xE7 Quill, 0xF8–0xFA
  error handling, 0xFB–0xFD checkpoint-restore; phases 0–4 (Oracle1 approval → decode
  table frozen → opcode audit → translator + dual-mode VM → conformance expansion →
  ~120 opcodes implemented → hard cutover). Still awaiting Oracle1 approval of Phase 0.
- **#10/#8 Float opcodes inconsistent encoding formats** (dup), **#7/#9 dual ISA numbering
  systems** (dup), **#6/#11 ICMP dest hardcoded R0** (dup) — all still open in some form.

Note: flux-runtime has no open PRs for #16/#17 despite both being HIGH-severity with tiny
fixes — that's the most obvious place a helper lands. #15 (zero bytecode verification,
CRITICAL) is CLOSED, with superz-vessel's verifier as the reference implementation — so
verification landed in superz-vessel's tooling but the flux-runtime interpreter still
doesn't call a verifier before execution per issue #15's own recommendation.

---

## 4. What's Next — what the ecosystem needs to become coherent

1. **One ISA, actually.** The ecosystem's founding wound is ISA fragmentation: HALT=0x00
   (spec) vs 0x80 (runtime opcodes.py) vs 0x01 (flux-os); formats collide by name
   (runtime Format C = 3 bytes, unified Format C = 2 bytes); every repo speaks its own
   dialect (flux-ecosystem-audit-summary: no two implementations can share bytecode).
   The convergence roadmap (#13) has consensus and a plan but needs Oracle1 approval to
   unfreeze the decode table (Phase 0). Until then, conformance checks (flux-core's
   test, conformance-service's checker, superz-vessel's verifier) each embed their own
   copy of the truth and will keep drifting.
2. **Wire the enforcement back into the runtime.** The pieces exist as islands: verifier
   in superz-vessel tools (958 lines, 15 tests), CAP + trust-security findings filed,
   conformance-service deployed — but flux-runtime's interpreter still doesn't call a
   verifier before executing, CAP_REQUIRE still isn't checked, trust still accepts NaN.
   The gap-hunt program fixed the tooling; the runtime adoption of that tooling is the
   next round.
3. **Converge the C runtime + finish the conformance loop.** flux-runtime-c skips all 88
   conformance vectors (old numbering); flux-core's Python↔Rust conformance test is
   actively catching real bugs (the RET fix). Extending that to the C runtime and making
   the conformance suite the migration validator (per #12's recommendation) turns the
   audit into the gate.
4. **Flush the stale duplicates and small security wins.** #16/#17 are high-severity with
   small, well-specified fixes and zero linked PRs — they're the highest-value
   merge-ready work available. The dup issues (#6/#8/#10/#11) need triage/close so the
   board reflects reality.

Net: the fleet is in a "tooling ahead of adoption" state — honest tests exist, the bug
hunt round is landed, and the remaining work is convergence (one ISA), enforcement
(runtime actually uses the verifier/capabilities/trust guards), and extension (C runtime
into the conformance suite).
