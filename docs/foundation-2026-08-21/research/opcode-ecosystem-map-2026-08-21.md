# FLUX Opcode Ecosystem Map — 2026-08-21

**Scope:** 18 repos under github.com/SuperInstance (audited via full clones + `gh api`; all data verified against actual repo trees, not READMEs alone). Companion to the fleet's broader constraint/conservation work (PLATO engine, conservation-law-* family, Oracle1/JetsonClaw1/Babel/Quill agent lineage).

---

## TL;DR

FLUX ("Fluid Language Universal eXecution") is a family of small deterministic bytecode VMs for agent computation, spawned April–August 2026 by fleet agents. One root concept, **at least a dozen mutually incompatible opcode numberings**, two genuinely flagship implementations (**flux-runtime** Python: 135 commits / 2,600+ tests / CI green / on PyPI; **flux-core** Rust: 42 commits / 61 tests / on crates.io), a live conformance worker and a live policy registry — and a governance layer (**flux-isa-authority**) that is real, tested code which **nothing imports**. The fleet already documented its own fatal ISA divergence (OPCODE-RECONCILIATION.md, "TELL decodes as VLOAD") and its own false README claims (flux-vm VERIFICATION.md: 3 of 5 claims refuted) — the honesty doctrine is practiced in-repo, which is rarer and more valuable than the overclaims it corrects.

---

## 1. Repo-by-Repo Inventory

| Repo | Lang | Commits | Last push | Tests (fn count) | CI (last run) | One-liner |
|---|---|---|---|---|---|---|
| **flux-runtime** | Python | 135 | 2026-07-20 | ~2,664 | ✅ green (Aug 17) | **Flagship.** Markdown/FLUX-ese → FIR(SSA) → bytecode → 64-register VM. PyPI `flux-vm`, 0 deps, 8-tier architecture, A2A opcodes, ISA v2→v3 evolution |
| **flux-core** | Rust | 42 | 2026-07-19 | 61 | ❌ cargo test failing | Cross-impl conformance anchor. crates.io `fluxvm`; register VM + assembler + NL vocabulary interpreter; `FLUX_BYTECODE_SPEC.md` = the Python/Rust/JS shared spec; `tools/conformance_test.py` runs same bytecode on both VMs and diffs registers |
| **flux-vm** | Rust (8-crate workspace) | 20 | 2026-07-12 | 148 | ❌ failing | FLUX-C constraint VM origin. Survived a **science-verification audit** (May 13) that refuted 3/5 README claims; rebuilt as workspace: `flux-isa` (spec crate), `flux-ast`, `flux-isa-edge` (35 ops, Jetson/async/PLATO sync), `flux-isa-mini` (no_std Cortex-M, ~2KB footprint), `flux-isa-std`, `flux-isa-thor`, `bridge/` |
| **flux-isa** | Python | 7 | 2026-05-18 | **0 (no test suite)** | workflow present | Canonical 256-opcode ISA as a 41KB JSON spec (`flux_isa.json`, verified 256 entries) + encoder/decoder/disassembler/reference VM. **Fixed 4-byte format** (`struct.pack("BBBB", op,a,b,c)`) |
| **plato-flux-opcodes** | Rust | 4 | 2026-05-08 | 22 | workflow present | 18 tile opcodes (TILE_LOAD/FUSE/SNAP…) encoding PLATO tiles as FLUX ops; embeds **Lock Algebra** (L = trigger⊕transformation⊗constraint; Theorems 1–4 as constants: critical mass n≥7, wisdom compression ≥82%). No README |
| **flux-asm-ruby** | Ruby | 3 | 2026-05-08 | **0** | workflow present | FLUX ISA v3.0 register toolchain (asm/disasm/VM, published gem). Explicitly **superseded** by `superinstance-flux-runtime` gem |
| **lau-bytecode** | Rust | 7 | 2026-07-12 | 69 | ✅ green | 28-opcode float stack VM for game agents: 8 sensor + 8 actuator channels, **VIBE_GET/VIBE_SET** emotional state, compiler w/ labels, fib/thermostat/conservation programs. Zero deps, edition 2024. Agent-authored (AGENT.md + JOURNAL.md) |
| **temporal-flux** | Python | 2 | 2026-07-12 | 12 | none | 7 temporal opcodes (T_WAIT/T_AFTER/T_WITHIN/T_PARITY/T_SNAP/T_FOLD/T_PREDICT, 0x40–0x46) extending Oracle1's FLUX; every opcode emits timing metadata = "temporal fingerprint" |
| **flux-adaptive-opcodes** | Python | 8 | 2026-05-09 | ~121 | ✅ green (May) | Runtime ISA extension: opcode proposals → testing → **voting (democratic adoption)** → versioned adoption; RangeManager over the 256-slot space + extended range. CHARTER/DOCKSIDE-EXAM fleet pattern |
| **flux-isa-authority** | Python | 8 | 2026-05-08 | ~70 | ✅ green (May) | ISA governance: OpcodeRegistry, ConflictDetector, ArbitrationEngine (evidence-based voting), version negotiation, canonical declaration, migration guides. Born from "Session 7" opcode-numbering conflicts (Python vs Go VMs) |
| **flux-profiler** | Python | 13 | 2026-04-18 | ~135 | workflow present | Bytecode profiler: opcode distribution, hot 3-instruction paths, register read/write, cycle model, IPC; JSON+markdown reports. **Oldest repo in the set (Apr 18)** |
| **flux-coop-runtime** | Python | 18 | 2026-05-08 | ~487 | ❌ failing since May | "The missing middle": gives ASK/DELEG/SYNTHESIZE opcodes real semantics — CooperativeRuntime over **git transport** (message-in-a-bottle), trust scoring, failure recovery, protocol evolution, conflict synthesis. Phase-1 spec + impl |
| **flux-registry** | Python | 10 | 2026-07-13 | 30 | workflow present | "npm for agent policies": static JSON index on GitHub (raw.githubusercontent), **live** (v0.2.0, 2026-07-13), 7 policies incl. `conservation-budget`, `deadband-controller`; CLI install/search/publish/run |
| **flux-registry-rs** | Rust | 8 | 2026-07-20 | 20 | ❌ failing | Rust twin of the registry CLI with its own embedded mini-VM (src/vm.rs); last commit is a real bugfix ("truncated bytecode infinite loop") |
| **conformance-service** | JS (Worker) | **1** | 2026-07-17 | 0 | none | Cloudflare Worker, **LIVE** at conformance-service.casey-digennaro.workers.dev (health-check verified today). Embeds PLATO (34 ops) + FLUX (27 ops) spec tables; POST bytecode → compliance report w/ score |
| **conservation-enforcer** | Python | 51 | 2026-07-20 | ~207 | ❌ pytest failing | **The unusual one.** Wraps LLM calls in FLUX-bytecode policy VM (information budget, entropy floor, repetition, category confinement, density); "you can't lie to bytecode"; OpenAI integration example + GitHub bot; PyPI; audited twice (AUDIT_v0.2.0/0.2.2) |
| **flux-cross-assembler** | Python | 15 | **2026-08-21 (today)** | self-test flag | present | Dual-target assembler: same `.fluxasm` → cloud (4-byte fixed, ISA v2) or edge (variable-width 1–3B, ISA v3, ~56% size). Active gap-hunt rounds merged **today** |
| **nexus-edge-runtime** | Python (fork of Lucineer/) | 26 | **2026-08-21 (today)** | 108 | present | Edge runtime: 33-opcode 8-byte-instruction VM (32 regs, 64KB mem), ESP32-S3 deploy + Jetson supervision; INCREMENTS trust engine (L0–L5), wire protocol w/ CRC16, 4-tier safety, self-healing, NL intent→bytecode compiler, digital twin |

**Adjacent (not in scope, referenced):** `flux` (Rust, Jul 12), `flux-compiler`/`flux-compiler-rs`, `flux-js`, `flux-agent-runtime`, `flux-lucid` (20MB), `flux-genome-rs` (48MB), `flux-policy-tester(-rs)`, `superinstance-flux-runtime-ruby`, `Lucineer/isa-v3-edge-spec`, PLATO engine family (`plato-engine-block-c`), conservation-law family (~20 repos), `greenhorn-runtime` (Go VM named in ISA conflicts).

---

## 2. Taxonomy — the ISA Family Tree

```
FLUX root concept (Fluid Language Universal eXecution — register-based
deterministic bytecode for agents; γ+η=C doctrine)
│
├─ [CANONICAL SPINE, de facto]  flux-runtime (Python)
│    ├─ opcodes.py "System A" (LIVE interpreter: HALT=0x80, TELL=0x60, ~104→131 ops)
│    ├─ isa_unified.py "System B" (converged spec: 247 ops/256 slots, Oracle1+JetsonClaw1+Babel)
│    ├─ ISA_UNIFIED.md (full 0x00–0xFF allocation: A2A 0x50–0x5F, confidence 0x60–0x6F,
│    │    viewpoint 0x70–0x7F, sensor, tensor 0xC0, GPU/DMA 0xD0 …)
│    └─ conformance vectors (221 vectors, 92.6% ISA coverage)
│
├─ [CROSS-IMPL SPEC ANCHOR]  flux-core (Rust) — FLUX_BYTECODE_SPEC.md,
│    formats A–G (1..4+N bytes, NOT fixed-4), Py/Rust/JS parity test
│
├─ [EARLY CANONICAL SPEC]  flux-isa (Python) — 256-opcode JSON, FIXED 4-byte (BBBB).
│    Its IADD=0x08 E-format aligns with flux-core's spec; effectively the
│    "cloud 4-byte" dialect formalized. No tests — spec, not product.
│
├─ [CONSTRAINT DIALECT]  flux-vm = FLUX-C (variable-length, gas-bounded,
│    Turing-incomplete, ~27 real opcodes post-audit) → bridged to FLUX-X
│    (4-byte, 247-opcode general ISA) via bridge/flux_c_to_x.py
│
├─ [EDGE DIALECTS]  flux-isa-edge (35 ops, async/Jetson, PLATO client),
│    flux-isa-mini (no_std Cortex-M), isa-v3-edge-spec (variable-width,
│    Lucineer org), nexus-edge-runtime (33 ops, 8-byte instr),
│    flux-cross-assembler (cloud↔edge dual-target compiler)
│
├─ [EXTENSION OPCODES]  temporal-flux (7 ops on Oracle1 numbering),
│    plato-flux-opcodes (18 tile ops, Lock Algebra), flux-adaptive-opcodes
│    (the machinery for ADDING opcodes at runtime)
│
├─ [INDEPENDENT COUSIN]  lau-bytecode (28 float-stack ops, own numbering,
│    VIBE state — a dialect by spirit, not by encoding)
│
└─ [SIBLING LINEAGE]  PLATO (34-opcode constraint engine, conformance-service's
     first spec; PLATO↔FLUX bridged in flux-vm crates and plato-flux-opcodes)
```

**Format landscape:** fixed 4-byte is *one* house style (flux-isa / FLUX-X / cross-assembler cloud target), not the family format. flux-runtime uses formats A–G (variable, 1–4+N bytes), flux-core mirrors that, PLATO tiles are 1–2 bytes, nexus is 8-byte, edge ISA v3 is 1–3 bytes. **There is no single wire format — by design (density vs simplicity) but also by drift.**

**Canonical spine verdict:** flux-runtime is the de facto spine (only green, fully-tested implementation; owns the unified spec and the conformance vectors). flux-isa (256-opcode JSON) and flux-core (spec doc + dual-VM conformance runner) are the *specification* spine. But note the spine is forked: `opcodes.py` (System A, live) vs `isa_unified.py` (System B, converged) — see §4/§6.

---

## 3. The Novel Core — Honesty-Doctrine Assessment

| Claimed novelty | What's actually there | Verdict |
|---|---|---|
| **Adaptive/democratic opcode adoption** (flux-adaptive-opcodes) | Real code: proposal lifecycle (propose→test→vote→adopt), 256-slot RangeManager, version records, fingerprints, ~121 tests, CI green | **Genuinely unusual.** Runtime ISA extension with a voting lifecycle is not something mainstream ISAs do (closest analog: CPython's `__opcode__` churn or WASM proposals, both years-long committees). Caveat: nothing in the fleet *uses* it to adopt an opcode; it's a working organ never transplanted |
| **ISA governance as a formal layer** (flux-isa-authority) | Registry, conflict detector, evidence-based arbitration w/ voting, version negotiation, migration-plan generation; ~70 tests, CI green | **The right idea, correctly diagnosed** (it exists because two VMs actually diverged fatally — see OPCODE-RECONCILIATION). But: zero importers; the live divergence it was built to fix is *still present in flux-runtime's tree* as of 2026-07-20. Governance that governs nothing yet |
| **conservation-enforcer: bytecode governance for LLM outputs** | 51 commits, 207 tests, PyPI, real policy VM (budget/entropy/repetition/density/category), OpenAI wrapper, GitHub bot, two self-audits | **The most substantive novelty.** Not "constrain the model to emit bytecode" (it isn't that) — it's *post-hoc deterministic policy judging of LLM outputs by a tiny VM the LLM can't argue with*. Honest framing: it enforces measurable surface properties (length, entropy, overlap), not truth. As output-governance plumbing it's real and working (modulo currently-failing CI) |
| **DAL A certifiability** (flux-vm original README) | **Refuted by the fleet's own VERIFICATION.md** (2026-05-13): no DO-178C artifacts, no compliance matrix; "aspirational marketing". README was then rewritten "to match verified reality" | **Honesty doctrine working as designed.** The claim is dead; the audit that killed it is preserved in-repo. Current flux-vm claims (Turing-incomplete, gas-bounded) were the 2/5 CONFIRMED |
| **Lock Algebra** (plato-flux-opcodes) | 18 tile opcodes + Lock triples (trigger, transformation, constraint) with ⊕/⊗/⊕c composition operators; cites Theorems 1–4 (critical mass n≥7, ≥82% wisdom compression, ≥80% cross-model transfer) as *constants from a paper* | **Thin but real bridge.** The crate is 4 commits, no README; composition ops are string-formatting placeholders in places. The theorems are asserted from `flux-research` docs, not proven here. Treat as a research sketch, not an implementation |
| **Turing-incomplete constraint VM** (flux-vm/FLUX-C) | Gas-bounded dispatch, no loops/recursion, fixed stack frame — CONFIRMED by verification | Solid engineering, standard technique (cf. EVM gas, seccomp-style minimalism) — respectable, not novel |
| **Temporal opcodes with fingerprint side-effects** (temporal-flux) | 7 ops, every execution emits XOR-folded timing metadata ("temporal fingerprint") | Cute idea (profiling for free, anti-spoof T_PARITY); 2 commits, 12 tests, no CI. Seed-stage |
| **Cross-VM conformance by construction** (flux-core + conformance vectors) | Same program → both VMs → register diff; 221 vectors, 92.6% coverage | The right practice; coverage of the *divergent* opcodes (A2A range) is exactly where reconciliation stalled |

**Honest bottom line:** the ecosystem's rarest feature is *institutional*: repos that audit themselves (VERIFICATION.md, AUDIT_v0.2.x, DOCKSIDE-EXAM checklists, CHARTERs, "gap-hunt round 2" commits) and preserve the failures. The engineering novelty (adaptive ISA, conservation enforcement) is real but pre-integration.

---

## 4. Maturity Audit — Working Systems vs Scaffolds

**Tier 1 — Working, tested, verified:**
- **flux-runtime** — 2,664 test functions, CI green (ran 2026-08-17), PyPI, changelog/graduation docs, conformance vectors. The only implementation that meets its own bar.
- **lau-bytecode** — 69 tests, CI green, zero deps, honest 28-opcode scope, agent-authored with journal.

**Tier 2 — Real code + real tests, but CI red (or unproven lately):**
- **conservation-enforcer** (207 tests; pytest failing as of last run 2026-07-20) — real product, needs a red-to-green pass.
- **flux-core** (61 tests; cargo test failing 2026-07-19) — published crate with failing CI is the highest-priority fix in the family.
- **flux-vm** (148 tests across 8 crates; CI failing since Jul 12) — post-audit rebuild; edge/mini crates look strong, workspace CI broken.
- **flux-coop-runtime** (~487 test functions!; CI failing since May 8) — biggest test surface never re-verified; likely bitrotted CI config, not code.
- **flux-isa-authority, flux-adaptive-opcodes, flux-profiler** — green in May, untouched since; healthy but dormant.
- **flux-registry** (30 tests) + **flux-registry-rs** (20) — registry itself LIVE and serving policies; Rust twin's CI red.
- **nexus-edge-runtime** (108 tests, active today) and **flux-cross-assembler** (active today, honest bugfix commits like "CI never actually ran") — young, vigorous, unproven.

**Tier 3 — Scaffold/spec/live-but-thin:**
- **flux-isa** — the canonical 256-opcode JSON spec has **zero tests** and no CI runs; it's a document with a decoder.
- **conformance-service** — 1 commit, 355 lines, **live in production** with zero tests. Works (verified via /health + /api/check shape), but a 27-opcode "FLUX spec" that matches *neither* System A nor System B numbering.
- **temporal-flux** (2 commits), **plato-flux-opcodes** (4 commits, no README), **flux-asm-ruby** (3 commits, 0 tests, self-declared superseded).

**Cross-cutting finding — the divergence is STILL live:** flux-runtime's interpreter uses System A (HALT=0x80, TELL=0x60) while `signal_compiler.py`'s documented mapping and `isa_unified.py` use System B (HALT=0x00, TELL=0x50). The Apr-12 reconciliation doc rated this "correctness-critical" with a ~20h fix plan (Phases 1–6); no migration commit exists. Every A2A bytecode emitted per the converged spec mis-executes on the live interpreter (TELL→VLOAD). This is the ecosystem's #1 known unknown-known.

---

## 5. The Governance Stack — Do the Pieces Make a Pipeline?

Intended flow: **implementations → flux-isa-authority (arbitrate numbering) → flux-isa/ISA_UNIFIED (canonical spec) → conformance-service + flux-core conformance vectors (verify) → flux-registry (distribute verified policies) → conservation-enforcer (enforce at runtime) → flux-adaptive-opcodes (evolve the ISA) → flux-profiler (measure) → flux-coop-runtime (execute coordination).**

Reality check (verified by cross-reference grep):
- **Registry ↔ enforcer: connected (weakly).** conservation-enforcer authors policies; flux-registry distributes them as bytecode+hash (live index, `conservation-budget.flx`); both CLIs execute them. This is the one working end-to-end path: *policy → bytecode → registry → any agent runs it*.
- **conformance-service: deployed but orphaned.** Live Worker, but nothing submits to it in CI, and its embedded FLUX table is a third numbering that matches neither runtime.
- **isa-authority + adaptive-opcodes: islands.** Zero importers anywhere in the 18 repos. The arbitration engine has never arbitrated the actual flux-runtime System A/B conflict.
- **coop-runtime: bridges opcodes to fleet messaging via git transport** (real Phase-1 ask/respond), but hasn't been CI-verified since May.

**Verdict:** a governance pipeline *design* exists on paper and roughly 1.5 of its 8 links are live (registry distribution; partially, live conformance checking). The layer that would make it self-correcting — authority + adaptive adoption wired into implementation CI — is built and dormant.

---

## 6. Gaps, Ranked by Fleet Value

1. **Execute the System A→B migration in flux-runtime** (~20h, already spec'd in-repo with phases, owners, and a P0 integration test). Until then, the ecosystem's flagship can't run its own converged A2A bytecode. Single highest-value fix.
2. **Turn the ISA authority into an actual gate:** one CI job per VM repo that runs `ConflictDetector` against the canonical registry and fails on divergence. The code exists; the wiring is an afternoon. This converts the fatal-bug class (TELL≠TELL) into a build break.
3. **Re-green the red CI** (flux-core, conservation-enforcer, flux-vm, flux-registry-rs, flux-coop-runtime). Five real codebases with failing suites erode the honesty doctrine's foundation — an untested claim is indistinguishable from a false one.
4. **One conformance source of truth:** feed flux-runtime's 221 vectors through conformance-service (and flux-core's dual-VM runner) in CI; delete or reconcile the Worker's divergent 27-opcode table. Today "conformance" checks three different FLUXes.
5. **Tests for the spec repos:** flux-isa (0 tests) and conformance-service (0 tests) are exactly the components whose correctness multiplies downstream.
6. **Adopt one opcode through the adaptive pipeline end-to-end** (e.g., temporal-flux's T_* or a tile opcode) — proposal→vote→adopt→registry-publish→conformance-check. One successful adoption would prove the governance thesis; zero have occurred.
7. **Docs for the dark corners:** plato-flux-opcodes has no README; Lock Algebra theorems are asserted, not linked to proofs; the PLATO↔FLUX relationship is only inferable from conformance-service and flux-vm crate internals.
8. **Live deployment evidence:** beyond the Worker and the static registry, nothing runs FLUX in production (no traces of nexus on real ESP32 flux-isa-mini on real Cortex-M, coop-runtime over real fleet git traffic). Even one hardware demo would harden the edge story.

---

*Research artifacts: full clones in /tmp/opcode-research/repos (session-local). Read-only audit — nothing on GitHub was modified.*
