# deepagents migration — architecture and isolation

How CLEANROOM-AGENT's agents run on LangChain's `deepagents`, and — the part that matters —
how the clean-room isolation guarantee survives the move.

## The guarantee

Three agents must never see each other's output:

| Agent | Derives | Must never see |
|---|---|---|
| **Code** | implementation | test cases (the oracle it is scored against) |
| **Test** | black-box tests | implementation code |
| **Proof** | Dafny proofs | tests |

This is the paper's central claim, not a style preference. A leak is a **correctness bug**,
and CI treats it as build-breaking.

## How isolation is enforced: `StateBackend`, not the tool list

The first thing to understand about `create_deep_agent` is that **you cannot take its
filesystem tools away**. `ls`, `read_file`, `edit_file`, `glob` and `grep` are always
installed; passing `tools=` is *additive*. So the tool list is not where isolation lives.

Isolation comes from the **backend** — what those tools can see:

| Backend | Files live | Escape risk | Verdict |
|---|---|---|---|
| **`StateBackend`** (default) | LangGraph state, per thread | **None** — no disk beneath it | ✅ **what we use** |
| `FilesystemBackend` | real disk under `root_dir` | **High.** Docs: *"the default (`virtual_mode=False`) provides no security even with `root_dir` set"* | ❌ rejected |
| `StoreBackend` | LangGraph Store, cross-thread | none, but **persists across threads** | ❌ rejected |
| `CompositeBackend` | routed per path prefix | inherits the risk of whatever it routes to | ❌ unnecessary |

We seed each agent's `StateBackend` with **exactly the files it is permitted to see**. The
Test Agent's filesystem does not contain the implementation and then hide it — the file does
not exist there at all. There is nothing to escape *to*, because there is no disk under a
`StateBackend` and no cross-thread store.

### Why not separate `Disk` backends per agent

An earlier design sketch proposed `FilesystemBackend` instances rooted at
`workspace/code/`, `workspace/test/`, `workspace/proof/`. **That is strictly weaker**, and we
rejected it:

- Even with `virtual_mode=True`, everything under the root is readable. The guarantee then
  rests on a path-confinement check rather than on the file's absence.
- With `virtual_mode=False` (the *default*, easy to omit) `..`, `~` and absolute paths all
  resolve, and the whole machine — including the other agents' directories, `.env`, and the
  pipeline's own source — is readable.
- It is a runtime check that can regress silently. Seeding is a structural property you can
  assert on directly.

`StateBackend` makes the guarantee **structural**: absence of the file, not denial of access.

### Why not subagents

`deepagents` subagents isolate *messages* and *skills* — a subagent's transcript does not
enter the parent's context. They do **not** isolate the filesystem: a subagent inherits the
parent's backend. Expressing Code/Test/Proof as subagents of one supervisor would quietly
merge all three pools into one filesystem.

So each agent is a **separate top-level invocation**, never a subagent of another. The
supervisor-with-subagents pattern from the deepagents quickstart is exactly the shape we must
not adopt.

### Also deliberately unused

- **`StoreBackend`** — persists across threads. Two runs of the pipeline would share a
  namespace, so a Test Agent could read a *previous* run's implementation. Cross-run leakage
  is still leakage.
- **Async subagents** — inherit the same filesystem as sync ones; concurrency adds no
  isolation and makes provenance harder to audit.
- **`skills=`** (the parameter) — see "Skills" below. We use the *idea* but not the parameter,
  because it resolves through the backend and is a no-op under `StateBackend`.
- **`execute` / shell tools** — fail closed under `StateBackend` (not a sandbox backend),
  and we keep it that way. A shell is a filesystem escape by definition.

## Layout

```
src/cleanroom/agents/deep/
├── runtime.py         shared: build_agent, seed_files, invoke_agent, virtual paths
├── planning.py        Stage 1 — the planner (no isolation requirement)
├── generation.py      Stage 2 — Code / Test / Proof (isolation-critical)
├── compile_repair.py  Java compile-repair driver (Stage 5b)
├── recovery.py        recovery step (b) — the ONE sanctioned isolation break
└── README.md          this file
```

Virtual roots are distinct per pool (`/code`, `/tests`, `/spec`, `/proof`) so that a stray
glob in one driver can never *name* a path belonging to another.

### What each agent's filesystem contains

| Generator | Seeded with | Never seeded with |
|---|---|---|
| `deep_generate_code` | `/spec` contracts | `/tests`, `/proof` |
| `deep_generate_tests` | `/spec` contracts | `/code`, `/proof` |
| `deep_generate_dafny` | `/spec` contracts | `/code`, `/tests` |

`/spec` is the **one shared input**: the planner's contract is deliberately common read-only
ground, which is exactly what makes three independent derivations target one interface. The
sheets are byte-identical across the three — if they were briefed differently, a disagreement
between their artifacts would be an artifact of the briefing rather than evidence about
independent derivation.

Artifact paths are **not** pre-seeded as empty placeholders: `write_file` refuses to overwrite
an existing file, so a placeholder would block that route. Paths are given in the opening
message instead.

## Stage 1 — planning

[`planning.py`](planning.py) is a drop-in for `PlanningAgent._design_feature`: same inputs,
same `{fr_id: FRPlan}` return, so the caller's normalization, dedupe and docstring composition
are untouched.

The substantive change is **per-FR submission**. The deterministic planner makes one
`with_structured_output(FeaturePlan)` call per feature, so a single bad field fails the whole
feature's output. The agent submits FRs one at a time through `submit_fr_plan`, and a
rejection names the offending field so it can fix that one FR and resubmit. That matters much
more on the smaller open-weight models we now self-host than it did on a frontier API.

Validation goes beyond the schema: `_param_mismatch` rejects a plan whose `args_json` or
`example_inputs_json` disagree with the signature's parameters. Three agents bind to that
signature independently, so a name that appears in one place and not another becomes three
mutually inconsistent artifacts.

## Stage 2 — the three generators

[`generation.py`](generation.py). Each is a separate top-level agent with its own
`StateBackend`; identity fields (`fr_id`, `feature_id`, `path`, `mvc_layer`) always come from
the planner's contract, and the agent authors **content only** — the same split the
deterministic generators use.

The proof generator optionally takes a `verifier` callable, letting the agent check its own
Dafny and iterate. **This is not an isolation break:** the Dafny verifier is a proof checker
over the agent's own text, not the oracle the pipeline scores against. The test suite stays
outside every agent — that distinction is the whole reason recovery has to be a separate,
separately-reported phase. A verifier that raises degrades to no-verification rather than
killing the run.

## Skills

`deepagents` skills package domain guidance as documents the agent reads **on demand** rather
than carrying in every prompt. We use the idea. We cannot use the `skills=` parameter.

### Why `skills=` does not work here

Measured against deepagents 0.6.12, not assumed:

1. Skills are resolved **through the backend**. `create_deep_agent` lists the skill source
   path using the same backend the file tools use.
2. Under `StateBackend` that path does not exist, so **no skill is discovered and the
   parameter silently does nothing** — no error, no metadata in the system prompt.
3. With a `FilesystemBackend` it works: the system prompt gains
   `**probe-skill**: <description>` and `-> Read '/probe-skill/SKILL.md' for full instructions`,
   and the agent loads the body with `read_file`.

So `skills=` requires a disk-backed backend — precisely the thing that breaks isolation. The
good news from the same experiment: passing `skills=` alongside `StateBackend` does **not**
open a disk door. `/etc/passwd`, the repo source and the skill file itself were all
unreachable; the agent still saw only its seeded pool.

### What we do instead

[`load_skills()`](runtime.py) seeds the documents into the agent's own virtual filesystem
under `/skills`, and the system prompt carries a one-line catalogue of them. This preserves
the property that matters — the agent reads a document only when it decides it needs it —
with no disk access at all.

| Agent | Skills seeded |
|---|---|
| Code | `clean-room-implementation` |
| Test | `blackbox-testing` |
| Proof | `dafny-proofs`, `dafny-patterns` |

### This fixed a real regression

The deterministic `DafnyAgent` inlines `dafny-patterns.md` and `dafny-proofs.md` into **every**
system prompt (`_build_system`). The deep proof driver built its own prompt and so had been
running with **none of that tuned Dafny guidance** — a silent quality regression on the stage
that needs help most, since Dafny is the lowest-resource language in the pipeline. Seeding
them restores it, and progressively: ~4.5 KB of guidance is now read on demand rather than
occupying the prompt on every call, which matters on a 32k context.

### The isolation risk skills DO carry

A skill is an instruction channel into every agent, so a document that quoted generated code
or test cases would leak across pools **while every filesystem assertion still passed**.
`test_skills_never_mention_another_pool_s_artifacts` guards this: skill documents must stay
spec-level. Keep them about *method*, never about a particular run's artifacts.

## The sanctioned break: recovery

The paper permits exactly one controlled isolation break: after a feature fails both proof
and the clean-room first pass, its code is regenerated **with the failing test cases
visible**. [`recovery.py`](recovery.py) implements this as a separate, explicitly-invoked
phase that runs only after the isolated pass completes, and it is reported separately in the
metrics — it is never the default path.

Two properties keep it honest:

- **The agent gets no test-execution tool.** The frozen suite is re-run by the *outer*
  recovery loop, so the agent cannot iterate against the oracle it is scored on. Keeping the
  verifier outside the agent is what stops recovery from degenerating into fitting the suite.
- **Tests are never regenerated.** The oracle must not move.

## Verification

[`tests/test_isolation.py`](../../../../tests/test_isolation.py) — 9 tests, fully offline
(scripted fake model), so they need no GPU, endpoint, or API key.

Each isolation assertion is paired with a **positive control**. Without one, a broken harness
that could read nothing at all would look like perfect isolation:

| Test | Asserts |
|---|---|
| `test_agent_can_read_its_own_pool` | control — reads genuinely work |
| `test_agent_can_list_its_own_pool` | control — enumeration genuinely works |
| `test_test_agent_cannot_read_code_agent_output` | **the core claim** |
| `test_code_agent_cannot_read_test_agent_output` | converse — no oracle leak |
| `test_proof_agent_is_isolated_from_both` | proof pool independent |
| `test_escape_attempts_do_not_reach_the_real_disk` | `/etc/passwd`, `..`, absolute paths |
| `test_pools_do_not_leak_through_a_shared_agent_object` | no state carried between runs |
| `test_deep_drivers_never_use[...]` | static guard: no `FilesystemBackend`, `StoreBackend`, `root_dir`, `subagents` |
| `test_every_generator_seeds_only_its_own_pool[...]` | static guard: no generator references another pool's root |

The static guards protect the *mechanism*, not just today's behaviour. They parse the **AST**
rather than grepping text — these drivers' docstrings deliberately discuss `FilesystemBackend`
and `subagents` to explain why they are unused, and a substring guard would fire on that prose
while missing an aliased import.

`test_every_generator_seeds_only_its_own_pool` catches the likeliest real regression: seeding
the code agent with tests "just for context" while debugging and forgetting to remove it.

Assertions are scoped to **enumeration output** (`ls`/`glob`/`grep`) when checking that a
path is invisible. A `read_file` on a foreign path echoes that path back in its `not found`
error — that is the agent's own words, not a leak, and asserting naively on it produces a
false positive.

```bash
uv run pytest tests/test_isolation.py -v      # 15 passed
uv run pytest tests/ -q                       # 50 passed
```

[`tests/test_deep_generation.py`](../../../../tests/test_deep_generation.py) covers the Stage 1
and Stage 2 drivers' wiring separately: what they return, what they reject (invented FR ids,
invalid layers, malformed JSON, params that disagree with the signature), and that identity
always comes from the contract rather than the agent.

Enforced automatically by [`.github/workflows/isolation.yml`](../../../../.github/workflows/isolation.yml)
on every push and PR. A local pre-commit hook is also available:

```bash
git config core.hooksPath .githooks
```

## Migration status

| Stage | Component | State |
|---|---|---|
| — | `runtime.py` shared harness | ✅ done |
| — | Java compile-repair driver | ✅ done, with isolation tests |
| 3 | Recovery step (b) | ✅ done |
| — | Isolation verification + CI | ✅ done |
| 1 | Planning stage | ✅ done |
| 2 | Code / Test / Proof generation | ✅ done |
| — | Wiring into `run_pipeline` (`--gen-driver`) | ✅ done |
| — | Dependency stage | ⬜ left deterministic (see below) |

**Dependency analysis stays deterministic on purpose.** Its LLM use is one narrow call
inferring semantic FR→FR edges; the rest is regex reference resolution and a topological sort.
An agent loop would add cost and nondeterminism to a stage that is mostly not a model problem.

## Running it

```bash
uv run python run_pipeline.py data/srs/Human.xml --gen-driver deepagent --prove --certify
```

`--gen-driver` mirrors the existing `--repair-driver`:

| Flag | Stages affected |
|---|---|
| `--gen-driver {deterministic,deepagent}` | planning, code, test, proof |
| `--repair-driver {deterministic,deepagent}` | compile-repair, recovery-regen |

**`deterministic` stays the default.** Every recorded result was produced that way, so the
deep path is opt-in and both arms remain runnable for comparison — which is what the paper
needs. `gen_driver` is written into the run report next to `repair_driver`, so a run's arm is
recoverable from its artifacts.

Delegation happens *inside* each agent (`CodeAgent.generate`, `TestAgent.generate`,
`DafnyAgent.generate_feature`, `PlanningAgent._design_feature`) rather than at the pipeline
call sites, so all the surrounding logic — adapter features, `skip_feature_ids`, proof caching,
normalization — is shared by both arms and cannot drift between them.

The code agent runs **one agent per feature**, not per FR and not per run: FRs in a feature
share data shapes, so the agent can cross-read them and keep them consistent, while a
whole-IR invocation would exhaust the context window on a large SRS. Feature order still
follows the planner's dependency order.
