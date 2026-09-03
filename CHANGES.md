# Migration to deep agents — change log

Branch `deep-agents-only`, 2026-09-02 → 09-03. Nine commits, 76 tests.

Two things happened here. The pipeline moved to deepagents-only generation and lost the
cot/mot dimension; and reviewing that move surfaced several defects, three of which it had
introduced (§5 prerequisites, §6 Dafny kernel, §7 test stack). All are recorded below,
including what went wrong.

**The pattern worth noting:** each of those three was information the deterministic prompt
carried that the deep prompt never had, because it never had to while the deterministic path
was the default. Removing that path removed the briefing with it, and the suite stayed green
every time. Any agent prompt reviewed from here should be diffed against what the deleted
template used to say, not just read on its own.

---

## 1. Deep agents became the only generation driver — `4fc0dd5`

`--gen-driver` selected between one structured LLM call per contract/feature and a deepagents
agent. The switch and the deterministic arm are gone; every generation stage is now an agent.

**Removed:** `RunConfig.gen_driver`, `uses_deep_generation()`, the `--gen-driver` flag, the
single-shot paths in Planning/Code/Test/Proof, and `DafnyAgent`'s fixed `max_rounds`
generate→verify→revise loop with both its system prompts (−232 lines in that file alone).
A stale `--gen-driver` now fails with an argparse error rather than being ignored.

**Tool calling became the contract.** An artifact counts only when it arrives through a
`submit_*` tool. The virtual filesystem is a scratchpad and is no longer harvested; a
free-text reply is never scraped for a code fence. The submit tools are where the `fr_id` is
checked, emptiness rejected and validation errors handed back — the old fallbacks routed
around all of it and let a run whose tool calling had silently failed still report artifacts.
Code and test raise `DeepGenerationIncomplete` naming what is missing. Two deliberate
exceptions, both with an existing degraded path: proof returns empty source (an unproved
feature is a recorded outcome, not a run failure), and planning leaves an unsubmitted FR
absent (`plan()` substitutes a default contract plus a note).

**Test guard fixed.** The isolation and driver tests used to `skip` when `deepagents` was
missing. Correct while the drivers were optional; now that they are the only path and
`deepagents` is a hard dependency, skipping on a failed import would turn a pipeline that
cannot generate anything into a green suite — and retire the evidence for the clean-room
claim while everything else passed. They fail loudly instead.

`overnight.sh` was passing `--gen-driver`, which would have aborted every run in a batch.

## 2. Deleted the templates that arm owned — `e6ec3bb`

`code_template()` / `test_template()` had no callers left. Removed from all four language
targets along with the 24 templates they named (12 `generate_code*`, 9 `generate_tests*`,
3 `plan_feature*`) — −1,903 lines. `run_baseline.py` renders a separate
`generate_code_naive_*.j2` family, so the baseline arm was unaffected.

## 3. Removed the cot/mot prompting strategies — `9ad6f37`

Every `_mot` template belonged to Planning, Code or Test, so none survived §2 and `mot`
already resolved to `_cot` everywhere — the flag no longer varied generation at all.

**Removed:** the 7 remaining `*_cot.j2`, `cot_template()` from the prompt renderer,
`prompt_strategy` from `RunConfig`, the CLI, `run_pipeline`'s threading and all eight agents,
and the two runner scripts (`run_cot_experiment.py`, `run_mot_matrix_parallel.py`), which
existed only to drive these arms and would now die on an unrecognised flag.

**Kept:** `compare_three_way.py`, `emit_cot_results.py`, `emit_mot_results.py`. They only walk
archived run JSON, so the recorded CoT/MoT results stay readable even though the arms cannot
be re-run. README RQ2 is marked retired rather than describing commands that no longer exist.

## 4. Full-toolset exploratory arm — `1fcfceb`, `1313074`

`--full-toolset`: the four generation stages as deepagents with the complete built-in toolset
(`write_todos`, filesystem tools, shell `execute`) on one shared `LocalShellBackend`, plus
per-agent tool-call counting — new information this arm exists to gather.

Built by **substitution** (~150 lines) rather than forking `run_pipeline` (~1200): it swaps
four seams and lets the rest run untouched, so every existing metric is computed identically
and the arms are genuinely comparable, and the arm cannot drift out of sync with the pipeline.

Lives in `src/cleanroom/experiments/` deliberately. `tests/test_isolation.py` fails the build
if `FilesystemBackend`, `root_dir` or `subagents` appears in any `agents/deep/*.py`, and
`LocalShellBackend` needs `root_dir` — keeping the arm additive means that guard goes on
protecting the clean-room drivers, both arms stay runnable, and the default is unchanged.

**This arm has no isolation, by design:** agents share a filesystem, so the test agent can read
generated code. The caveat ships *inside the run report*, because the artifacts are otherwise
indistinguishable from a clean-room run. The proof cache is namespaced so a cached clean-room
proof cannot be served into it.

`virtual_mode=True` is set explicitly — left `False`, deepagents documents that absolute paths
and `..` bypass `root_dir`. It does **not** sandbox `execute`, which runs on the host as the
invoking user.

`1313074` then pulled Spec and Dependency back out. Both are deterministic-first — the SRS
parse is pure XML reading, dependency resolution is regex plus a topological sort — wrapped
around one narrow interpretive call each. An agent loop there adds cost and nondeterminism
without a plausible payoff, and leaving them out means the arm varies only the four stages
where the toolset could matter.

## 5. Prompt fixes across the four deep agents — `ea8300b`

- **Prerequisite signatures restored to the briefing.** The single-call code generator passed
  `prereq_ifaces()` into its prompt; the deep path dropped it, so an FR calling something
  another *feature* creates had only the planner's docstring note and had to guess that
  function's parameters and return shape. `contract_sheet` now carries the signatures — never
  bodies, from the planner's contracts, so the input stays spec-derived. Added to **all three**
  generators together: the sheet is deliberately identical across Code/Test/Proof, and
  briefing one better would make an agreement between their artifacts partly an artifact of
  the briefing rather than evidence of independent derivation.
- **Planning's "exactly once" contradiction.** The prompt said to call `submit_fr_plan` exactly
  once while all three rejection paths say "resubmit" and the tool replaces on resubmit. A
  model reading it literally treats a rejection as terminal and drops the FR to a default
  contract — the exact failure per-FR submission exists to prevent.
- **`entity_identifier` now enforced.** The prompt stated it must be a key of
  `example_inputs_json`; nothing checked it, while every neighbouring rule was checked. The
  value flows into the code agent's prompt as though the planner had vouched for it.
- **Step budgets stated.** No prompt named its budget, though code and test now raise on
  exhaustion — agents were failed against a limit they were never told.
- **Accuracy drifts:** "error cases" → the real contract fields; planning's opening now lists
  exact sheet paths instead of a bare directory; the test agent is told `submit_test_case`
  *accumulates*, so it stops resending an accepted case and leaving a stale duplicate.

Adding `{max_steps}` broke the exploratory arm, which formats the same prompt strings and had
nothing rendering them in tests — a `KeyError` would have surfaced only mid-run. Fixed, with a
test that renders all four prompts for both arms.

## 6. Dafny kernel contract restored — `9bc0938`

**A regression introduced by §1.** That commit deleted `DafnyAgent._SYSTEM` and
`_feature_prompt` as dead code — nothing called them once `generate_feature` became deep-only.
Reachability was the wrong test: they were the *only* statement of the kernel contract, and
`PROOF_PROMPT` never carried it because it never had to while the deterministic path was the
default.

The proof agent was no longer told to `include "Replay.dfy"`, to define `<mod>Domain refines
Domain` and `<mod>Kernel refines Kernel`, or to provide `Model`/`Action`/`Inv`/`Init`/`Apply`/
`Normalize`, `InitSatisfiesInv` and `StepPreservesInv`. That is load-bearing: the packager
compiles `out/<mod>-py/` and `generate_adapter` imports `<mod>Domain` and calls the proved
`Normalize(Apply(state, action))`. A module that verifies without refining the kernel is
useless downstream. The structure survived only as a worked example in `dafny-patterns.md`,
which the agent reads only if it chooses to — seeding guidance is not stating a requirement.

**Second bug on the same path:** `deep_generate_dafny` invented `Feature_4_1` while
`DafnyAgent` wrote `F4_1.dfy` and recorded `F4_1` as the proved module, because the caller
never passed its name down. Silent, because `_generate_feature_deep` overrode whatever the
agent submitted. The name is now owned by the caller, and `submit_dafny` rejects both a
renamed module and a source with no `<mod>Domain refines Domain`.

## 7. Test agent made stack-aware again — `eca8f2f`

Same shape as §6, on the test side. `TestAgent.stack` is documented as shaping "plain function
calls (python) vs TestClient HTTP requests (fastapi) — and the failure oracle (ValueError vs
HTTPException)", but `generate()` only passed `language` down, and `deep_generate_tests` had no
`stack` parameter at all.

The deleted `generate_tests.j2` branched on stack throughout. What went with it, for FastAPI:
the HTTP endpoint per FR (derived from `file_path`, which the contract sheet does not carry);
`expected_json` of `{"raises": "HTTPException"}` — `TEST_PROMPT` said flatly that
`oracle="raises"` asserts a ValueError, and `runner.py` defaults to ValueError when
`expected_json` says nothing; the TestClient instructions for `test_source`; and the
`setup_json` guidance that the database starts EMPTY, so a case editing an entity must first
create it using the same entity identifier.

None of it survived in `blackbox-testing.md` either — that document is stack-agnostic and
mentions HTTP zero times. So on a FastAPI run the agent wrote direct-call ValueError tests
against a web app, and every such case fails a correct implementation.

---

## Two process lessons

**Reachability is not coverage.** The Dafny regression came from a dead-name scan: "does
anything call this?" rather than "does anything still *say* this?". The signal was present and
misread — the same commit noted the proof cache was "hashing prompts that no longer reach any
model" and treated it as a stale hash to rebase, when it was the symptom. When deleting the
last caller of a prompt, check whether the prompt was the sole carrier of a requirement.

**Nothing failed.** The full suite stayed green through both defects, because no test asserted
the proof prompt's content and the module mismatch was silent by construction. Both now have
tests.

---

## Still open

- **The FoodSaver full-toolset run.** Waiting on the Minsky vLLM tunnel; `localhost:8000` was
  unreachable. Launch with:
  `uv run python run_pipeline.py data/srs/foodsaver.xml --language python --full-toolset --prove --certify`
- **Re-baseline.** Every archived result came from arms that no longer exist, and
  `overnight.sh`'s two arms now differ only in repair driver.
- **`submit_test_case` accumulates** where every other submit tool replaces. An accepted case
  cannot be withdrawn, so a corrected duplicate leaves both in the oracle and the stale one
  fails against correct code. Mitigated in the prompt; the real fix is a tool signature change.
- **`/spec` is writable.** Prompts call the sheets read-only; nothing enforces it, so an agent
  can rewrite its own briefing and satisfy that instead, invisibly. `FilesystemPermission`
  works (verified) but does not cover the full-toolset arm's `execute`.
- **`results/new_feature_2026-08-26/{NOTES,README}.md`** show as deleted and are not in
  `results.tgz`.
- **Adapter and repair paths** still use `with_structured_output`, under the separate
  `--repair-driver` flag.
