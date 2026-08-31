"""The Dafny proof track: generate VERIFIED Dafny per feature .

For each feature it casts the FRs into a Dafny ``Domain`` state machine (one ``Action`` per FR;
BehavioralContract -> Inv / precondition guards / postcondition lemmas) that refines our vendored
``Replay.dfy`` kernel, then runs the real Dafny verifier and feeds proof errors back for up to
``max_rounds`` rounds — the generate->verify->revise loop validated by the feasibility spikes
(needs a proof-capable configured model; weaker models struggle with map-comprehension proofs).

Prompt knowledge is OWNED here: a DAFNY_REF syntax cheat-sheet plus two Dafny guidance docs
(``skills/dafny-patterns.md`` and ``skills/dafny-proofs.md``) vendored in this repo.

Isolation: the agent derives everything ONLY from the spec (FR text + behavioral contracts) and
never reads tests or any generated test artifact — the same clean-room guarantee as the Python
Code Agent, just targeting Dafny.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.cleanroom.agents.dafny.schema.dafny import FeatureDafny, GeneratedDafny
from src.cleanroom.utils.dafny_verify import verify_dafny
from src.cleanroom.utils.llm_client import DAFNY_MODEL, get_llm


def _mod_name(feature_id: str) -> str:
    """Turn an SRS feature id into a valid Dafny module name / safe file stem.

    Some specs (e.g. gemini's foodsaver extraction) emit ids like ``3.1.10b/11``;
    a bare ``.replace(".", "_")`` leaves the slash, producing an uncreated subdir
    (``F3_1_10b/11.dfy``) and an invalid Dafny module name. Sanitize every
    non-identifier character to ``_``.
    """
    return "F" + re.sub(r"[^0-9A-Za-z]", "_", feature_id)


# Dafny constructs models routinely get wrong; priming these turned an infinite syntax loop
# into actual progress in the feasibility spikes.
DAFNY_REF = """=== DAFNY SYNTAX REFERENCE (use these EXACT forms; wrong forms cause "gets expected"/"closeparen expected") ===
- Record type: use a datatype, NOT a named tuple:  datatype Item = Item(name: string, price: int)
- Map type:        map<string, int>
- Map update:      m[k := v]                  // add or overwrite one entry
- Map removal:     m - {k}                    // remove a key (the RHS is a SET literal)
- Map membership:  k in m   /   k !in m
- Map lookup:      m[k]      (only valid when k in m)
- Map comprehension (filter/transform): map k | k in m && <predicate> :: m[k]
  e.g. keep non-empty entries:  map k | k in m && k != "" && m[k] != "" :: m[k]
- Sequence: s + [x] (append), s[1..] (tail), |s| (length), s[i] (index)
- match: each case on its own line; NO `requires` inside Apply.
- Quantifiers & comprehensions use `|` for the RANGE and `::` for the BODY/term — NEVER `::` twice:
    forall x | P(x) :: Q(x)        (WRONG: `forall x :: P :: Q`)
    exists x | P(x) :: Q(x)
    set x | x in S :: f(x)         (every bound var MUST appear in the range; MUST end with a term)
    map k | k in m :: f(k)
  Flatten nested sets with MULTIPLE bound vars: set o, d | o in orders && d in orders[o] :: d
- Functions are PURE: NO reassignment. `var x := e; <expr>` is a one-shot let-binding — you may
  NOT then write `x := e2;`. Use fresh names / nested `var` lets instead.
- Every `if ... then ...` in a function MUST have an `else` (functions are total).
- Empty literals: empty map is `map[]`, empty set is `{}`, empty seq is `[]` — NEVER `set {}` or `map []`."""

# Dafny authoring guidance, vendored in-repo (src/cleanroom/agents/dafny/skills/). Tuned reference
# text that achieved 6/10 — used instead of a hand-condensed paraphrase so the prompt isn't degraded
# by our edits. Wall-specific tactics live in `_targeted_hint`, fired only on the matching error.
_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def _load_skill(name: str) -> str:
    return (_SKILLS_DIR / f"{name}.md").read_text()


_SYSTEM = """You are a Dafny author. Produce ONE Dafny file that refines the abstract `Domain`
below into a verified state machine for a single feature. Output ONLY Dafny source — no markdown,
no prose, no code fences.

CRITICAL GOTCHA: a refining `Apply` function and the refining `StepPreservesInv` lemma must NOT
repeat `requires Inv(m)` — it is inherited from the abstract Domain and repeating it is an error.

{ref}

=== DAFNY PATTERNS (state-machine pattern + common mistakes) ===
{dafny_skill}

=== DAFNY PROOFS (proof-maximizing workflow) ===
{proofs_skill}

=== ABSTRACT DOMAIN INTERFACE (refine this; do not redefine it) ===
{domain}
"""

# CoT variant (--prompt-strategy cot): a LIGHT reasoning pass before the Dafny. Kept deliberately
# light — the reasoning is short and, critically, the Dafny MUST be emitted inside a single ```dafny
# code fence so `_strip_fences` keeps the reasoning OUT of the written .dfy file (prose in a .dfy
# would fail to verify). Everything else (the Domain interface, gotchas, skills) is unchanged.
_SYSTEM_COT = """You are a Dafny author. Produce ONE Dafny file that refines the abstract `Domain`
below into a verified state machine for a single feature.

THINK FIRST, BRIEFLY. Before writing the file, reason in 3-6 short lines about:
  1. MODEL — the concrete `type Model` that represents this feature's state.
  2. INVARIANT — the `Inv` that must always hold (and how `Normalize` repairs it).
  3. PROOF OBLIGATIONS — that `Init` satisfies `Inv`, that each `Apply` case preserves `Inv`
     (`StepPreservesInv`), and which postcondition lemma(s) back the contract postconditions.
Keep this reasoning short and high-level; do NOT write Dafny inside it.

THEN output the COMPLETE Dafny file inside a SINGLE ```dafny ... ``` code fence — nothing after
the closing fence. Only the fenced Dafny source is compiled; your reasoning above it is ignored.

CRITICAL GOTCHA: a refining `Apply` function and the refining `StepPreservesInv` lemma must NOT
repeat `requires Inv(m)` — it is inherited from the abstract Domain and repeating it is an error.

{ref}

=== DAFNY PATTERNS (state-machine pattern + common mistakes) ===
{dafny_skill}

=== DAFNY PROOFS (proof-maximizing workflow) ===
{proofs_skill}

=== ABSTRACT DOMAIN INTERFACE (refine this; do not redefine it) ===
{domain}
"""


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:dafny)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _targeted_hint(messages: list[dict]) -> str:
    """Map the concrete Dafny errors to a specific fix tactic (not just 'try again')."""
    blob = " ".join(m.get("message", "") for m in messages).lower()
    hints: list[str] = []
    if "postcondition could not be proved" in blob:
        hints.append(
            "A POSTCONDITION failed. If the function returns a map/set comprehension, bind it "
            "(`var result := ...;`) and add `assert forall k | k in result :: <predicate>;` before "
            "returning, so Z3 sees the property. Otherwise add the missing `assert`/helper lemma that "
            "proves the `ensures` on this return path.")
    if "precondition could not be proved" in blob:
        hints.append(
            "A PRECONDITION failed — almost always a map/seq lookup `m[k]`/`s[i]` without first "
            "establishing `k in m` / `0 <= i < |s|`. Add that `assert` (or a guard) immediately before "
            "the lookup.")
    if "rbrace expected" in blob or "expected" in blob:
        hints.append(
            "A SYNTAX/parse error. Do NOT use `let ... in` inside `ensures`/`requires`; inline the "
            "expression or define a `predicate`/`function` and call it. Each spec clause is ONE boolean "
            "expression.")
    return ("\n\nTARGETED FIX:\n- " + "\n- ".join(hints)) if hints else ""


class DafnyAgent:
    """Generate verified Dafny modules from spec contracts, one per feature."""

    def __init__(self, project_dir: Path | str, llm=None, model: str = DAFNY_MODEL,
                 max_rounds: int = 6, prompt_strategy: str = "baseline",
                 gen_driver: str = "deterministic") -> None:
        # 'deepagent' routes generate_feature() through agents/deep/generation.py, where the
        # agent calls the Dafny verifier itself instead of us running a fixed round loop.
        self.gen_driver = gen_driver
        self.project_dir = Path(project_dir)
        self.dafny_dir = self.project_dir / "dafny"
        self.model = model
        self.max_rounds = max_rounds
        # 'baseline' = original prompt; 'cot'/'mot' = a LIGHT reasoning block (model/invariant/proof
        # obligations) before the Dafny — kept light so it does not destabilize Dafny syntax. The
        # proof track has no structural decomposition of its own, so 'mot' reuses the CoT reasoning.
        self.prompt_strategy = prompt_strategy
        self.llm = llm or get_llm(model=model, temperature=0.0)
        self._system = self._build_system()
        # Per-feature proof cache: lets the (otherwise monolithic) proof tier survive a mid-run
        # crash/blip — already-proved features are reloaded instead of re-proved on the next run.
        # MUST live OUTSIDE project_dir: scaffold_dafny_project() rmtree's project_dir at the start
        # of every proof run, so the cache sits next to it (parent dir) to survive that wipe.
        self.cache_path = self.project_dir.parent / f"{self.project_dir.name}__proof_cache.json"

    def _feature_sig(self, ir: dict, feature_id: str, mod: str) -> str:
        """Stable hash of everything that determines a feature's proof input, so the cache is only
        reused when the spec/planning + prompt + round budget are unchanged."""
        blob = f"{self.model}|{self.max_rounds}|{self._system}|{self._feature_prompt(ir, feature_id, mod)}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text())
        except Exception:
            return {}

    def _save_cache(self, cache: dict) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cache, indent=2))
            tmp.replace(self.cache_path)
        except Exception:
            pass   # caching is a best-effort optimization; never fail the proof on a cache write

    def _build_system(self) -> str:
        domain = (self.dafny_dir / "Replay.dfy").read_text().split(
            "abstract module {:compile false} Kernel")[0]
        template = _SYSTEM_COT if self.prompt_strategy in ("cot", "mot") else _SYSTEM
        return template.format(
            ref=DAFNY_REF,
            dafny_skill=_load_skill("dafny-patterns"),
            proofs_skill=_load_skill("dafny-proofs"),
            domain=domain,
        )

    def _feature_prompt(self, ir: dict, feature_id: str, mod: str) -> str:
        feat = next((f for f in ir.get("features", []) if f.get("id") == feature_id), {})
        contracts = [c for c in (ir.get("planning") or {}).get("contracts", [])
                     if c.get("feature_id") == feature_id]

        clines = []
        for c in contracts:
            bc = c.get("contract") or {}
            clines.append(
                f"- FR {c['fr_id']}: stimulus={bc.get('stimulus','')!r}; "
                f"precondition={bc.get('precondition','')!r}; response={bc.get('response','')!r}; "
                f"postcondition={bc.get('postcondition','')!r}")

        return (
            f"Feature {feature_id}: {feat.get('name','')}\n\n"
            f"Functional requirements (each becomes an Action variant; back each postcondition with a "
            f"matching `ensures` lemma):\n" + "\n".join(clines) + "\n\n"
            'Write a file that: starts with `include "Replay.dfy"`; defines '
            f"`module {mod}Domain refines Domain {{ ... }}` with a concrete `type Model`, a "
            "`datatype Action` (ONE variant per FR), `ghost predicate Inv`, `function Init`, "
            "`function Apply` (one match case per Action — NO `requires`), `function Normalize` "
            "(repair to satisfy Inv), lemmas `InitSatisfiesInv` and `StepPreservesInv` (NO "
            "`requires`), and at least one domain-specific postcondition lemma with an `ensures` "
            f"matching a postcondition above; and defines `module {mod}Kernel refines Kernel "
            f"{{ import D = {mod}Domain }}`. Derive everything ONLY from the spec above. "
            + ("Reason briefly first (model / invariant / proof obligations), then output the "
               "complete Dafny file inside ONE ```dafny code fence."
               if self.prompt_strategy in ("cot", "mot") else "Output ONLY the Dafny source."))

    def generate_feature(self, ir: dict, feature_id: str) -> FeatureDafny:
        mod = _mod_name(feature_id)
        target = self.dafny_dir / f"{mod}.dfy"
        if getattr(self, "gen_driver", "deterministic") == "deepagent":
            return self._generate_feature_deep(ir, feature_id, mod, target)
        messages = [SystemMessage(self._system), HumanMessage(self._feature_prompt(ir, feature_id, mod))]

        last = FeatureDafny(feature_id=feature_id, module=mod, dafny_source="", verified=False)
        for rnd in range(1, self.max_rounds + 1):
            resp = self.llm.invoke(messages)
            code = _strip_fences(resp.content if isinstance(resp.content, str) else str(resp.content))
            target.write_text(code)
            res = verify_dafny(target)
            last = FeatureDafny(feature_id=feature_id, module=mod, dafny_source=code,
                                verified=res.ok, rounds=rnd, residual_errors=res.messages,
                                axioms=res.axioms)
            if res.ok:
                return last
            errs = "\n".join(f"  line {m['line']}: {m['message']}" for m in res.messages[:12])
            messages.append(resp)
            _out_instr = (
                "Reason briefly about what caused each error, then output the FULL corrected Dafny "
                "module inside ONE ```dafny code fence."
                if self.prompt_strategy in ("cot", "mot")
                else "Output the FULL corrected Dafny module (only Dafny source).")
            messages.append(HumanMessage(
                f"`dafny verify` failed with these errors:\n{errs}{_targeted_hint(res.messages)}\n\n"
                f"{_out_instr} Do not repeat "
                "`requires Inv(m)` on refining Apply/StepPreservesInv."))
        return last

    def _generate_feature_deep(self, ir: dict, feature_id: str, mod: str, target: Path) -> FeatureDafny:
        """deepagents counterpart to :meth:`generate_feature`.

        The agent gets a ``dafny_verify`` tool and decides when to iterate, instead of us
        running ``max_rounds`` fixed rounds. Handing it the verifier is NOT a clean-room
        break: Dafny checks the agent's own proof text against its own specification, whereas
        the test suite — the oracle the pipeline scores against — stays outside every agent.
        """
        from src.cleanroom.agents.deep.generation import deep_generate_dafny  # noqa: PLC0415

        contracts = [c for c in ((ir.get("planning") or {}).get("contracts") or [])
                     if str(c.get("feature_id")) == str(feature_id)]

        def verifier(source: str) -> tuple[bool, str]:
            """Write a draft and run the real Dafny verifier over it."""
            target.write_text(source)
            res = verify_dafny(target)
            return res.ok, "\n".join(res.messages or [])

        out, metrics = deep_generate_dafny(
            ir, feature_id, contracts, verifier=verifier, model=self.model)

        code = out.get("source") or ""
        if not code.strip():
            return FeatureDafny(feature_id=feature_id, module=mod, dafny_source="",
                                verified=False, rounds=metrics.get("verify_calls", 0),
                                residual_errors=[{
                                    "line": 0, "col": 0,
                                    "message": "the proof agent produced no Dafny source"}])
        # Re-verify the SUBMITTED source: the agent may have submitted something other than
        # the last text it verified, so the recorded verdict must come from this final check.
        target.write_text(code)
        res = verify_dafny(target)
        return FeatureDafny(feature_id=feature_id, module=mod, dafny_source=code,
                            verified=res.ok, rounds=metrics.get("verify_calls", 0) + 1,
                            residual_errors=res.messages)

    def generate(self, ir: dict) -> GeneratedDafny:
        feature_ids = sorted({c["feature_id"] for c in (ir.get("planning") or {}).get("contracts", [])})
        cache = self._load_cache()
        features: list[FeatureDafny] = []
        for fid in feature_ids:
            mod = _mod_name(fid)
            sig = self._feature_sig(ir, fid, mod)
            ent = cache.get(fid)
            if ent and ent.get("sig") == sig:
                # Cache hit: reuse the proven result and re-materialize its .dfy so the later
                # compile/translate step still finds every module on disk.
                fd = FeatureDafny.model_validate(ent["data"])
                try:
                    (self.dafny_dir / f"{mod}.dfy").write_text(fd.dafny_source)
                except Exception:
                    pass
                features.append(fd)
                continue
            fd = self.generate_feature(ir, fid)
            cache[fid] = {"sig": sig, "data": fd.model_dump()}
            self._save_cache(cache)   # incremental: persist after EACH feature so a crash keeps them
            features.append(fd)
        return GeneratedDafny(features=features)
