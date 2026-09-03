"""The Dafny proof track: generate VERIFIED Dafny per feature .

For each feature it casts the FRs into a Dafny ``Domain`` state machine (one ``Action`` per FR;
BehavioralContract -> Inv / precondition guards / postcondition lemmas) that refines our vendored
``Replay.dfy`` kernel, then verifies it. Generation is driven by a deepagents agent
(``agents/deep/generation.py``): it is handed a ``dafny_verify`` tool and decides for itself when
to iterate, replacing the old fixed ``max_rounds`` generate->verify->revise loop. This class keeps
the surrounding concerns — the on-disk .dfy layout, the per-feature proof cache, and the final
re-verification of whatever the agent submitted.

The agent's briefing lives with the agent: its system prompt is ``deep.generation.PROOF_PROMPT``
and its guidance documents are the two Dafny skills vendored in this repo
(``skills/dafny-patterns.md`` and ``skills/dafny-proofs.md``), seeded into its filesystem.

Isolation: the agent derives everything ONLY from the spec (FR text + behavioral contracts) and
never reads tests or any generated test artifact — the same clean-room guarantee as the Python
Code Agent, just targeting Dafny.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


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


class DafnyAgent:
    """Generate verified Dafny modules from spec contracts, one per feature."""

    def __init__(self, project_dir: Path | str, llm=None, model: str = DAFNY_MODEL,
                 max_rounds: int = 6) -> None:
        # generate_feature() always runs through agents/deep/generation.py, where the agent
        # calls the Dafny verifier itself and decides when to iterate, instead of us running a
        # fixed round loop.
        self.project_dir = Path(project_dir)
        self.dafny_dir = self.project_dir / "dafny"
        self.model = model
        # `max_rounds` is accepted for caller compatibility but no longer steers generation: the
        # agent owns its own briefing (deep.generation.PROOF_PROMPT) and its own iteration budget
        # (a LangGraph step cap, tuned with CLEANROOM_DEEP_MAX_STEPS).
        self.max_rounds = max_rounds
        self.llm = llm or get_llm(model=model, temperature=0.0)
        # Per-feature proof cache: lets the (otherwise monolithic) proof tier survive a mid-run
        # crash/blip — already-proved features are reloaded instead of re-proved on the next run.
        # MUST live OUTSIDE project_dir: scaffold_dafny_project() rmtree's project_dir at the start
        # of every proof run, so the cache sits next to it (parent dir) to survive that wipe.
        self.cache_path = self.project_dir.parent / f"{self.project_dir.name}__proof_cache.json"

    def _feature_sig(self, ir: dict, feature_id: str, mod: str) -> str:
        """Stable hash of everything that determines a feature's proof input, so the cache is only
        reused when the spec/planning + agent briefing are unchanged.

        The briefing now belongs to the proof AGENT, not to this class: its system prompt and the
        skill documents it is seeded with live in agents/deep, so they are hashed from there.
        Editing PROOF_PROMPT or a dafny skill therefore still invalidates every cached proof.
        """
        from src.cleanroom.agents.deep.generation import PROOF_PROMPT  # noqa: PLC0415
        from src.cleanroom.agents.deep.runtime import load_skills      # noqa: PLC0415

        contracts = [c for c in (ir.get("planning") or {}).get("contracts", [])
                     if c.get("feature_id") == feature_id]
        skills = "".join(v for _, v in sorted(load_skills(["dafny-proofs", "dafny-patterns"]).items()))
        blob = "|".join([
            self.model, mod, PROOF_PROMPT, skills,
            json.dumps(contracts, sort_keys=True, default=str),
        ])
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

    def _abstract_domain(self) -> str:
        """The abstract ``Domain`` from the vendored kernel, for the proof agent to refine.

        Everything up to the ``Kernel`` declaration — the same slice the removed
        ``_build_system`` inlined into every proof prompt. Missing/unreadable is not fatal:
        the prompt falls back to pointing at the worked example in the skill document.
        """
        try:
            return (self.dafny_dir / "Replay.dfy").read_text().split(
                "abstract module {:compile false} Kernel")[0]
        except OSError:
            return ""

    def generate_feature(self, ir: dict, feature_id: str) -> FeatureDafny:
        mod = _mod_name(feature_id)
        target = self.dafny_dir / f"{mod}.dfy"
        return self._generate_feature_deep(ir, feature_id, mod, target)

    def _generate_feature_deep(self, ir: dict, feature_id: str, mod: str, target: Path) -> FeatureDafny:
        """Generate and verify one feature's Dafny module with the proof agent.

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

        # Pass OUR module name and the abstract kernel the module must refine. Both used to
        # reach the model through the deterministic prompt this class no longer has; without
        # them the agent invents a name and may not refine Replay.dfy at all, and the adapter
        # that imports <mod>Domain has nothing to bind to.
        out, metrics = deep_generate_dafny(
            ir, feature_id, contracts, module=mod, domain=self._abstract_domain(),
            verifier=verifier, model=self.model)

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
