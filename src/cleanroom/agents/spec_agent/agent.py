"""Specification Generator Agent.

Stage 1 of the pipeline. Two phases:

  1. PARSE (deterministic, no LLM): parse the SRS via SRSReader, assign stable IDs, and
     group functional requirements into features. The LLM is intentionally kept OUT so
     IDs, text, and grouping can never drift, be dropped, or be hallucinated.

  2. CONTRACT SYNTHESIS (LLM): for each functional requirement, write a BEHAVIORAL
     CONTRACT — a design-by-contract specification (stimulus, precondition, response,
     postcondition). One structured call per feature (never the whole IR). The LLM authors
     only the contract fields; the requirement id always comes from the parser.

FR-ONLY: the pipeline processes functional requirements only. The SRSReader extracts
requirements ONLY from functional-requirement sections and ignores narrative prose
(introduction, scope) and non-functional sections (performance, safety, security, legal).
Everything that reaches this agent is treated as a functional requirement.
"""

import json
import sys
from pathlib import Path

from src.cleanroom.agents.spec_agent.schema.ir import (
    BehavioralContract,
    Feature,
    FeatureContractSet,
    FunctionalRequirement,
    IntermediateRepresentation,
)
from src.cleanroom.agents.spec_agent.tools.srs_reader import SRSReader
from src.cleanroom.utils.ir import feature_id_of, normalize_ir_features, requirement_text
from src.cleanroom.utils.llm_client import get_llm
from src.cleanroom.utils.prompt_renderer import PromptRenderer


def _norm_id(raw: str) -> str:
    """Normalize an id the LLM echoed back (e.g. '[2.2.1]') to the bare parser id."""
    return (raw or "").strip().strip("[]").strip()


class SpecAgent:
    def __init__(self, llm=None) -> None:
        self.reader = SRSReader()
        # `llm` is injectable so tests/stubs avoid network calls. Resolved lazily: the
        # deterministic parse in run() never touches it; only synthesize_contracts() does.
        self._llm = llm
        self.renderer = PromptRenderer()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def run(self, srs_path: Path, output_dir: Path | None = Path("outputs")) -> IntermediateRepresentation:
        """Parse the SRS deterministically into features of functional requirements. No LLM.

        Writes ``<stem>_ir.json`` only when ``output_dir`` is given. The full pipeline passes
        ``output_dir=None`` (the bare parse is a strict subset of the final full_ir.json, so the
        per-stage dump would be redundant); standalone callers/CLI still get the file."""
        feature_dicts = self.reader.read_features(srs_path)

        features = []
        for f in feature_dicts:
            reqs = f.get("functional_requirements", [])
            if not reqs:
                continue  # skip features with no functional requirements (prose/NFR-only sections)
            features.append(
                Feature(
                    id=f["id"],
                    name=f["name"],
                    description=f.get("description", ""),
                    functional_requirements=[
                        FunctionalRequirement(id=req["id"], text=requirement_text(req)) for req in reqs
                    ],
                )
            )

        ir = IntermediateRepresentation(
            project_name=srs_path.stem,
            source_file=srs_path.name,
            features=features,
            contracts=[],
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{srs_path.stem}_ir.json").write_text(ir.model_dump_json(indent=2))
        return ir

    def synthesize_contracts(self, ir: dict, output_dir: Path | None = None) -> dict:
        """Write a behavioral contract for every functional requirement (LLM).

        One structured call per feature. Returns the IR with a top-level 'contracts'
        list (one BehavioralContract per FR). Optionally writes '<project>_contracts.json'.
        """
        normalize_ir_features(ir)
        contracts: list[BehavioralContract] = []

        for feature in ir.get("features", []):
            frs = feature.get("functional_requirements", [])
            if not frs:
                continue
            feature_id = feature_id_of(feature)

            authored = self._contract_feature(feature.get("name", ""), feature_id, frs)

            for req in frs:
                rid = req["id"]
                fields = authored.get(rid)
                if fields is None:
                    contracts.append(
                        BehavioralContract(
                            fr_id=rid,
                            feature_id=feature_id,
                            stimulus="",
                            precondition="",
                            response="",
                            postcondition="",
                        )
                    )
                    continue
                contracts.append(
                    BehavioralContract(
                        fr_id=rid,
                        feature_id=feature_id,
                        stimulus=fields.stimulus,
                        precondition=fields.precondition,
                        response=fields.response,
                        postcondition=fields.postcondition,
                    )
                )

        enriched = {**ir, "contracts": [c.model_dump() for c in contracts]}
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{ir.get('project_name', 'project')}_contracts.json").write_text(
                json.dumps(enriched, indent=2)
            )
        return enriched

    # --- helpers ---------------------------------------------------------------
    def _contract_feature(self, feature_name: str, feature_id: str, frs: list[dict]) -> dict:
        """One structured LLM call for a single feature's FRs. Returns {fr_id: FRContract}."""
        requirements = [
            {"id": r["id"], "text": requirement_text(r)} for r in frs
        ]
        prompt = self.renderer.render(
            "write_contract.j2",
            {"feature_name": feature_name, "feature_id": feature_id, "requirements": requirements},
        )
        result: FeatureContractSet = self.llm.with_structured_output(FeatureContractSet).invoke(prompt)
        return {_norm_id(c.id): c for c in result.contracts}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.spec_agent.agent <srs.xml>")
        sys.exit(1)

    agent = SpecAgent()
    ir_obj = agent.run(Path(sys.argv[1]))
    enriched = agent.synthesize_contracts(ir_obj.model_dump(), output_dir=Path("outputs"))
    print(f"Project   : {enriched['project_name']}")
    print(f"Features  : {len(enriched['features'])}")
    print(f"Contracts : {len(enriched['contracts'])}\n")
    for c in enriched["contracts"]:
        print(f"  [{c['fr_id']}] stimulus={c['stimulus'][:60]!r}")
