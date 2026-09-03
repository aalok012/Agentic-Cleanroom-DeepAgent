"""Frontend Agent.

Generates the browser UI for a packaged app: one self-contained HTML page per feature, wired
to the backend's real endpoints. Driven by a deepagents agent (``agents/deep/generation.py``),
one agent per feature, briefed with that feature's per-FR contract sheets.

==================  UNSCORED, BUT STILL CLEAN-ROOM  ==================
The UI is a DELIVERABLE, not evidence. Nothing certifies it — there is no black-box oracle for
a screen — so it is excluded from pass@k, the verification ratio and the test-case pass ratio,
and a feature whose page fails to generate degrades the deliverable without failing the run.

Isolation still holds, and for a reason worth stating: a UI would seem to need the backend, but
it does not. Every endpoint it calls is ``route_for(contract.file_path)`` — the same
deterministic function the packager uses to mount the router and the certification oracle uses
to call it. So the page binds to the contract exactly as the code, tests and proofs do, and the
agent never reads a generated file.

  * No public method takes a code- or test-related parameter.
  * It never imports the code or test agent, and never reads `generated_code`.
  * Its agent runs on a StateBackend seeded with contract sheets and its own /ui pool.
======================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.cleanroom.agents.planning.agent import PlanningAgent
from src.cleanroom.utils.ir import feature_id_of, normalize_ir_features


class FrontendAgent:
    def __init__(self, llm=None, stack: str = "fastapi") -> None:
        # `llm` is unused by generate() — the deep driver builds its own agent on get_llm() —
        # and is accepted only so callers can construct this like every other agent.
        self.llm = llm
        # Only the FastAPI stack currently serves a UI: the packager mounts app/static and the
        # routes are derivable from file_path. Other targets return no pages rather than
        # emitting a page that points at endpoints their packager never creates.
        self.stack = stack

    def generate(self, ir: dict) -> dict[str, str]:
        """``{feature_id: html}`` for every feature with contracts. Input is the spec."""
        from src.cleanroom.agents.deep.generation import deep_generate_frontend  # noqa: PLC0415

        if self.stack != "fastapi":
            return {}

        normalize_ir_features(ir)
        PlanningAgent.normalize_ir_planning(ir)

        by_feature: dict[str, list[dict]] = {}
        for c in (ir.get("planning") or {}).get("contracts", []) or []:
            by_feature.setdefault(str(c["feature_id"]), []).append(c)

        names = {str(feature_id_of(f)): f.get("name", "") for f in ir.get("features", []) or []}

        pages: dict[str, str] = {}
        for feature_id, contracts in by_feature.items():
            html, _metrics = deep_generate_frontend(
                ir, feature_id, names.get(feature_id, ""), contracts)
            if html.strip():
                pages[feature_id] = html
        return pages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.frontend.agent <enriched_ir.json>")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as fh:
        enriched = json.load(fh)
    for fid, page in FrontendAgent().generate(enriched).items():
        print(f"  feature {fid}: {len(page.splitlines())} lines")
