"""Dependency Analyzer Agent.

Produces a NESTED dependency graph, purely deterministically (no LLM unless the optional
semantic pass is enabled):

  OUTER (feature level): a DAG over features plus a build order, detected from explicit
  cross-references in functional-requirement text (e.g. "see section 2.2"). Only FEATURES
  THAT HAVE FUNCTIONAL REQUIREMENTS are nodes.

  INNER (FR level): within each feature, a DAG over its FUNCTIONAL requirements, detected
  the same way (an FR whose text references another requirement in the same feature). Each
  feature carries its own `fr_order`, `fr_edges`, and `fr_cycles`.

The pipeline is FR-only: every requirement reaching this agent is a functional requirement.

Not every cross-reference is a dependency. A pointer like "more information is found in
section 2.5" tells the reader where to look; it is NOT a build prerequisite. Such
informational references are filtered out (see INFORMATIONAL_CUES) so they don't create
spurious edges — and, in turn, spurious cycles. This filtering is applied to BOTH the
regex path and the text handed to the optional semantic LLM pass, so the two stay consistent.

Edge direction convention (easy to get backwards):
    edge.source DEPENDS ON edge.target  =>  edge.target must be built BEFORE edge.source.
"""

import json
import re
import sys
import time
from collections import defaultdict

from src.cleanroom.agents.dependency.schema.graph import (
    DependencyEdge,
    DependencyGraph,
    FeatureFRDeps,
)
from src.cleanroom.utils.ir import feature_id_of, normalize_ir_features, requirement_text
from src.cleanroom.utils.prompt_renderer import PromptRenderer

# Matches explicit cross-references like "section 2.2.2" or "Section 2.5".
SECTION_REF = re.compile(r"section\s+(\d+(?:\.\d+)*)", re.IGNORECASE)

# Phrases that, when they immediately precede a "section N" reference, mark it as
# INFORMATIONAL ("for more detail, see ...") rather than a build prerequisite.
INFORMATIONAL_CUES: tuple[str, ...] = (
    "more information",
    "for more",
    "found in",
    "for details",
    "for further",
    "further detail",
    "see also",
    "additional information",
    "refer to the",
    "described in",
    "reference",
)
# How many characters before the reference to inspect for an informational cue.
_CUE_WINDOW = 45


def _is_informational(description: str, ref_start: int) -> bool:
    """True if the reference at `ref_start` is a 'see X for more' style pointer."""
    window = description[max(0, ref_start - _CUE_WINDOW):ref_start].lower()
    return any(cue in window for cue in INFORMATIONAL_CUES)


def _strip_informational_refs(text: str) -> str:
    """Remove 'section X' references that are informational pointers, so the text handed to
    the semantic LLM pass does not invite prerequisite edges the regex path would drop.

    Only the informational 'section X' tokens are removed; the surrounding prose stays, so
    the LLM still sees the requirement's actual behavior.
    """
    out, last = [], 0
    for match in SECTION_REF.finditer(text):
        if _is_informational(text, match.start()):
            out.append(text[last:match.start()])
            last = match.end()
    out.append(text[last:])
    return "".join(out)


def _req_text(req: dict) -> str:
    """Requirement text — delegates to canonical IR helper."""
    return requirement_text(req)


def _norm_id(raw: str) -> str:
    """Normalize an id the LLM echoed back (e.g. '[2.2.6.6]') to the bare parser id."""
    return (raw or "").strip().strip("[]").strip()


def feature_of(req_id: str) -> str:
    """Feature id is the first two dot-segments of a requirement id: '2.2.6.6' -> '2.2'."""
    return ".".join(req_id.split(".")[:2])


def _numeric_key(id_str: str) -> tuple[tuple[int, int, str], ...]:
    """Sort key so '2.2.6.10' comes after '2.2.6.9', not before.

    Non-numeric segments (e.g. 'FR-1') fall back to string comparison so the order
    stays deterministic for any spec.
    """
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in id_str.split("."))


class DependencyAnalyzer:
    def __init__(self, llm=None) -> None:
        # The OUTER graph and the regex INNER edges are always deterministic. `llm` is
        # optional: when provided, the inner level ALSO runs a semantic pass that infers
        # FR->FR prerequisites the text implies but never states as a "section X" ref
        # (e.g. "store a result and link it to the test request" depends on "create a
        # test request"). Left None, the agent is fully deterministic.
        self.llm = llm
        self.renderer = PromptRenderer() if llm is not None else None

    # ------------------------------------------------------------------
    # OUTER: feature-level graph
    # ------------------------------------------------------------------
    def analyze(self, ir: dict) -> DependencyGraph:
        # (a) Nodes and edge sources both come from FUNCTIONAL requirements (FR-only pipeline).
        fr_features: set[str] = set()
        scan: list[tuple[str, str, str]] = []  # (req_id, feature_id_of_source, text)
        for feature in ir.get("features", []):
            for req in feature.get("functional_requirements", []):
                fr_features.add(feature_of(req["id"]))
                scan.append((req["id"], feature_of(req["id"]), _req_text(req)))

        nodes = sorted(fr_features, key=_numeric_key)
        known = set(nodes)

        # (b) Scan text for explicit section references and build feature-level edges.
        edges: list[DependencyEdge] = []
        seen: set[tuple[str, str]] = set()
        for req_id, source_feature, text in scan:
            if source_feature not in known:
                continue  # not a buildable node, so not an edge source
            for match in SECTION_REF.finditer(text):
                target_feature = feature_of(match.group(1))
                if target_feature == source_feature:
                    continue  # self-reference within the same feature
                if target_feature not in known:
                    continue  # reference to a section that isn't a feature node
                if _is_informational(text, match.start()):
                    continue  # "more info found in section X" — a pointer, not a dependency
                if (source_feature, target_feature) in seen:
                    continue  # de-duplicate
                seen.add((source_feature, target_feature))
                edges.append(
                    DependencyEdge(
                        source=source_feature,
                        target=target_feature,
                        reason=f"Requirement {req_id} references section {match.group(1)}",
                    )
                )

        build_order, cycles = self._topo_sort(nodes, edges)
        return DependencyGraph(nodes=nodes, edges=edges, build_order=build_order, cycles=cycles)

    # ------------------------------------------------------------------
    # INNER: FR-level graph within one feature
    # ------------------------------------------------------------------
    def analyze_feature(self, feature: dict) -> tuple[list[str], list[DependencyEdge], list[list[str]]]:
        """Build the dependency graph over a single feature's FUNCTIONAL requirements.

        Returns (fr_order, fr_edges, fr_cycles). A reference resolves to another FR in the
        SAME feature. A "section X" reference is matched to the MOST SPECIFIC requirement
        id present (an exact id if it exists, otherwise the nearest enclosing one), instead
        of fanning out to every requirement nested under X. References to the feature root
        or to other features are left to the outer graph and ignored here.
        """
        fr_ids = [r["id"] for r in feature.get("functional_requirements", [])]
        nodes = sorted(fr_ids, key=_numeric_key)
        within = set(fr_ids)
        # The feature id, so we can ignore "section <feature>" whole-feature self-references.
        feature_id = feature_id_of(feature) or (feature_of(fr_ids[0]) if fr_ids else "")

        edges: list[DependencyEdge] = []
        seen: set[tuple[str, str]] = set()
        for req in feature.get("functional_requirements", []):
            source, text = req["id"], _req_text(req)
            for match in SECTION_REF.finditer(text):
                ref = match.group(1)
                if ref == feature_id:
                    continue  # reference to the whole feature, not a specific FR dependency
                if _is_informational(text, match.start()):
                    continue
                target = self._resolve_ref(ref, within, source)
                if target is None or (source, target) in seen:
                    continue
                seen.add((source, target))
                edges.append(
                    DependencyEdge(
                        source=source,
                        target=target,
                        reason=f"Requirement {source} references section {ref}",
                    )
                )

        # Semantic pass (only when an LLM is configured): infer prerequisites the prose
        # implies but never writes as a "section X" reference. Merged with the regex edges.
        if self.llm is not None:
            for source, target in self._infer_semantic_edges(feature, within):
                if source == target or (source, target) in seen:
                    continue
                seen.add((source, target))
                edges.append(
                    DependencyEdge(source=source, target=target, reason="Inferred prerequisite (semantic)")
                )

        build_order, cycles = self._topo_sort(nodes, edges)
        return build_order, edges, cycles

    @staticmethod
    def _resolve_ref(ref: str, within: set[str], source: str) -> str | None:
        """Resolve a 'section X' reference to ONE requirement id in this feature.

        Prefers an exact id match; otherwise the most specific (longest-id) requirement
        enclosing the reference. Returns None if nothing matches or it would self-link.
        This avoids one reference fanning out into an edge per nested sub-requirement.
        """
        if ref in within and ref != source:
            return ref
        candidates = [t for t in within if t.startswith(ref + ".") and t != source]
        if not candidates:
            return None
        # most specific = the id with the fewest extra segments beyond the reference
        return min(candidates, key=lambda t: (len(t.split(".")), _numeric_key(t)))

    def _infer_semantic_edges(self, feature: dict, within: set[str]) -> list[tuple[str, str]]:
        """One structured LLM call per feature. Returns validated (source, target) pairs;
        any id the parser did not produce for THIS feature is dropped (never invented).

        Informational 'section X' pointers are stripped from the text first, so the LLM
        sees the same filtered view the regex path uses and cannot resurrect a dependency
        the deterministic path deliberately drops.
        """
        frs = feature.get("functional_requirements", [])
        if len(frs) < 2:
            return []
        requirements = [{"id": r["id"], "text": _strip_informational_refs(_req_text(r))} for r in frs]
        prompt = self.renderer.render(
            "infer_fr_deps.j2",
            {"feature_name": feature.get("name", feature_id_of(feature)), "requirements": requirements},
        )
        result: FeatureFRDeps | None = self.llm.with_structured_output(FeatureFRDeps).invoke(prompt)
        # Function-calling structured output returns None when the model emits no tool call
        # (e.g. an open model replying "no dependencies" in prose rather than calling the tool).
        # Semantically that means "no inferred edges" for this feature — treat it as empty.
        if result is None:
            return []

        pairs: list[tuple[str, str]] = []
        for dep in result.dependencies:
            source = _norm_id(dep.id)
            if source not in within:
                continue  # invented / cross-feature source — drop
            for raw_target in dep.prerequisite_ids:
                target = _norm_id(raw_target)
                if target in within and target != source:
                    pairs.append((source, target))
        return pairs

    # ------------------------------------------------------------------
    # handoff
    # ------------------------------------------------------------------
    def enrich(self, ir: dict, output_dir=None) -> dict:
        """Fold BOTH levels into the IR (without mutating the input):
          - outer graph under 'dependency_graph'
          - inner graph on each feature as 'fr_order' / 'fr_edges' / 'fr_cycles'
        Optionally writes the result to <output_dir>/<project>_dependency.json.
        """
        normalize_ir_features(ir)
        graph = self.analyze(ir)

        new_features: list[dict] = []
        for feature in ir.get("features", []):
            order, fr_edges, fr_cycles = self.analyze_feature(feature)
            new_features.append(
                {
                    **feature,
                    "fr_order": order,
                    "fr_edges": [e.model_dump() for e in fr_edges],
                    "fr_cycles": fr_cycles,
                }
            )

        enriched = {**ir, "features": new_features, "dependency_graph": graph.model_dump()}

        if output_dir is not None:
            from pathlib import Path

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{ir.get('project_name', 'project')}_dependency.json").write_text(
                json.dumps(enriched, indent=2)
            )
        return enriched

    # ------------------------------------------------------------------
    @staticmethod
    def _topo_sort(
        nodes: list[str], edges: list[DependencyEdge]
    ) -> tuple[list[str], list[list[str]]]:
        """Kahn's algorithm. Used for BOTH the outer (feature) and inner (FR) graphs.

        Since source depends on target, the build graph runs target -> source: a node is
        ready when all its targets (prerequisites) are already placed. Nodes never reached
        are part of a cycle and reported (grouped by connected component) rather than crashed.
        """
        prereqs_left: dict[str, int] = {node: 0 for node in nodes}
        dependents: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            prereqs_left[edge.source] += 1
            dependents[edge.target].append(edge.source)

        ordered: list[str] = []
        ready = sorted((n for n in nodes if prereqs_left[n] == 0), key=_numeric_key)
        while ready:
            node = ready.pop(0)
            ordered.append(node)
            for dependent in dependents[node]:
                prereqs_left[dependent] -= 1
                if prereqs_left[dependent] == 0:
                    ready.append(dependent)
            ready.sort(key=_numeric_key)

        leftover = set(nodes) - set(ordered)
        cycles: list[list[str]] = []
        neighbors: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            if edge.source in leftover and edge.target in leftover:
                neighbors[edge.source].add(edge.target)
                neighbors[edge.target].add(edge.source)
        while leftover:
            stack = [min(leftover, key=_numeric_key)]
            component: set[str] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(neighbors[node] - component)
            leftover -= component
            cycles.append(sorted(component, key=_numeric_key))

        return ordered, cycles


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.dependency.agent <spec_ir.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        spec_ir = json.load(f)

    analyzer = DependencyAnalyzer()
    graph = analyzer.analyze(spec_ir)

    print(f"OUTER — features ({len(graph.nodes)}): {', '.join(graph.nodes)}")
    for edge in graph.edges:
        print(f"  {edge.source} depends on {edge.target}  ({edge.reason})")
    print(f"  build order: {' -> '.join(graph.build_order)}")
    print(f"  cycles: {graph.cycles or 'none'}\n")

    print("INNER — FR order per feature:")
    for feature in spec_ir.get("features", []):
        order, fr_edges, fr_cycles = analyzer.analyze_feature(feature)
        if order:
            print(f"  [{feature.get('id', '?')}] {' -> '.join(order)}"
                  + (f"   cycles={fr_cycles}" if fr_cycles else ""))