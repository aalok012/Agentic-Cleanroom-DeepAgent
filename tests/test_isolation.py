"""The clean-room isolation guarantee, asserted structurally.

The paper's central claim is that the Code, Test and Proof agents derive their artifacts
independently: no agent may read another's output. Under deepagents that guarantee is NOT
enforced by the tool list — ``create_deep_agent`` always installs its built-in filesystem
tools and ``tools=`` is additive — so it has to come from the backend.

These tests drive real agents with a scripted model and assert on the tool output the agent
actually received. Each test pairs a POSITIVE control (an agent can read its own pool, so the
tooling works and the test is not passing vacuously) with the isolation assertion (it cannot
read another pool). Without the positive control a broken harness would look like perfect
isolation.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from src.cleanroom.agents.deep import runtime as deep
from tests.conftest import script

# NOT skipped when deepagents is missing — see the note in test_deep_generation.py. It matters
# most here: these tests ARE the clean-room guarantee, and a silent skip would retire the
# evidence for it while every other test still passed.
def test_deepagents_is_installed():
    """Guard: the isolation assertions below are vacuous without a real agent to run."""
    assert deep.deepagents_available(), (
        "deepagents is not importable, so the isolation guarantee is UNVERIFIED. "
        "It is a required dependency: `uv sync`.")

CODE_CONTENT = "def add(a, b):\n    return a + b  # CODE_POOL_MARKER\n"
TEST_CONTENT = "def test_add():\n    assert add(1, 2) == 3  # TEST_POOL_MARKER\n"
PROOF_CONTENT = "lemma AddCommutes(a: int, b: int) // PROOF_POOL_MARKER\n"

CODE_PATH = deep.virtual_path(deep.CODE_ROOT, 0, "add.py")
TEST_PATH = deep.virtual_path(deep.TEST_ROOT, 0, "test_add.py")
PROOF_ROOT = "/proof"
PROOF_PATH = deep.virtual_path(PROOF_ROOT, 0, "Add.dfy")


# Tools that ENUMERATE the filesystem. Their output is the interesting surface: a read of a
# foreign path echoes that path back in its "not found" error, which is the agent's own words
# rather than a leak, so path-absence is only meaningful for these.
LISTING_TOOLS = {"ls", "glob", "grep"}


def _tool_output(state, only: set[str] | None = None) -> str:
    """Tool results the agent saw, concatenated — what the model could condition on.

    ``only`` restricts to particular tools (see :data:`LISTING_TOOLS`)."""
    return "\n".join(
        str(m.content) for m in state["messages"]
        if isinstance(m, ToolMessage) and (only is None or getattr(m, "name", None) in only))


def probe(*turns):
    """``script`` with a padded tail.

    The agent may take more turns than the probe scripts (a rejected tool call costs an
    extra round trip), and an exhausted ``ScriptedModel`` raises StopIteration mid-graph.
    Trailing replies are only consumed if the agent actually asks for them.
    """
    return script(*turns, *["done"] * 8)


def _run(model, seeded: dict[str, str], fake_llm) -> tuple[str, str]:
    """Run one agent over exactly ``seeded``.

    Returns ``(all_tool_output, listing_output)``."""
    fake_llm(model)
    agent = deep.build_agent([], "You are under test.", name="probe")
    state = deep.invoke_agent(agent, "probe the filesystem", deep.seed_files(seeded))
    return _tool_output(state), _tool_output(state, only=LISTING_TOOLS)


# --- positive controls: the harness genuinely reads files ---------------------------
def test_agent_can_read_its_own_pool(fake_llm):
    """Control: an agent seeded with a file CAN read it. If this fails the isolation
    assertions below are meaningless, because everything would look unreadable."""
    out, listed = _run(
        probe(("read_file", {"file_path": CODE_PATH}), "read it"),
        {CODE_PATH: CODE_CONTENT},
        fake_llm,
    )
    assert "CODE_POOL_MARKER" in out, "harness cannot read even a seeded file"


def test_agent_can_list_its_own_pool(fake_llm):
    out, listed = _run(
        probe(("ls", {"path": "/"}), ("glob", {"pattern": "**/*"}), "listed"),
        {CODE_PATH: CODE_CONTENT},
        fake_llm,
    )
    assert CODE_PATH in out


# --- the isolation guarantee --------------------------------------------------------
def test_test_agent_cannot_read_code_agent_output(fake_llm):
    """THE core assertion: the Test Agent may not see the Code Agent's implementation.

    The two agents are separate top-level invocations with separate StateBackends, so the
    code file is not merely hidden from the test agent — it does not exist in its filesystem.
    """
    out, listed = _run(
        probe(
            ("read_file", {"file_path": CODE_PATH}),
            ("ls", {"path": "/"}),
            ("glob", {"pattern": "**/*"}),
            ("grep", {"pattern": "CODE_POOL_MARKER"}),
            "could not find it",
        ),
        {TEST_PATH: TEST_CONTENT},          # test pool only
        fake_llm,
    )
    assert "CODE_POOL_MARKER" not in out, "implementation leaked into the Test Agent"
    assert "def add" not in out
    assert "not found" in out, "the foreign read should have been denied"
    assert CODE_PATH not in listed, "the code path was enumerable from the Test Agent"
    assert "No matches found" in out, "grep for the code marker should have found nothing"
    assert TEST_PATH in listed, "test agent lost its own pool (harness broken)"


def test_code_agent_cannot_read_test_agent_output(fake_llm):
    """The converse: the Code Agent must not see the oracle it is scored against."""
    out, listed = _run(
        probe(
            ("read_file", {"file_path": TEST_PATH}),
            ("glob", {"pattern": "**/*"}),
            ("grep", {"pattern": "TEST_POOL_MARKER"}),
            "could not find it",
        ),
        {CODE_PATH: CODE_CONTENT},
        fake_llm,
    )
    assert "TEST_POOL_MARKER" not in out, "test oracle leaked into the Code Agent"
    assert "assert add" not in out
    assert "not found" in out, "the foreign read should have been denied"
    assert TEST_PATH not in listed, "the test path was enumerable from the Code Agent"


def test_proof_agent_is_isolated_from_both(fake_llm):
    """The Proof Agent derives Dafny from the contract alone — not from code or tests."""
    out, listed = _run(
        probe(
            ("read_file", {"file_path": CODE_PATH}),
            ("read_file", {"file_path": TEST_PATH}),
            ("glob", {"pattern": "**/*"}),
            "could not find them",
        ),
        {PROOF_PATH: PROOF_CONTENT},
        fake_llm,
    )
    assert "CODE_POOL_MARKER" not in out
    assert "TEST_POOL_MARKER" not in out
    assert CODE_PATH not in listed and TEST_PATH not in listed
    assert PROOF_PATH in listed, "proof agent lost its own pool (harness broken)"


def test_escape_attempts_do_not_reach_the_real_disk(fake_llm):
    """Path traversal and absolute real paths must not resolve.

    StateBackend has no disk under it at all, so this is structural rather than a filter —
    but the pipeline's own source tree is the obvious thing an escaping agent would find,
    and a regression to FilesystemBackend would surface it here.
    """
    out, listed = _run(
        probe(
            ("read_file", {"file_path": "/etc/passwd"}),
            ("read_file", {"file_path": "../../src/cleanroom/agents/code/agent.py"}),
            ("read_file", {"file_path": "/code/../tests/00_test_add.py"}),
            ("glob", {"pattern": "/**/*.py"}),
            "no escape",
        ),
        {CODE_PATH: CODE_CONTENT},
        fake_llm,
    )
    assert "root:x:" not in out, "escaped to the real /etc/passwd"
    assert "TEST_POOL_MARKER" not in out, "traversal reached the test pool"
    assert "class CodeAgent" not in out, "escaped to the real source tree"


def test_pools_do_not_leak_through_a_shared_agent_object(fake_llm):
    """Reusing one agent object across pools must not carry files between runs.

    ``invoke_agent`` passes ``files`` fresh per call, so state does not accumulate. If a
    future refactor moved seeding into construction, this is what would catch it.
    """
    fake_llm(probe(("ls", {"path": "/"}), "first"))
    agent = deep.build_agent([], "You are under test.", name="probe")
    deep.invoke_agent(agent, "run one", deep.seed_files({CODE_PATH: CODE_CONTENT}))

    fake_llm(probe(("ls", {"path": "/"}), ("grep", {"pattern": "CODE_POOL_MARKER"}), "second"))
    state = deep.invoke_agent(agent, "run two", deep.seed_files({TEST_PATH: TEST_CONTENT}))
    out = _tool_output(state)

    assert "CODE_POOL_MARKER" not in out, "files carried over between invocations"
    assert CODE_PATH not in _tool_output(state, only=LISTING_TOOLS)


# --- guards against a regression in HOW isolation is achieved -----------------------
# These parse the AST rather than grepping text: the drivers' docstrings deliberately DISCUSS
# FilesystemBackend, StoreBackend and subagents to explain why they are not used, and a
# substring guard would fire on that prose while missing, say, an aliased import.
def _deep_sources():
    import pathlib

    return sorted(pathlib.Path("src/cleanroom/agents/deep").glob("*.py"))


def _code_identifiers(path):
    """Every identifier and keyword-argument name actually used in code (not in strings)."""
    import ast

    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[-1])
    return names


@pytest.mark.parametrize("banned, why", [
    ("FilesystemBackend", "exposes the real disk, where all three pools coexist"),
    ("StoreBackend", "persists across threads, so a later run could read an earlier one's"),
    ("root_dir", "only meaningful for a disk-backed backend"),
    ("subagents", "a subagent inherits the parent's backend, merging the pools"),
])
def test_deep_drivers_never_use(banned, why):
    """Fail the build if a refactor introduces a construct that would break isolation."""
    offenders = [p.name for p in _deep_sources() if banned in _code_identifiers(p)]
    assert not offenders, f"{banned} used in {offenders}: {why}"


def _function_names(module, func_name: str) -> set[str]:
    """Identifiers used in the body of one top-level function — code only, not comments."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    raise AssertionError(f"{func_name} not found — did it get renamed?")


@pytest.mark.parametrize("generator, forbidden", [
    ("deep_generate_code", ("TEST_ROOT", "PROOF_ROOT", "UI_ROOT")),
    ("deep_generate_tests", ("CODE_ROOT", "PROOF_ROOT", "UI_ROOT")),
    ("deep_generate_dafny", ("CODE_ROOT", "TEST_ROOT", "UI_ROOT")),
    # The UI calls the backend over HTTP, which makes it the generator most tempting to hand
    # the implementation to. It must not be: the endpoints come from route_for(file_path).
    ("deep_generate_frontend", ("CODE_ROOT", "TEST_ROOT", "PROOF_ROOT")),
])
def test_every_generator_seeds_only_its_own_pool(generator, forbidden):
    """Each generator may reference only its own artifact root plus the shared /spec.

    Catches the likeliest real regression: seeding the code agent with tests "just for
    context" while debugging and forgetting to remove it. The drivers' comments name the
    other roots deliberately, so this reads the AST rather than the text.
    """
    from src.cleanroom.agents.deep import generation

    used = _function_names(generation, generator)
    leaked = [root for root in forbidden if root in used]
    assert not leaked, f"{generator} referenced another pool's root: {leaked}"


# --- skills: guidance must not become a side channel --------------------------------
def test_skills_are_seeded_and_readable(fake_llm):
    """Each generator's authored guidance reaches its own filesystem under /skills."""
    import contextlib

    from src.cleanroom.agents.deep.generation import (
        DeepGenerationIncomplete,
        deep_generate_tests,
    )

    fake_llm(probe(("ls", {"path": "/skills"}),
                   ("read_file", {"file_path": "/skills/blackbox-testing.md"}), "read it"))
    from langchain_core.messages import ToolMessage as _TM  # noqa: PLC0415

    import src.cleanroom.agents.deep.generation as gen
    captured = {}
    original = gen.invoke_agent

    def capture(agent, prompt, files, **kw):
        state = original(agent, prompt, files, **kw)
        captured["out"] = "\n".join(
            str(m.content) for m in state["messages"] if isinstance(m, _TM))
        return state

    gen.invoke_agent = capture
    # This scripted agent only reads its skill and never submits, so the driver rightly reports
    # an incomplete suite. What is under test here is what the agent could READ, so the
    # completeness check is irrelevant to it — the captured tool output is the assertion.
    try:
        with contextlib.suppress(DeepGenerationIncomplete):
            deep_generate_tests({"features": []}, "1", [{
                "fr_id": "1.1", "feature_id": "1", "signature": "def f() -> None",
                "docstring": "d", "mvc_layer": "model"}])
    finally:
        gen.invoke_agent = original

    assert "Black-box Test Design" in captured["out"], "the skill body was not readable"


def test_skills_never_mention_another_pool_s_artifacts():
    """Authored guidance is a prompt channel into every agent, so it must stay spec-level.

    A skill that quoted generated code or test cases would leak across pools while every
    filesystem assertion still passed — the one way skills could break the guarantee.
    """
    import pathlib

    banned = ("CODE_POOL_MARKER", "TEST_POOL_MARKER", "PROOF_POOL_MARKER",
              "generated_code", "generated_tests")
    for directory in ("src/cleanroom/agents/deep/skills",
                      "src/cleanroom/agents/dafny/skills"):
        for doc in pathlib.Path(directory).glob("*.md"):
            text = doc.read_text()
            hits = [b for b in banned if b in text]
            assert not hits, f"{doc} references pipeline artifacts: {hits}"


def test_skills_do_not_widen_the_filesystem(fake_llm):
    """/skills is added; the foreign artifact pools still are not."""
    from src.cleanroom.agents.deep.runtime import load_skills

    seeded = {TEST_PATH: TEST_CONTENT, **load_skills(["blackbox-testing"])}
    out, listed = _run(
        probe(("glob", {"pattern": "**/*"}),
              ("read_file", {"file_path": CODE_PATH}), "looked"),
        seeded, fake_llm)

    assert "/skills/blackbox-testing.md" in listed
    assert CODE_PATH not in listed, "seeding skills exposed the code pool"
    assert "CODE_POOL_MARKER" not in out
