"""Golden tests that run the REAL Dafny binary.

These exist because the pipeline spent 37 foodsaver features and four Human runs reporting
"unproved" for a reason that had nothing to do with the model: the verifier closure raised
TypeError on every FAILING verification, and the agent's tool turned that into "the verifier
could not be run ... continue without it". No unit test caught it because every existing test
used a stub verifier, and because a PASSING verification returns an empty message list that
joins fine -- the fault was invisible on success and fatal on failure.

Skipped when dafny is not installed.
"""
from pathlib import Path

import pytest

from src.cleanroom.utils.dafny_verify import dafny_available, verify_dafny

pytestmark = pytest.mark.skipif(not dafny_available(), reason="dafny binary not on PATH/$DAFNY")

KERNEL = Path("src/cleanroom/agents/dafny/kernel/Replay.dfy")

# Verifies against the real kernel. Multi-field datatype model, a two-constructor Action,
# a datatype update, and refining lemmas that do NOT repeat the inherited requires/ensures.
GOOD = '''include "Replay.dfy"

module GoldDomain refines Domain {
  datatype Model = Model(items: map<string, int>, active: bool)
  datatype Action = Add(key: string, n: int) | Close

  ghost predicate Inv(m: Model) { true }

  function Init(): Model { Model(map[], true) }

  function Apply(m: Model, a: Action): Model {
    match a
    case Add(k, n) => Model(m.items[k := n], m.active)
    case Close     => m.(active := false)
  }

  function Normalize(m: Model): Model { m }

  lemma InitSatisfiesInv() {}
  lemma StepPreservesInv(m: Model, a: Action) {}
}
'''

# Every construct here is a real mistake the model made, and each is what Dafny rejects.
BAD = '''include "Replay.dfy"

module BadDomain refines Domain {
  type Model = { active: bool }
}
'''


def _kernel_into(tmp_path: Path) -> None:
    src = next((p for p in [KERNEL, Path("src/cleanroom/agents/dafny/Replay.dfy")] if p.is_file()), None)
    if src is None:
        src = next(iter(Path("src").rglob("Replay.dfy")), None)
    if src is None:
        pytest.skip("Replay.dfy kernel not found in the repo")
    (tmp_path / "Replay.dfy").write_text(src.read_text())


def test_a_known_good_module_actually_verifies(tmp_path):
    """Ground truth. If this fails the kernel or the toolchain is broken, not the model."""
    _kernel_into(tmp_path)
    target = tmp_path / "Gold.dfy"
    target.write_text(GOOD)
    res = verify_dafny(target)
    assert res.ok, f"the golden module must verify; dafny said:\n{res.raw}"


def test_the_verifier_closure_survives_a_FAILING_module(tmp_path):
    """The regression. The closure must return the error as text, never raise."""
    from src.cleanroom.agents.dafny.agent import DafnyAgent

    _kernel_into(tmp_path)
    verifier = DafnyAgent.make_verifier(tmp_path / "Bad.dfy")

    ok, output = verifier(BAD)          # must not raise

    assert ok is False
    assert output.strip(), "a failing verification must carry text back to the agent"
    assert "SynonymTypeDecl" in output, f"the real diagnostic must survive, got:\n{output}"
    assert "^" in output, "Dafny's caret must survive -- it is what localizes the fault"


def test_the_closure_reports_success_for_the_good_module(tmp_path):
    """The other half: a passing verification must come back ok with no error text."""
    from src.cleanroom.agents.dafny.agent import DafnyAgent

    _kernel_into(tmp_path)
    ok, output = DafnyAgent.make_verifier(tmp_path / "Gold.dfy")(GOOD)
    assert ok is True, f"golden module failed through the closure:\n{output}"


@pytest.mark.parametrize("n_contracts", [1, 2, 3])
def test_the_seeded_skeleton_verifies_as_written(tmp_path, n_contracts):
    """The seeded skeleton is worthless -- worse, actively misleading -- unless it verifies.

    The agent is told "this passes dafny_verify as written". If that is false it starts from a
    broken file and we have made the problem harder, so prove it for real, at several sizes.
    """
    from src.cleanroom.agents.deep.generation import dafny_skeleton

    _kernel_into(tmp_path)
    contracts = [{"fr_id": f"3.{i}.REQ-1"} for i in range(1, n_contracts + 1)]
    target = tmp_path / "Skel.dfy"
    target.write_text(dafny_skeleton("Skel", contracts))

    res = verify_dafny(target)
    assert res.ok, f"the seeded skeleton must verify; dafny said:\n{res.raw}"


def test_the_syntax_reference_lost_in_the_migration_is_back():
    """ea40d29 dropped DAFNY_REF when the proof track moved onto the deep agent.

    The two lemmafit skill docs it was replaced with cover proof STRATEGY, not Dafny SYNTAX,
    and every failure measured since is an item in this reference. Pin the entries that map to
    an observed failure so they cannot be dropped again.
    """
    from src.cleanroom.agents.deep.generation import DAFNY_REF, PROOF_PROMPT

    for needle, seen_as in [
        ("datatype", "type Model = {a: bool} / named tuples"),
        ("m[k := v]", "m with [k := v]  (Elm/F# record update)"),
        ("map[]", "map<string,string>{}"),
        ("match", "| Ctor(..) =>  instead of  case Ctor(..) =>"),
    ]:
        assert needle in DAFNY_REF, f"DAFNY_REF must cover {seen_as!r}"

    assert "{dafny_ref}" in PROOF_PROMPT, "the reference must reach the agent's system prompt"


def test_targeted_hints_are_restored():
    """The other half of what ea40d29 dropped: error text -> a concrete fix tactic."""
    from src.cleanroom.agents.deep.generation import _targeted_hint_from_text

    assert "SYNTAX" in _targeted_hint_from_text("Error: rbrace expected")
    assert "PRECONDITION" in _targeted_hint_from_text("precondition could not be proved")
    assert "POSTCONDITION" in _targeted_hint_from_text("postcondition could not be proved")
    assert _targeted_hint_from_text("") == "", "no hint when nothing matches"
