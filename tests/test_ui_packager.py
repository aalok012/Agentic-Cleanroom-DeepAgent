"""The manual test console: derived from contracts, and harmless when absent."""

from __future__ import annotations

import json
from pathlib import Path

from src.cleanroom.utils.ui_packager import _params, build_ui

CONTRACTS = {
    "features": [{"functional_requirements": [
        {"id": "1.1", "description": "The system shall create an event."}]}],
    "planning": {"contracts": [{
        "fr_id": "1.1", "feature_id": "1", "mvc_layer": "controller",
        "signature": "def create_event(title: str, capacity: int) -> dict",
        "docstring": "Create an event.", "file_path": "controllers/create_event.py",
        "example_inputs_json": '{"title": "Talk", "capacity": 50}',
        "expected_return_json": '{"status": "ok"}', "error_mode": "raise",
        "failure_inputs_json": '{"title": ""}'}]},
}


def test_signature_parsing_handles_generics_and_defaults():
    """A comma inside dict[str, int] must not split a parameter in two."""
    assert _params("def f(a: int, b: str) -> X") == ["a", "b"]
    assert _params("def f(m: dict[str, int], xs: list[tuple[int, int]]) -> None") == ["m", "xs"]
    assert _params("def f(self, a: int = 3) -> None") == ["a"]
    assert _params("def f() -> None") == []


def test_console_is_built_from_the_contract(tmp_path: Path):
    page = build_ui(CONTRACTS, tmp_path)
    assert page is not None and page.exists()
    text = page.read_text()

    # the route must match what the packager actually mounts
    assert "POST /controllers/create_event" in text
    assert "The system shall create an event." in text
    assert "load failure case" in text          # the documented failure case is offered


def test_no_contracts_writes_nothing(tmp_path: Path):
    assert build_ui({"planning": {"contracts": []}}, tmp_path) is None
    assert not (tmp_path / "static").exists()


def test_console_never_contains_generated_code(tmp_path: Path):
    """The console is spec-derived. Generated code must not leak into it even when the IR
    carries it, so it stays a debugging aid rather than a channel."""
    ir = dict(CONTRACTS, generated_code={"files": [{
        "fr_id": "1.1", "feature_id": "1", "path": "controllers/create_event.py",
        "mvc_layer": "controller", "content": "SECRET_IMPLEMENTATION_MARKER = 1"}]})
    text = build_ui(ir, tmp_path).read_text()
    assert "SECRET_IMPLEMENTATION_MARKER" not in text


def test_malformed_example_json_still_renders(tmp_path: Path):
    """A planner that emits unparseable example inputs must not break the console."""
    ir = json.loads(json.dumps(CONTRACTS))
    ir["planning"]["contracts"][0]["example_inputs_json"] = "{not json"
    page = build_ui(ir, tmp_path)
    assert page is not None and "not json" in page.read_text()
