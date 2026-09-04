"""Drive the Dafny verifier synchronously.

This is the heart of the Dafny verification track: locate a ``dafny`` binary, run ``dafny verify``
on a ``.dfy`` file, and return a structured pass/fail + the proof errors (fed back to the agent's
revise loop) + any ``assume {:axiom}`` escape hatches (so an unprovable obligation can pass as an
explicit, auditable *assumption* rather than silently). The binary is found via ``$DAFNY`` or PATH.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# "Dafny program verifier finished with 10 verified, 0 errors"
_SUMMARY = re.compile(r"finished with (\d+) verified, (\d+) error", re.IGNORECASE)
# "path.dfy(30,16): Error: a refining function is not allowed to add preconditions"
_ERROR_LINE = re.compile(r"\((\d+),(\d+)\):\s*(?:Error|.*?error)\b[:]?\s*(.*)", re.IGNORECASE)

# `assume {:axiom} <expr>;` — an explicit, auditable escape hatch: the obligation is ASSUMED,
# not proved. We surface these so a feature can pass with documented assumptions.
_AXIOM = re.compile(r"assume\s*\{:axiom\}")


def diagnostics(res: "DafnyResult", limit: int = 4000) -> str:
    """The text to hand an agent after a failed verification.

    Prefer Dafny's RAW output over the parsed messages. Dafny prints the offending source line
    and a caret under the exact token:

        F3_2.dfy(43,17): Error: rbrace expected
           |
        43 |     requires not m.user_authenticated
           |                  ^

    The caret is the diagnostic. Dafny's parser cannot know that `not` should be `!`, so the
    message alone ("rbrace expected") localizes nothing, and an agent handed only "43:17:
    rbrace expected" has to reconstruct the file from memory to guess what is at column 17 —
    which is exactly how a proof agent gets stuck re-emitting the same error every round.
    """
    raw = (getattr(res, "raw", "") or "").strip()
    if raw and raw != "timeout":
        return raw[:limit]
    return format_messages(getattr(res, "messages", []) or "")[:limit]


def format_messages(messages: list[dict]) -> str:
    """Render DafnyResult.messages as text an agent can act on: ``line:col: message``.

    ``messages`` is a list of DICTS. Joining it directly raises TypeError, and because a
    PASSING verification returns an empty list, that mistake is invisible on success and fatal
    on every failure — which is exactly how it stayed hidden.
    """
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            line, col = m.get("line", 0), m.get("col", 0)
            text = str(m.get("message", "")).strip()
            out.append(f"{line}:{col}: {text}" if (line or col) else text)
        else:
            out.append(str(m))
    return "\n".join(out)


@dataclass
class DafnyResult:
    ok: bool
    verified: int = 0
    errors: int = 0
    messages: list[dict] = field(default_factory=list)   # [{line, col, message}]
    axioms: list[dict] = field(default_factory=list)      # [{line, content}] assume {:axiom}
    raw: str = ""


def dafny_binary() -> str | None:
    """Path to a runnable ``dafny`` launcher: ``$DAFNY`` or the first ``dafny`` on PATH."""
    env = os.environ.get("DAFNY")
    if env and Path(env).exists():
        return env
    return shutil.which("dafny")


def dafny_available() -> bool:
    return dafny_binary() is not None


def extract_axioms(dfy_path: Path | str) -> list[dict]:
    """Scan a .dfy file for ``assume {:axiom}`` lines — guarantees ASSUMED rather than proved."""
    axioms: list[dict] = []
    try:
        for i, line in enumerate(Path(dfy_path).read_text().splitlines(), start=1):
            if _AXIOM.search(line):
                axioms.append({"line": i, "content": line.strip()})
    except OSError:
        pass
    return axioms


def verify_dafny(dfy_path: Path | str, timeout: float = 180.0) -> DafnyResult:
    """Run ``dafny verify`` on a file and parse the verdict + proof errors.

    Never raises on a verification failure — a failed proof is data, returned in the result.
    Raises only if the Dafny binary cannot be found (a setup error, not a proof outcome).
    """
    binary = dafny_binary()
    if binary is None:
        raise RuntimeError("dafny binary not found — set $DAFNY, put `dafny` on PATH, or install it")

    try:
        proc = subprocess.run(
            # --allow-warnings: Dafny exits non-zero on STYLE warnings (e.g. ==> indentation)
            # even when verification fully succeeds; we judge by the verification summary, not
            # warnings, so they must not be treated as a failure.
            [binary, "verify", "--allow-warnings", str(dfy_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return DafnyResult(ok=False, errors=1,
                           messages=[{"line": 0, "col": 0, "message": "dafny verify timed out"}],
                           raw="timeout")

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary = _SUMMARY.search(out)
    verified = int(summary.group(1)) if summary else 0
    errors = int(summary.group(2)) if summary else 0

    messages: list[dict] = []
    for line in out.splitlines():
        m = _ERROR_LINE.search(line)
        if m:
            messages.append({"line": int(m.group(1)), "col": int(m.group(2)),
                             "message": m.group(3).strip()})

    # Judge by the verification summary, NOT the exit code: Dafny exits non-zero on mere style
    # warnings even when every proof obligation is discharged. Pass iff a clean summary with zero
    # verification errors and no error-location lines.
    ok = summary is not None and errors == 0 and not messages
    # No summary at all => a resolution/parse error (nothing was verified); capture it as a failure.
    if summary is None and not messages:
        messages.append({"line": 0, "col": 0, "message": out.strip()[:300] or "dafny produced no output"})
        ok = False
    return DafnyResult(ok=ok, verified=verified, errors=max(errors, len(messages)),
                       messages=messages, axioms=extract_axioms(dfy_path), raw=out)
