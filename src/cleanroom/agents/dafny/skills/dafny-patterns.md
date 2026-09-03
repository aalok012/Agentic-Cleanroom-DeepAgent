# Dafny Patterns

## When to Write Code in Dafny
- ALL `verifiable:true` entries in the spec MUST be written in Dafny (do not write verifiable code directly in JavaScript or TypeScript)

## Dafny Pattern Example

Given the `Replay` kernel, a simple counter app with inherited undo/redo could be written like this

```dafny
include "Replay.dfy"

module CounterDomain refines Domain {
  // The model is the state of your application
  type Model = int

  // Actions are the ways the state can change
  datatype Action = Inc | Dec

  // Invariant: what must always be true about the state
  predicate Inv(m: Model) {
    m >= 0
  }

  // Initial state
  function Init(): Model {
    0
  }

  // How actions transform the state
  function Apply(m: Model, a: Action): Model {
    match a
    case Inc => m + 1
    case Dec => m - 1
  }

  // Normalization: fix invalid states (called after Apply)
  function Normalize(m: Model): Model {
    if m < 0 then 0 else m
  }

  // Proof that Init satisfies the invariant
  lemma InitSatisfiesInv()
    ensures Inv(Init())
  {
  }

  // Proof that every step preserves the invariant
  lemma StepPreservesInv(m: Model, a: Action)
    // requires Inv(m) is inherited and should not be repeated
    ensures Inv(Normalize(Apply(m, a)))
  {
  }
}

module CounterKernel refines Kernel {
  import D = CounterDomain
}

module AppCore {
  import K = CounterKernel
  import D = CounterDomain

  function Init(): K.History { K.InitHistory() }

  function Inc(): D.Action { D.Inc }
  function Dec(): D.Action { D.Dec }

  function Dispatch(h: K.History, a: D.Action): K.History requires K.HistInv(h) { K.Do(h, a) }
  function Undo(h: K.History): K.History { K.Undo(h) }
  function Redo(h: K.History): K.History { K.Redo(h) }

  function Present(h: K.History): D.Model { h.present }
  function CanUndo(h: K.History): bool { |h.past| > 0 }
  function CanRedo(h: K.History): bool { |h.future| > 0 }
}
```


## The model usually has SEVERAL fields — use a `datatype`, not a type synonym

The counter above uses `type Model = int` because its whole state is one number. That is the
exception. A real feature's state has several fields, and then `Model` MUST be a `datatype`
with named fields. Dafny tuples are POSITIONAL — `(int, bool)` — so you cannot name their
components, and Dafny has no record-literal type at all.

```dafny
// WRONG — Dafny tuple components cannot be named. Parse error: "closeparen expected".
type Model = (details: map<string, string>, applicationActive: bool)

// WRONG — this is TypeScript, not Dafny. Parse error: "invalid SynonymTypeDecl".
type Model = { isLoggedIn: bool }

// RIGHT — a datatype with one constructor gives you named fields AND `m.field` access.
datatype Model = Model(details: map<string, string>, applicationActive: bool)
```

Both wrong forms fail at PARSE time, before Dafny checks a single proof obligation, so the
whole module scores zero however good the reasoning inside it is.

Here is the multi-field shape in full — it verifies (`3 verified, 0 errors`) against the
`Replay.dfy` kernel in this repo:

```dafny
include "Replay.dfy"

module EventDomain refines Domain {
  datatype Model = Model(
    details: map<string, string>,
    requirements: map<string, int>,
    applicationActive: bool)

  datatype Action =
    | UpdateDetails(newDetails: map<string, string>, newRequirements: map<string, int>)
    | CloseApplication

  ghost predicate Inv(m: Model) {
    m.applicationActive ==> |m.details| >= 0
  }

  function Init(): Model {
    Model(map[], map[], true)      // empty map is `map[]`; `map<string, string>{}` won't parse
  }

  function Apply(m: Model, a: Action): Model {
    match a
    case UpdateDetails(d, r) => Model(d, r, m.applicationActive)
    case CloseApplication    => m.(applicationActive := false)   // datatype update
  }

  function Normalize(m: Model): Model { m }

  // No `requires`/`ensures` here: both are inherited from the abstract Domain and
  // REPEATING THEM IS AN ERROR ("a refining method is not allowed to add preconditions").
  lemma InitSatisfiesInv() {}
  lemma StepPreservesInv(m: Model, a: Action) {}
}
```

Field access is `m.applicationActive` (with the dot). Copy-with-change is
`m.(applicationActive := false)`. Both need the `datatype` form above.

## Common Mistakes to Avoid

- It is an error to repeat inherited `requires` clauses (see the refinement note above).
- `Model` with more than one field must be a `datatype`, never `type Model = (a: T, b: U)`
  (named tuple, invalid) or `type Model = { a: T }` (TypeScript, invalid).
- The empty map literal is `map[]`. `map<string, string>{}` is a parse error ("invalid Ident").
- It is OK to have `assume {:axiom} false` in _proofs_, temporarily, as the pieces are put together. Strive for zero such axioms eventually.
- Nested pattern matching _is_ allowed, but needs to be properly parenthesized. Example (out of context):
```
function optimize(e: exp): exp
{
    match e
    case EInt(v) => e
    case EVar(x) => e
    case EAdd(e1, e2) => (match (optimize(e1), optimize(e2))
        case (EInt(0), e2) => e2
        case (e1, EInt(0)) => e1
        case (e1, e2) => EAdd(e1, e2))
}
```
