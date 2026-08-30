# Black-box Test Design

How to derive a test suite from a specification you can verify against, without ever seeing
an implementation. Read this before writing cases.

## The rule that makes the suite evidence

Every case must be justifiable from the contract alone. If you cannot point at the sentence
in the requirement or the field in the contract that a case checks, the case does not belong
in the suite — it is a guess about an implementation you have not seen, and a passing guess
proves nothing.

Concretely, never assume:
* a private helper, attribute, or internal data structure the contract does not name;
* an ordering the contract does not state (dict/set iteration, list sort stability);
* an error *message* — the contract fixes the error *type*, not its wording;
* that state persists between cases unless the contract says the entity is persisted.

## Coverage checklist

For each requirement, work through these in order and stop when the contract runs out:

1. **Happy path** — the documented example inputs, the documented return. Always present.
2. **Documented failure** — whatever `failure_inputs_json` names, asserted with
   `oracle="raises"` when the error mode is `raise`, or as an error return otherwise.
3. **Boundaries** — for every numeric or length constraint in the contract, the value at the
   limit and the first value past it. A contract saying "at most 100" implies 100 and 101.
4. **Equivalence classes** — one representative per class the contract distinguishes
   (e.g. empty / single / many; present / absent). One good representative beats five
   near-duplicates.
5. **Identity round-trip** — if the contract names an `entity_identifier`, create it, then
   look it up by that field. This catches the commonest cross-FR shape disagreement.

## Choosing the oracle

* `oracle="eq"` — compare the returned value against `expected_json`. Prefer it.
* `oracle="raises"` — assert a `ValueError`. Use only where the contract's error mode is
  `raise` and the precondition is genuinely violated.

If a check cannot be expressed as either, it is probably an assertion about internals. Drop it.

## Setup state

`setup_json` is a list of prior calls needed to reach the state under test. Keep it minimal:
the shortest sequence that establishes the precondition. A long setup makes a failure hard to
attribute to the requirement being tested.

## Sizing

Aim for coverage of the contract, not a case count. Two precise cases that pin the contract
beat ten that all exercise the happy path with different literals. Every case you add is a
case a correct implementation must also satisfy — a wrong case is worse than a missing one,
because it fails a correct implementation.
