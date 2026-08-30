# Clean-room Implementation

How to implement a contract you cannot test against, because the test suite is written
independently and you will never see it. Read this before writing code.

## What you are optimising for

Not "passes the tests" — you cannot see them. **Satisfies the contract as written, in
general.** A test suite derived from the same contract by another agent will agree with you
exactly to the extent that you both read the specification the same way. So implement the
requirement, not a guess at what someone might check.

The practical consequence: never special-case the example inputs. `example_inputs_json` is
one sample of the requirement, not the requirement. Code that returns the documented value
for the documented input and nothing sensible otherwise will fail every case you did not see.

## Binding to the interface exactly

The signature is a contract three agents bind to independently:

* do not rename the function, or reorder/rename its parameters;
* do not change the return shape — if the contract says a dict with `status`, return exactly
  that key, not `state` or `result`;
* honour the error mode. `raise` means `raise ValueError(...)` on a precondition violation;
  `return` means return the documented error dict. Getting this backwards fails every
  failure-path case at once.

## Shared shapes across requirements

Requirements in one feature usually share a data shape (the entity they operate on). Before
you invent a field name, `read_file` the sibling files in your pool and reuse theirs. Two
files that disagree about whether the key is `id` or `identifier` will both look correct in
isolation and fail together.

Where the contract names an `entity_identifier`, that field is the lookup key. Store by it,
find by it, and keep its type stable.

## Self-containment

Each file must import everything it needs and be importable on its own — no prose, no
markdown fences, no references to modules that do not exist. A file that cannot be imported
scores zero regardless of how correct its logic is.

## Validate at the boundary

Check preconditions first, then do the work. Validation that happens halfway through leaves
partial state behind, which turns one failing requirement into several.
