# Documentation rules

Docs under `docs/` are **specification and explanation**, not a second copy of the source.

When writing or editing docs, describe the library at a conceptual level:

- architecture and layering;
- mathematics, derivations, and invariants;
- algorithms — via prose, math, index/thread mappings, and review checklists;
- public interfaces — function signatures, parameters, shapes, and short descriptions of
  what each function does;
- design decisions and the gotchas worth flagging to whoever implements.

Do **not** put full, copy-paste-ready implementation bodies in docs — complete kernel or
function bodies belong in the source, not the documentation. A function *signature* with
parameter descriptions is documentation; a filled-in `__global__` kernel body or host
wrapper body is implementation and should be left to the source files.

Rationale: implementation lives in exactly one place (the source), and docs stay a stable,
readable explanation of *what* the code does and *why*, rather than drifting into a
duplicate of *how* that has to be kept in sync line-by-line.

This complements [`cuda_cpp.md`](cuda_cpp.md): review and describe CUDA/C++ as prose,
pseudocode-level algorithm description, or review comments — not as finished source.
