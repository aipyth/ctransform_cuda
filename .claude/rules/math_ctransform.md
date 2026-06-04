# c-transform mathematical rules

When analyzing algorithms, check:

1. Problem definition
    - mathematical definition of the c-transform;
    - source domain;
    - target domain;
    - support representation;
    - cost function c(x, y);
    - discrete, semi-discrete, or continuous formulation.

2. Correctness invariants
    - correct minimization or maximization convention;
    - consistency with the mathematical definition;
    - expected monotonicity or envelope properties.

3. Numerical stability
    - overflow and underflow risks;
    - precision loss in float32;
    - cancellation effects;
    - handling of large costs;
    - handling of nearly equal minima;
    - stable reduction procedures.

4. CUDA-specific numerical risks
    - nondeterministic reductions;
    - unsafe atomics;
    - race conditions;
    - warp divergence;
    - uncoalesced memory access;
    - shared-memory bank conflicts;
    - reduction correctness.

Claude should propose mathematical checks and test cases before implementation changes.
