# CUDA C++ review rules

Claude should review CUDA C++ code for:

- memory ownership;
- const-correctness;
- host/device boundary clarity;
- kernel launch configuration;
- syncronization;
- bounds checks;
- grid-stride loops;
- shared memory use;
- reduction correctness;
- deterministic vs nondeterministic behavior;
- precision choices;
- error checking after CUDA calls;
- separation between mathematical logic and hardware-specific kernels.

Claude must not directly edit CUDA/C++ source files.

It may suggest changes as prose, pseudocode, or review comments.
