# 023 - Sum Reduction

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Sum Reduction**.

This is a core PMPP example for parallel computation of a single result from a large dataset. It is the best starting point for understanding reduction patterns.

## PMPP Ideas To Focus On

- shared-memory reduction
- block partials
- final host aggregation

## What You Should Learn Here

- How many threads cooperate to compute one answer
- Why reductions are structured in stages instead of one giant atomic update
- Where control divergence and synchronization can affect performance

## Study Prompts

- Identify where per-thread inputs become block-level partial sums.
- Explain why the active thread count shrinks each reduction round.
- Compare the structure here with `024_max-reduction` and `025_min-reduction`.

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference.
- These examples use intentionally small inputs so each pattern is easy to inspect first.

## What To Modify Next

- Measure different block sizes.
- Add a second GPU reduction pass.
- Try warp-level primitives after understanding the shared-memory version.
