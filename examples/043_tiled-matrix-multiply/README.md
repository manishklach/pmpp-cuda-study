# 043 - Tiled Matrix Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Tiled Matrix Multiply**.

This is the foundational matrix multiplication optimization example in the repo. It is meant to teach thread mapping, memory access patterns, tiling, and shared memory usage.

## PMPP Ideas To Focus On

- shared-memory tiles
- data reuse
- block-level synchronization

## What You Should Learn Here

- How tiling reduces repeated global-memory traffic
- Why threads in a block cooperate through shared memory
- Where `__syncthreads()` matters for correctness

## Study Prompts

- Identify one load that becomes reusable because of shared-memory tiling.
- Explain what would break if synchronization were removed.
- Compare the memory-access idea here with `042_naive-matrix-multiply`.

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or stays within tolerance.
- Start with the included tiny matrices before scaling up.

## What To Modify Next

- Try tile sizes 8, 16, and 32.
- Compare numerical output with the naive kernel.
- Measure the effect of tile size on runtime.
