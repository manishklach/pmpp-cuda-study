# 016 - 1D Stencil

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **1D Stencil**.

## PMPP Ideas To Focus On

- neighbor access
- boundary conditions
- stencil decomposition

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or expected pattern.
- Start with the built-in small inputs before scaling up.

## What To Modify Next

- Expand to a 5-point stencil.
- Add a shared-memory version later.
