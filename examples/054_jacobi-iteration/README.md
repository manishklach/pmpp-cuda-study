# 054 - Jacobi Iteration

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Jacobi Iteration**.

## PMPP Ideas To Focus On

- iterative solvers
- ping-pong buffers
- convergence checks

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

- Run more iterations.
- Measure residual error after each iteration.
