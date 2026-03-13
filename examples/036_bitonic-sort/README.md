# 036 - Bitonic Sort

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Bitonic Sort**.

## PMPP Ideas To Focus On

- sorting networks
- structured compare-exchange
- power-of-two inputs

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

- Sort descending instead of ascending.
- Use random seeds for stronger tests.
