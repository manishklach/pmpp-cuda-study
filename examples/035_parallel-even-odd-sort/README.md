# 035 - Parallel Even Odd Sort

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Parallel Even Odd Sort**.

## PMPP Ideas To Focus On

- alternating compare-swap phases
- small-array sorting
- iterative kernel launches

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

- Increase the array size gradually.
- Compare with bitonic sort on the same data.
