# 026 - Prefix Sum Naive Scan

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Prefix Sum Naive Scan**.

## PMPP Ideas To Focus On

- Hillis-Steele scan
- iterative passes
- inclusive prefix behavior

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

- Convert to exclusive scan.
- Handle larger-than-one-block inputs.
