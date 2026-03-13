# 027 - Prefix Sum Work Efficient Scan

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Prefix Sum Work Efficient Scan**.

## PMPP Ideas To Focus On

- Blelloch upsweep/downsweep
- shared memory
- work efficiency

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

- Turn the inclusive result into exclusive form.
- Tile across multiple blocks.
