# 038 - Parallel Binary Search Over Sorted Chunks

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Parallel Binary Search Over Sorted Chunks**.

## PMPP Ideas To Focus On

- batched queries
- independent searches
- chunk boundaries

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

- Use different chunk sizes.
- Return insertion positions for missing values.
