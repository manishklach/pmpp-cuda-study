# 020 - Matrix Transpose With Shared Memory

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Matrix Transpose With Shared Memory**.

## PMPP Ideas To Focus On

- shared memory tiles
- synchronization
- avoiding bank conflicts

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

- Change tile size to 16 or 32.
- Compare against the naive transpose.
