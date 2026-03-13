# 031 - Gather

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Gather**.

## PMPP Ideas To Focus On

- indirect reads
- index arrays
- memory locality

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

- Try repeated indices.
- Compare sorted versus random index patterns.
