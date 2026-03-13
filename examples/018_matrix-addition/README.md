# 018 - Matrix Addition

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Matrix Addition**.

## PMPP Ideas To Focus On

- 2D launches
- flattened storage
- matrix validation

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

- Use rectangular matrices.
- Compare 1D vs 2D launches.
