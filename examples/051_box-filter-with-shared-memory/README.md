# 051 - Box Filter With Shared Memory

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Box Filter With Shared Memory**.

## PMPP Ideas To Focus On

- tile loading
- halo regions
- shared-memory reuse

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

- Change tile dimensions.
- Compare with a global-memory box filter.
