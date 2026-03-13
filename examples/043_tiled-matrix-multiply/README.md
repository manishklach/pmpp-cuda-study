# 043 - Tiled Matrix Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Tiled Matrix Multiply**.

## PMPP Ideas To Focus On

- shared-memory tiles
- data reuse
- block-level synchronization

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

- Try tile sizes 8, 16, and 32.
- Compare numerical output with the naive kernel.
