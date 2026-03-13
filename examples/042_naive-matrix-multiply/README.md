# 042 - Naive Matrix Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Naive Matrix Multiply**.

## PMPP Ideas To Focus On

- 2D output mapping
- global memory baseline
- correctness first

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

- Change matrix sizes.
- Compare against the tiled version.
