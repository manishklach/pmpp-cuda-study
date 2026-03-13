# 050 - Median Filter

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Median Filter**.

## PMPP Ideas To Focus On

- small-window sorting
- nonlinear filtering
- impulse-noise removal

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

- Try a 5x5 window later.
- Compare against Gaussian blur on noisy input.
