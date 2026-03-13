# 042 - Naive Matrix Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Naive Matrix Multiply**.

This is the baseline matrix multiplication example. It is intentionally simple so you can understand thread mapping and memory access patterns before moving to tiling and shared memory.

## PMPP Ideas To Focus On

- 2D output mapping
- global memory baseline
- correctness first

## What You Should Learn Here

- How a 2D output matrix maps onto threads
- Why naive matrix multiply rereads input values from global memory many times
- Why a slow baseline is still important for understanding the optimized version

## Study Prompts

- Point to the exact lines where a thread chooses its output row and column.
- Count conceptually how many repeated global-memory reads happen for one output tile.
- Compare this structure with `043_tiled-matrix-multiply`.

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
- Try a 2D block configuration if you want a more image-like mapping.
