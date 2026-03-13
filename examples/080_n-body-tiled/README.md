# 080 - N Body Tiled

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Reuse particle positions through shared-memory tiling to reduce global memory traffic in the N-body force computation.

## PMPP Ideas To Focus On

- shared-memory tiles
- all-pairs interaction reuse
- performance-minded evolution of example 079

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- The correctness target is the same acceleration field as the naive version.
- This is one of the classic PMPP shared-memory examples.
- A next step is adding velocity integration and measuring speedup over the naive kernel.
