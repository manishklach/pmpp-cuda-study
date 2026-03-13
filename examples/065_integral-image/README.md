# 065 - Integral Image

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Build a summed-area table on the GPU using a simple per-output formulation so the indexing stays easy to follow.

## PMPP Ideas To Focus On

- prefix-style accumulation in 2D
- inclusive region sums
- correctness-first staging before optimized scans

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This implementation is intentionally straightforward, not asymptotically optimal.
- The result is useful for fast box-filter and region-sum queries.
- A next PMPP step is converting the row and column passes into parallel scans.
