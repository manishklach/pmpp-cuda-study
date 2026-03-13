# 063 - Template Matching

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Compute a simple sum-of-squared-differences score map for a template over a small image and find the best match.

## PMPP Ideas To Focus On

- 2D output grids
- sliding window access patterns
- small-kernel correctness before optimization

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- The kernel writes one score per valid template location.
- This version favors readability over shared-memory tiling.
- You can extend it with normalized cross-correlation or texture-backed reads later.
