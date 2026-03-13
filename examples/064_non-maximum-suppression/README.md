# 064 - Non Maximum Suppression

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Suppress non-peak responses in a small score map by keeping only values that are greater than or equal to their 8-neighborhood.

## PMPP Ideas To Focus On

- neighborhood-based decisions
- branching around image boundaries
- post-processing after score generation

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a simplified local-maxima suppression pass rather than the full orientation-aware Canny variant.
- The CPU reference makes it easy to inspect which pixels should survive.
- A next step is to chain this after template matching or edge detection.
