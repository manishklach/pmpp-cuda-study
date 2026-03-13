# 066 - Canny Pipeline Stages

- Track: `Image and Signal`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Walk through a simplified Canny-style pipeline with three kernels: blur, gradient magnitude, and thresholding.

## PMPP Ideas To Focus On

- kernel chaining
- intermediate buffers
- decomposing a larger algorithm into easy-to-test stages

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a teaching-oriented subset of Canny, not a full implementation with hysteresis and direction-aware suppression.
- The three-stage structure is the main PMPP idea here.
- A next step is integrating example 064 as a suppression stage and adding edge tracking.
