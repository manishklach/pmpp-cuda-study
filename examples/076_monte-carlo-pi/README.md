# 076 - Monte Carlo Pi

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Estimate pi by throwing random points into the unit square and counting how many land inside the unit circle.

## PMPP Ideas To Focus On

- independent random trials
- reduction via per-thread hit counts
- stochastic validation

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a classic PMPP example for massively parallel Monte Carlo work.
- A lightweight LCG keeps the example self-contained.
- A next step is switching to CURAND or adding confidence-interval estimates.
