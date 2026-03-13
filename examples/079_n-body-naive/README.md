# 079 - N Body Naive

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Compute one force accumulation per particle using the straightforward O(n^2) all-pairs formulation.

## PMPP Ideas To Focus On

- all-pairs interactions
- softening for numerical stability
- mapping one particle update per thread

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is the clear baseline before applying shared-memory tiling.
- The example only computes one integration step on a tiny system.
- A next step is comparing memory traffic and reuse with example 080.
