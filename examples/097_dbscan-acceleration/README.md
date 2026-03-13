# 097 - DBSCAN Acceleration

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Accelerate the expensive neighbor-count stage of DBSCAN by computing epsilon-neighborhood sizes and core-point flags on the GPU.

## PMPP Ideas To Focus On

- pairwise distance checks
- separating acceleration stages from full clustering logic
- core-point detection as a reusable primitive

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This focuses on the most GPU-friendly DBSCAN subproblem instead of the full cluster expansion.
- The output tells you which points are core points for a chosen epsilon and `minPts`.
- A next step is building cluster growth on top of these neighbor relationships.
