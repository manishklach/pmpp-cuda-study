# 081 - Lennard Jones Forces

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Compute pairwise Lennard-Jones forces on a tiny particle system and compare GPU results against a CPU reference.

## PMPP Ideas To Focus On

- all-pairs interactions
- nonlinear force laws
- numerical stability via distance clamping

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a close cousin of the N-body examples, but with a short-range molecular dynamics force law.
- Small deterministic particle sets make validation much easier.
- A next step is truncating the force with a cutoff radius or introducing cell lists.
