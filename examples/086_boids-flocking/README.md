# 086 - Boids Flocking

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Compute one boids-style steering update using alignment, cohesion, and separation on a tiny flock.

## PMPP Ideas To Focus On

- all-to-all local interaction rules
- combining several behavioral terms per agent
- updating positions and velocities from shared state

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is another example where O(n^2) clarity is useful before adding spatial partitioning.
- The three steering terms map nicely to separate accumulators in each thread.
- A next step is adding a grid-based neighbor search to reduce the interaction cost.
