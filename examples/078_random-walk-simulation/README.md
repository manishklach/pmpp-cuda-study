# 078 - Random Walk Simulation

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Simulate many independent 1D random walkers in parallel and collect their final positions.

## PMPP Ideas To Focus On

- independent trajectories
- simple random branching
- thread-private state

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- Each walker takes the same number of steps, which makes the work decomposition clean.
- The average final position should stay close to zero.
- A next step is storing intermediate paths or moving to 2D walkers.
