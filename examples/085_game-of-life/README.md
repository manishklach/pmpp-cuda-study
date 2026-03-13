# 085 - Game Of Life

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Apply one Conway's Game of Life update on a small toroidal grid.

## PMPP Ideas To Focus On

- neighborhood population counts
- double-buffered cellular automata updates
- branch-heavy per-cell rules

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- Cellular automata are a great fit for one-thread-per-cell mappings.
- Toroidal wrapping keeps the example compact by avoiding edge-condition branches.
- A next step is visualizing multiple generations or experimenting with alternative rulesets.
