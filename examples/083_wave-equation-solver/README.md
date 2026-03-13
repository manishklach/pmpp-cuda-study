# 083 - Wave Equation Solver

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Advance a 1D wave equation by one explicit time step using previous, current, and next state buffers.

## PMPP Ideas To Focus On

- multi-buffer time stepping
- second-order stencil updates
- fixed-end boundary conditions

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- The three-buffer pattern appears in many simulation codes.
- One spatial dimension keeps the indexing compact while still showing the core idea.
- A next step is expanding to 2D wave propagation and multiple iterations.
