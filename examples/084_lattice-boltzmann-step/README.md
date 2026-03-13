# 084 - Lattice Boltzmann Step

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Run a simplified D2Q9 collide-and-stream step on a tiny periodic grid.

## PMPP Ideas To Focus On

- multiple populations per cell
- local collision plus neighbor streaming
- structured-grid data layout

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a teaching-friendly single-step version of a common fluid simulation pattern.
- The example keeps periodic boundaries to avoid extra boundary-condition code.
- A next step is adding macroscopic density and velocity reconstruction.
