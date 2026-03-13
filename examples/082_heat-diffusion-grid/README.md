# 082 - Heat Diffusion Grid

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Apply one explicit heat-diffusion update to a 2D grid with fixed boundary handling.

## PMPP Ideas To Focus On

- 2D stencil updates
- Jacobi-style time stepping
- separating old and new state buffers

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is one of the canonical stencil-based GPU workloads.
- The example performs a single step to keep validation easy.
- A next step is looping over many iterations and visualizing the temperature field.
