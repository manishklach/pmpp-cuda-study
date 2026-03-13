# 088 - Julia Renderer

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Render a small Julia-set iteration-count image for a fixed complex constant.

## PMPP Ideas To Focus On

- independent per-pixel complex iteration
- parameterized fractal kernels
- comparing closely related rendering formulas

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This pairs naturally with the Mandelbrot example because the loop body is almost the same but the parameter mapping differs.
- The CPU/GPU comparison again uses deterministic iteration counts.
- A next step is exposing the complex constant as a runtime input.
