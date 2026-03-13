# 087 - Mandelbrot Renderer

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Render a small Mandelbrot iteration-count image on the GPU.

## PMPP Ideas To Focus On

- mapping pixels to complex-plane coordinates
- independent escape-time evaluation
- image generation with one thread per pixel

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- Fractal renderers are nice examples because every pixel is independent.
- Integer iteration counts give an easy CPU/GPU validation target.
- A next step is adding color mapping or supersampling.
