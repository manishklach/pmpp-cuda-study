# 089 - Ray Sphere Tracer

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Trace one primary ray per pixel against a sphere and output a simple Lambertian shading value.

## PMPP Ideas To Focus On

- one-thread-per-pixel ray generation
- geometric intersection tests
- image-space validation with a tiny deterministic scene

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a compact stepping stone toward path tracing.
- Keeping the scene to one sphere makes the math easy to inspect.
- A next step is adding a ground plane or shadows.
