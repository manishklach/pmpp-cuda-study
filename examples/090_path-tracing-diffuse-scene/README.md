# 090 - Path Tracing Diffuse Scene

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Estimate simple diffuse radiance with one random bounce against a sphere and a sky contribution.

## PMPP Ideas To Focus On

- stochastic sampling per pixel
- per-thread random number state
- comparing noisy Monte Carlo output against the same CPU estimator

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This intentionally keeps the bounce count and scene complexity low so the estimator is still easy to inspect.
- The CPU reference uses the same deterministic RNG sequence as the GPU kernel.
- A next step is adding multiple samples per pixel or a second bounce.
