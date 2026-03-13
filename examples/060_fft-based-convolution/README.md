# 060 - FFT Based Convolution

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `041-060`

## Goal

Build and study a library-aware study scaffold of **FFT Based Convolution**.

## PMPP Ideas To Focus On

- frequency-domain multiplication
- padding strategy
- FFT library workflow

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example is intentionally a stronger study scaffold because it typically depends on CUDA math libraries not available in this authoring environment.
- Use the README plus code comments as a roadmap for the eventual implementation.

## What To Modify Next

- Document the exact cuFFT calls needed.
- Compare direct and FFT convolution crossover points.
