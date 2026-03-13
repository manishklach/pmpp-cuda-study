# 049 - Gaussian Blur

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Gaussian Blur**.

This is one of the key image-processing examples in the repo. It helps illustrate 2D thread mapping, neighborhood access, and weighted image filtering.

## PMPP Ideas To Focus On

- weighted smoothing
- normalized kernels
- image denoising

## What You Should Learn Here

- How each output pixel depends on a neighborhood of input pixels
- Why blur kernels are a good bridge from simple pixel transforms to stencil-style computation
- How this differs from `012_rgb-to-grayscale` even though both use image-shaped data

## Study Prompts

- Identify the neighborhood each thread reads for one blurred pixel.
- Explain why border handling matters in blur kernels.
- Compare this direct blur against a separable version elsewhere in the repo.

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or stays within tolerance.
- Start with the included tiny matrices before scaling up.

## What To Modify Next

- Increase the radius.
- Compare separable and direct blur variants.
- Try caching neighborhoods with shared memory.
