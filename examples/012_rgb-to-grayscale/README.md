# 012 - RGB To Grayscale

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **RGB To Grayscale**.

This is a core image-processing example for learning multidimensional grids and thread mapping on 2D data.

## PMPP Ideas To Focus On

- pixel structs
- image indexing
- weighted color transforms

## What You Should Learn Here

- How one thread maps to one pixel
- How RGB channels are combined into a single grayscale output
- Why image kernels usually use 2D launch configurations

## Study Prompts

- Find the expression that converts pixel coordinates into a linear array index.
- Explain why this example is a natural fit for independent threads.
- Try alternative grayscale weights and describe the visual meaning.

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or expected pattern.
- Start with the built-in small inputs before scaling up.

## What To Modify Next

- Try alternative luminance weights.
- Preserve alpha in a uchar4 variant.
- Experiment with different 2D block shapes.
