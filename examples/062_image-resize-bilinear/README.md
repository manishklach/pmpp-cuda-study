# 062 - Image Resize Bilinear

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Resize a small grayscale image with bilinear interpolation to study weighted sampling in a 2D CUDA kernel.

## PMPP Ideas To Focus On

- fractional coordinates
- neighbor fetch patterns
- interpolation arithmetic and edge clamping

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example keeps the data small enough that you can inspect the four contributing pixels for any output sample.
- Comparing this against example 061 makes the smoothing effect easy to see.
- A next improvement would be using textures or normalized coordinates.
