# 061 - Image Resize Nearest Neighbor

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Resize a small grayscale image on the GPU with nearest-neighbor sampling and compare it against a CPU reference.

## PMPP Ideas To Focus On

- 2D thread mapping
- source-to-destination coordinate transforms
- boundary-safe pixel fetches

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- Each output pixel is independent, so this is a good first image-resampling kernel.
- The example uses deterministic dimensions so you can inspect a few known coordinates by hand.
- A natural next step is to compare the visual and numerical difference against bilinear interpolation.
