# 061 - Image Resize Nearest Neighbor

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Resize a grayscale image with nearest-neighbor sampling and validate against a CPU reference.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 256
```
