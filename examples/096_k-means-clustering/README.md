# 096 - K Means Clustering

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Run a few K-means iterations on a tiny 2D dataset using separate assignment and centroid-update phases.

## PMPP Ideas To Focus On

- point-parallel distance evaluation
- reduction into cluster sums
- iterative ML-style optimization loops

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a compact clustering example with clearly separated phases.
- Atomic sums are acceptable here because the dataset is intentionally tiny.
- A next step is using shared memory or mini-batches for larger inputs.
