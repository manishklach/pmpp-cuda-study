# 148 - KMeans Assignment Update

## Overview

Study one assignment and centroid-update step for k-means.

## Why this matters

K-means is a practical clustering workload with both dense distance work and irregular accumulation.

## Expected kernel structure

Assign points to nearest centroids, accumulate per-cluster sums and counts, then update centroids.

## Future implementation notes

Use deterministic point sets and exact CPU validation for both assignments and updates.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
