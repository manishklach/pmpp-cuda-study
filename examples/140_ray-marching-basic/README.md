# 140 - Ray Marching Basic

## Overview

March rays through a simple signed-distance scene.

## Why this matters

Ray marching is a good study kernel for branchy iterative work per pixel.

## Expected kernel structure

Assign one thread per pixel, step along the ray, and stop on hit or max distance.

## Future implementation notes

Keep the first scene tiny and deterministic so image correctness is inspectable.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
