# 122 - Ellpack_SpMV

## Overview

Study ELLPACK as a regularized sparse matrix-vector format.

## Why this matters

ELLPACK shows how padding can trade space for more regular memory access.

## Expected kernel structure

Assign one thread per row and iterate a padded width with sentinel entries.

## Future implementation notes

Compare memory regularity against wasted padded work.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
