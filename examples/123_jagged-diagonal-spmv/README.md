# 123 - Jagged Diagonal SpMV

## Overview

Explore a jagged-diagonal sparse matrix layout.

## Why this matters

JDS is a useful format study for load balancing irregular row lengths.

## Expected kernel structure

Reorder rows by nonzero count, then traverse jagged diagonals with regularized access.

## Future implementation notes

Document the preprocessing cost and how it changes the compute layout.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
