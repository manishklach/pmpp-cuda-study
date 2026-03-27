# 147 - Winograd Conv Demo

## Overview

Show the structure of a tiny Winograd-style convolution transform.

## Why this matters

Winograd is an advanced study topic because it changes both arithmetic and dataflow.

## Expected kernel structure

Transform tiles and filters, multiply in the transformed domain, then inverse-transform outputs.

## Future implementation notes

Keep the demo small and heavily commented rather than chasing aggressive optimization.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
