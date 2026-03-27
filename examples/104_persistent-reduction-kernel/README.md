# 104 - Persistent Reduction Kernel

## Overview

Study a persistent-thread approach to repeatedly reducing work queues or large arrays.

## Why this matters

Persistent kernels matter when launch overhead or dynamic work distribution becomes visible.

## Expected kernel structure

Long-lived blocks pull work in a loop, accumulate local state, and cooperate on final reduction.

## Future implementation notes

Keep the first implementation simple and make the work assignment logic explicit in comments.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
