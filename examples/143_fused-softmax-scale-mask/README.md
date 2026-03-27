# 143 - Fused Softmax Scale Mask

## Overview

Study a masked and scaled softmax pattern similar to attention preprocessing.

## Why this matters

This is a useful bridge from plain softmax to attention-style kernels.

## Expected kernel structure

Apply scale and mask before the stable softmax reduction and normalization steps.

## Future implementation notes

Document what is fused, what stays separate, and why.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
