# 134 - Fft Butterfly Demo

## Overview

Visualize or study one stage of an FFT butterfly.

## Why this matters

Butterfly patterns are foundational for FFT thinking even in a small demo.

## Expected kernel structure

Pair elements by stage stride, apply twiddle factors, and write the transformed values to staged buffers.

## Future implementation notes

Focus the first version on one or two stages with deterministic inputs.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
