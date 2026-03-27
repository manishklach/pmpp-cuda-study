# 146 - Grouped Convolution Basic

## Overview

Study a simple grouped convolution structure.

## Why this matters

Grouped convolution is a useful kernel pattern for modern compact networks.

## Expected kernel structure

Partition channels by group, then run convolution independently per group.

## Future implementation notes

Keep the first pass direct and readable before considering tensor-core-oriented layouts.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
