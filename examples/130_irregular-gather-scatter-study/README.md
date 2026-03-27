# 130 - Irregular Gather Scatter Study

## Overview

Compare different gather and scatter patterns on irregular indices.

## Why this matters

Irregular gathers and scatters show how indexing alone can dominate performance.

## Expected kernel structure

Launch one thread per logical element, read or write through an index indirection, and validate exact placement.

## Future implementation notes

Keep the access pattern explicit so memory behavior is the main lesson.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
