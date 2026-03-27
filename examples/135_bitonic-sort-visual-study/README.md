# 135 - Bitonic Sort Visual Study

## Overview

Study the compare-exchange structure of bitonic sort.

## Why this matters

Bitonic sort is a useful GPU sorting pattern because the schedule is explicit.

## Expected kernel structure

Launch compare-exchange passes by stage and direction, with clear comments on partner selection.

## Future implementation notes

Use small deterministic arrays so each stage can be checked against a CPU sort.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
