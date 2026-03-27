# 107 - Stream Compaction Prefix Sum Fused

## Overview

Study a compaction pipeline that fuses predicate evaluation and prefix-sum style indexing.

## Why this matters

Fused compaction is useful when extra passes over the data are expensive.

## Expected kernel structure

Combine predicate generation, local scan, and output writes while keeping ordering semantics clear.

## Future implementation notes

Start from a correct compaction baseline and only then collapse phases together.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
