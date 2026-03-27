# 128 - Hash Table Probe Basic

## Overview

Study a simple GPU hash-table probe loop.

## Why this matters

Hash probing exposes branchy, irregular memory behavior in a compact example.

## Expected kernel structure

Use open addressing or linear probing with deterministic keys and probe sequences.

## Future implementation notes

Keep the first pass read-only or insert-light so correctness is easy to reason about.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
