# 039 - Merge Two Sorted Arrays

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Merge two sorted arrays into one sorted output on the GPU and validate against `std::merge`.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```
