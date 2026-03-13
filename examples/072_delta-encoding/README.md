# 072 - Delta Encoding

- Track: `Image and Signal`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Convert a monotonically increasing integer stream into first differences, which is a common compression pre-processing step.

## PMPP Ideas To Focus On

- neighbor dependencies with one-element lookback
- edge handling for the first output element
- compression-oriented transforms

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a simple but useful example of local differencing.
- It pairs well with scan and run-length encoding examples later in the repo.
- A next extension is decoding on the GPU with an exclusive scan.
