# 073 - Run Length Encoding

- Track: `Image and Signal`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Identify run starts in a symbol stream and emit compact `(value, length)` pairs with a simple atomic write-out.

## PMPP Ideas To Focus On

- flagging segment boundaries
- compaction-like output generation
- variable-length result handling

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This version favors a small and understandable pipeline over a fully parallel scan-based encoder.
- Each run-start thread claims one slot in the compact output arrays.
- A next PMPP step is replacing the atomic counter with prefix sums over the flags.
