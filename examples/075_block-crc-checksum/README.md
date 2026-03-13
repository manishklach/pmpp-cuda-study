# 075 - Block CRC Checksum

- Track: `Image and Signal`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Compute a CRC32 checksum for each fixed-size chunk of a byte buffer so every chunk can be verified independently.

## PMPP Ideas To Focus On

- chunk-level parallelism
- per-thread serial work on short segments
- checksums as data-integrity kernels

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example uses one thread per block-sized chunk for clarity.
- CRC is inherently sequential within each byte stream, but parallel across many chunks.
- A next extension is using slice-by-N tables or cooperative processing per chunk.
