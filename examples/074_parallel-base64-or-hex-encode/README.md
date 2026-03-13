# 074 - Parallel Base64 Or Hex Encode

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Implement the hex-encoding branch of a parallel text transform so each input byte expands independently into two output characters.

## PMPP Ideas To Focus On

- one-to-many output mapping
- lookup tables
- simple encoding pipelines

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- The example chooses hex because it is embarrassingly parallel and easy to validate.
- The title leaves room for experimenting with a more involved base64 version later.
- A next step is comparing global memory writes against vectorized stores.
