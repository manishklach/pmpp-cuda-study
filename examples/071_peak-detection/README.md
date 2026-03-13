# 071 - Peak Detection

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Detect 1D local maxima above a threshold in a short signal and emit a binary peak mask.

## PMPP Ideas To Focus On

- neighbor comparison
- thresholding
- signal-analysis post-processing

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a compact companion to the 2D suppression example.
- The output mask is easy to validate by printing peak indices.
- A follow-up is compacting the peak indices into a dense list.
