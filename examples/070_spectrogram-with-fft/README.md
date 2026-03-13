# 070 - Spectrogram With FFT

- Track: `Image and Signal`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Compute a small spectrogram using a direct DFT per time window so the workflow is visible even without cuFFT.

## PMPP Ideas To Focus On

- windowed signal analysis
- 2D output layout of time bins by frequency bins
- trading performance for transparency in a study example

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example intentionally uses a naive DFT instead of cuFFT so it stays self-contained.
- Each block handles one time window and each thread computes one frequency bin.
- A natural follow-up is replacing the inner DFT with cuFFT plans when a toolkit is available.
