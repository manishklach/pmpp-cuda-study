# 067 - Audio Gain And Mixing

- Track: `Image and Signal`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Mix two short audio buffers with independent gains and clamp the result into a normalized range.

## PMPP Ideas To Focus On

- embarrassingly parallel elementwise work
- gain staging
- saturation or clamp behavior

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is the signal-processing equivalent of a vector blend kernel.
- Short synthetic waveforms keep the validation deterministic.
- A next step is chaining this with FIR filtering or spectrogram analysis.
