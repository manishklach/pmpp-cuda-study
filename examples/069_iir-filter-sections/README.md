# 069 - IIR Filter Sections

- Track: `Image and Signal`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Run a simple first-order IIR section per channel to show how GPU parallelism can live across independent signal streams even when time within each stream remains sequential.

## PMPP Ideas To Focus On

- parallelism across channels
- loop-carried dependencies within each thread
- mapping limited intra-signal parallelism to CUDA

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a useful example because not every DSP algorithm parallelizes across time samples.
- Each thread owns one channel and steps through time sequentially.
- A next extension is stacking multiple biquad sections per channel.
