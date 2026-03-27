# PMPP CUDA Study

Structured CUDA study repository with 100 numbered examples arranged as a PMPP-style progression from basic kernels to larger parallel patterns, linear algebra, image processing, simulation, and graph / ML workloads.

## Repo Identity

This repo exists to make CUDA study concrete. Each example is meant to be small enough to read in one sitting, but complete enough to compile, run, validate against a CPU reference, and compare with a nearby baseline or optimized variant where that comparison is educational.

The repository is organized by numbered example folders under `examples/`. Lower numbers focus on fundamentals, the middle ranges introduce classic PMPP patterns such as reduction, scan, histograms, and dense linear algebra, and later ranges broaden into image, simulation, and graph / ML workloads. The goal is not to be a production CUDA framework. The goal is a disciplined study library that rewards careful reading, correctness checks, and incremental optimization.

## Implemented So Far

- Foundations: `001-020` are runnable, with `002_vector-addition`, `007_saxpy`, `019_matrix-transpose-naive`, and `020_matrix-transpose-with-shared-memory` as especially strong starting points.
- Parallel patterns: `021-040` are runnable, including polished reduction, scan, histogram, compaction, gather / scatter, and selection examples.
- Linear algebra: `041-056` and `061-100` are implemented, with `042_naive-matrix-multiply` and `043_tiled-matrix-multiply` forming the core tiling study pair.
- Template-only holdouts: `057_lu-factorization-sketch`, `058_cholesky-factorization`, `059_qr-factorization-sketch`, and `060_fft-based-convolution`.

## Best Examples To Start With

- `002_vector-addition`: the cleanest CUDA workflow baseline
- `020_matrix-transpose-with-shared-memory`: an early shared-memory win with clear mapping
- `021_dot-product`: simple map-plus-reduce composition
- `023_sum-reduction`: core PMPP reduction pattern
- `042_naive-matrix-multiply`: dense-kernel baseline before optimization
- `043_tiled-matrix-multiply`: classic shared-memory reuse example

## Best Examples For Optimization Study

- `020_matrix-transpose-with-shared-memory`
- `023_sum-reduction`
- `026_prefix-sum-naive-scan`
- `027_prefix-sum-work-efficient-scan`
- `028_histogram-global-atomics`
- `029_histogram-shared-memory`
- `041_matrix-vector-multiply`
- `042_naive-matrix-multiply`
- `043_tiled-matrix-multiply`

## Best Examples For Interview Prep / Learning CUDA Patterns

- `002_vector-addition`: kernel launch, indexing, and validation
- `023_sum-reduction`: shared memory, divergence, and synchronization
- `027_prefix-sum-work-efficient-scan`: up-sweep / down-sweep reasoning
- `029_histogram-shared-memory`: privatization and contention tradeoffs
- `030_stream-compaction`: filtering with atomics versus scan-based indexing
- `043_tiled-matrix-multiply`: tiling, reuse, and `__syncthreads()`

## How To Study This Repo

1. Start with correctness. Compile an example, run `--check`, and make sure you can explain why the CPU reference is trusted.
2. Read the baseline first when there is one. For example, study `042` before `043`, `026` before `027`, and `028` before `029`.
3. Compare mapping and memory behavior. Ask which thread owns which output, which values are reused, and where synchronization is required.
4. Only then switch to `--bench`. Use timing to support what the code already suggests rather than replacing the explanation.

## Build Instructions

Direct `nvcc` build from an example folder:

```powershell
cd examples\023_sum-reduction
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536 --block-size 256
```

Top-level CMake build:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Assumptions:

- CUDA Toolkit with `nvcc` is installed and on `PATH`
- examples target C++17
- benchmark output is illustrative and meant for local comparison, not publication-grade performance claims

## Benchmarking Philosophy

Correctness first, then performance. The examples use deterministic inputs, CPU reference checks, and a lightweight warmup plus timed-iteration pattern so small changes can be compared without hiding the algorithm. These are educational microbenchmarks. They are useful for studying memory behavior, synchronization costs, and baseline-versus-improved structure, but they are not substitutes for production benchmarking on controlled hardware.

## Progress / Status

Current summary:

| Status | Count | Notes |
|---|---:|---|
| Implemented | 96 | Runnable CUDA examples with code and README notes |
| Template | 4 | Intentionally scaffolded numerically heavy topics: `057-060` |
| Planned | 0 | The current repo already contains the 100-example sequence |

Full example index:

| Example | Name | Category | Status |
|---|---|---|---|
| 001 | Hello World Kernel | Foundations | Implemented |
| 002 | Vector Addition | Foundations | Implemented |
| 003 | Vector Subtraction | Foundations | Implemented |
| 004 | Scalar Vector Multiply | Foundations | Implemented |
| 005 | Elementwise Array Square | Foundations | Implemented |
| 006 | Elementwise Absolute Value | Foundations | Implemented |
| 007 | Saxpy | Foundations | Implemented |
| 008 | Copy Array Kernel | Foundations | Implemented |
| 009 | Reverse Array | Foundations | Implemented |
| 010 | Clamp Values To Range | Foundations | Implemented |
| 011 | Threshold Binary Mask | Foundations | Implemented |
| 012 | Rgb To Grayscale | Foundations | Implemented |
| 013 | Image Inversion | Foundations | Implemented |
| 014 | Brightness Adjustment | Foundations | Implemented |
| 015 | Contrast Adjustment | Foundations | Implemented |
| 016 | 1d Stencil | Foundations | Implemented |
| 017 | 2d Stencil | Foundations | Implemented |
| 018 | Matrix Addition | Foundations | Implemented |
| 019 | Matrix Transpose Naive | Foundations | Implemented |
| 020 | Matrix Transpose With Shared Memory | Foundations | Implemented |
| 021 | Dot Product | Parallel Patterns | Implemented |
| 022 | L2 Norm | Parallel Patterns | Implemented |
| 023 | Sum Reduction | Parallel Patterns | Implemented |
| 024 | Max Reduction | Parallel Patterns | Implemented |
| 025 | Min Reduction | Parallel Patterns | Implemented |
| 026 | Prefix Sum Naive Scan | Parallel Patterns | Implemented |
| 027 | Prefix Sum Work Efficient Scan | Parallel Patterns | Implemented |
| 028 | Histogram Global Atomics | Parallel Patterns | Implemented |
| 029 | Histogram Shared Memory | Parallel Patterns | Implemented |
| 030 | Stream Compaction | Parallel Patterns | Implemented |
| 031 | Gather | Parallel Patterns | Implemented |
| 032 | Scatter | Parallel Patterns | Implemented |
| 033 | Predicate Count | Parallel Patterns | Implemented |
| 034 | Find First Match | Parallel Patterns | Implemented |
| 035 | Parallel Even Odd Sort | Parallel Patterns | Implemented |
| 036 | Bitonic Sort | Parallel Patterns | Implemented |
| 037 | Odd Even Merge Sort | Parallel Patterns | Implemented |
| 038 | Parallel Binary Search Over Sorted Chunks | Parallel Patterns | Implemented |
| 039 | Merge Two Sorted Arrays | Parallel Patterns | Implemented |
| 040 | Top K Selection | Parallel Patterns | Implemented |
| 041 | Matrix Vector Multiply | Linear Algebra | Implemented |
| 042 | Naive Matrix Multiply | Linear Algebra | Implemented |
| 043 | Tiled Matrix Multiply | Linear Algebra | Implemented |
| 044 | Batched Matrix Multiply | Linear Algebra | Implemented |
| 045 | Convolution 1d | Linear Algebra | Implemented |
| 046 | Convolution 2d | Linear Algebra | Implemented |
| 047 | Separable Convolution | Linear Algebra | Implemented |
| 048 | Sobel Edge Detection | Linear Algebra | Implemented |
| 049 | Gaussian Blur | Linear Algebra | Implemented |
| 050 | Median Filter | Linear Algebra | Implemented |
| 051 | Box Filter With Shared Memory | Linear Algebra | Implemented |
| 052 | Sparse Matrix Vector Multiply Csr | Linear Algebra | Implemented |
| 053 | Sparse Matrix Dense Vector Multiply | Linear Algebra | Implemented |
| 054 | Jacobi Iteration | Linear Algebra | Implemented |
| 055 | Red Black Relaxation | Linear Algebra | Implemented |
| 056 | Power Iteration | Linear Algebra | Implemented |
| 057 | Lu Factorization Sketch | Linear Algebra | Template |
| 058 | Cholesky Factorization | Linear Algebra | Template |
| 059 | Qr Factorization Sketch | Linear Algebra | Template |
| 060 | Fft Based Convolution | Linear Algebra | Template |
| 061 | Image Resize Nearest Neighbor | Image And Signal | Implemented |
| 062 | Image Resize Bilinear | Image And Signal | Implemented |
| 063 | Template Matching | Image And Signal | Implemented |
| 064 | Non Maximum Suppression | Image And Signal | Implemented |
| 065 | Integral Image | Image And Signal | Implemented |
| 066 | Canny Pipeline Stages | Image And Signal | Implemented |
| 067 | Audio Gain And Mixing | Image And Signal | Implemented |
| 068 | Fir Filter | Image And Signal | Implemented |
| 069 | Iir Filter Sections | Image And Signal | Implemented |
| 070 | Spectrogram With Fft | Image And Signal | Implemented |
| 071 | Peak Detection | Image And Signal | Implemented |
| 072 | Delta Encoding | Image And Signal | Implemented |
| 073 | Run Length Encoding | Image And Signal | Implemented |
| 074 | Parallel Base64 Or Hex Encode | Image And Signal | Implemented |
| 075 | Block Crc Checksum | Image And Signal | Implemented |
| 076 | Monte Carlo Pi | Simulation | Implemented |
| 077 | Monte Carlo Option Pricing | Simulation | Implemented |
| 078 | Random Walk Simulation | Simulation | Implemented |
| 079 | N Body Naive | Simulation | Implemented |
| 080 | N Body Tiled | Simulation | Implemented |
| 081 | Lennard Jones Forces | Simulation | Implemented |
| 082 | Heat Diffusion Grid | Simulation | Implemented |
| 083 | Wave Equation Solver | Simulation | Implemented |
| 084 | Lattice Boltzmann Step | Simulation | Implemented |
| 085 | Game Of Life | Simulation | Implemented |
| 086 | Boids Flocking | Simulation | Implemented |
| 087 | Mandelbrot Renderer | Simulation | Implemented |
| 088 | Julia Renderer | Simulation | Implemented |
| 089 | Ray Sphere Tracer | Simulation | Implemented |
| 090 | Path Tracing Diffuse Scene | Simulation | Implemented |
| 091 | Parallel Bfs | Graph And ML | Implemented |
| 092 | Single Source Shortest Path | Graph And ML | Implemented |
| 093 | Pagerank | Graph And ML | Implemented |
| 094 | Connected Components | Graph And ML | Implemented |
| 095 | Union Find | Graph And ML | Implemented |
| 096 | K Means Clustering | Graph And ML | Implemented |
| 097 | Dbscan Acceleration | Graph And ML | Implemented |
| 098 | Neural Network Forward Pass | Graph And ML | Implemented |
| 099 | Mlp Backpropagation | Graph And ML | Implemented |
| 100 | Multi Gpu All Reduce Study | Graph And ML | Implemented |

## Supporting Docs

- [docs/example-conventions.md](docs/example-conventions.md)
- [docs/maturity-model.md](docs/maturity-model.md)
- [docs/status.md](docs/status.md)
- [scripts/validate_repo.py](scripts/validate_repo.py)
- [scripts/build_examples.py](scripts/build_examples.py)
