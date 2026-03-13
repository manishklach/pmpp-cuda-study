# PMPP CUDA Study With Templates

A structured CUDA study repository with **100 examples** ordered from simple kernels to advanced PMPP-style workloads.

## Included

- 100 numbered example folders with CUDA study material
- Real CUDA implementations for examples `001-020`
- Per-example study notes and implementation checklists
- A 5-batch GitHub publishing plan for `001-020` through `081-100`
- A scaffold generator script for the remaining template-heavy examples

## Quick Start

```powershell
cd examples\001_hello-world-kernel
nvcc -std=c++17 -O2 main.cu -o example.exe
.\example.exe
```

## Featured Study Examples

- Vector Addition: the canonical CUDA starting point for kernel launches, host/device allocation, and memory copies. See `002_vector-addition`.
- Matrix Multiplication: a foundational optimization example for thread mapping, tiling, memory reuse, and shared memory. See `042_naive-matrix-multiply` and `043_tiled-matrix-multiply`.
- Image Processing: useful for learning 2D grids and pixel mapping strategies. See `012_rgb-to-grayscale`, `049_gaussian-blur`, `061_image-resize-nearest-neighbor`, and `062_image-resize-bilinear`.
- Parallel Reduction: core PMPP material for collapsing large datasets into one answer while studying divergence and shared-memory optimization. See `023_sum-reduction`, `024_max-reduction`, and `025_min-reduction`.

## Example Index

| # | Example | Track | Difficulty | Link |
|---|---|---|---|---|
| 001 | Hello World Kernel | Foundations | Beginner | [Open](examples/001_hello-world-kernel/README.md) |
| 002 | Vector Addition | Foundations | Beginner | [Open](examples/002_vector-addition/README.md) |
| 003 | Vector Subtraction | Foundations | Beginner | [Open](examples/003_vector-subtraction/README.md) |
| 004 | Scalar Vector Multiply | Foundations | Beginner | [Open](examples/004_scalar-vector-multiply/README.md) |
| 005 | Elementwise Array Square | Foundations | Beginner | [Open](examples/005_elementwise-array-square/README.md) |
| 006 | Elementwise Absolute Value | Foundations | Beginner | [Open](examples/006_elementwise-absolute-value/README.md) |
| 007 | SAXPY | Foundations | Beginner | [Open](examples/007_saxpy/README.md) |
| 008 | Copy Array Kernel | Foundations | Beginner | [Open](examples/008_copy-array-kernel/README.md) |
| 009 | Reverse Array | Foundations | Beginner | [Open](examples/009_reverse-array/README.md) |
| 010 | Clamp Values To Range | Foundations | Beginner | [Open](examples/010_clamp-values-to-range/README.md) |
| 011 | Threshold Binary Mask | Foundations | Beginner | [Open](examples/011_threshold-binary-mask/README.md) |
| 012 | RGB To Grayscale | Foundations | Beginner | [Open](examples/012_rgb-to-grayscale/README.md) |
| 013 | Image Inversion | Foundations | Beginner | [Open](examples/013_image-inversion/README.md) |
| 014 | Brightness Adjustment | Foundations | Beginner | [Open](examples/014_brightness-adjustment/README.md) |
| 015 | Contrast Adjustment | Foundations | Beginner | [Open](examples/015_contrast-adjustment/README.md) |
| 016 | 1D Stencil | Foundations | Intermediate | [Open](examples/016_1d-stencil/README.md) |
| 017 | 2D Stencil | Foundations | Intermediate | [Open](examples/017_2d-stencil/README.md) |
| 018 | Matrix Addition | Foundations | Intermediate | [Open](examples/018_matrix-addition/README.md) |
| 019 | Matrix Transpose Naive | Foundations | Intermediate | [Open](examples/019_matrix-transpose-naive/README.md) |
| 020 | Matrix Transpose With Shared Memory | Foundations | Intermediate | [Open](examples/020_matrix-transpose-with-shared-memory/README.md) |
| 021 | Dot Product | Parallel Patterns | Intermediate | [Open](examples/021_dot-product/README.md) |
| 022 | L2 Norm | Parallel Patterns | Intermediate | [Open](examples/022_l2-norm/README.md) |
| 023 | Sum Reduction | Parallel Patterns | Intermediate | [Open](examples/023_sum-reduction/README.md) |
| 024 | Max Reduction | Parallel Patterns | Intermediate | [Open](examples/024_max-reduction/README.md) |
| 025 | Min Reduction | Parallel Patterns | Intermediate | [Open](examples/025_min-reduction/README.md) |
| 026 | Prefix Sum Naive Scan | Parallel Patterns | Intermediate | [Open](examples/026_prefix-sum-naive-scan/README.md) |
| 027 | Prefix Sum Work Efficient Scan | Parallel Patterns | Intermediate | [Open](examples/027_prefix-sum-work-efficient-scan/README.md) |
| 028 | Histogram Global Atomics | Parallel Patterns | Intermediate | [Open](examples/028_histogram-global-atomics/README.md) |
| 029 | Histogram Shared Memory | Parallel Patterns | Intermediate | [Open](examples/029_histogram-shared-memory/README.md) |
| 030 | Stream Compaction | Parallel Patterns | Intermediate | [Open](examples/030_stream-compaction/README.md) |
| 031 | Gather | Parallel Patterns | Intermediate | [Open](examples/031_gather/README.md) |
| 032 | Scatter | Parallel Patterns | Intermediate | [Open](examples/032_scatter/README.md) |
| 033 | Predicate Count | Parallel Patterns | Intermediate | [Open](examples/033_predicate-count/README.md) |
| 034 | Find First Match | Parallel Patterns | Intermediate | [Open](examples/034_find-first-match/README.md) |
| 035 | Parallel Even Odd Sort | Parallel Patterns | Intermediate | [Open](examples/035_parallel-even-odd-sort/README.md) |
| 036 | Bitonic Sort | Parallel Patterns | Intermediate | [Open](examples/036_bitonic-sort/README.md) |
| 037 | Odd Even Merge Sort | Parallel Patterns | Intermediate | [Open](examples/037_odd-even-merge-sort/README.md) |
| 038 | Parallel Binary Search Over Sorted Chunks | Parallel Patterns | Intermediate | [Open](examples/038_parallel-binary-search-over-sorted-chunks/README.md) |
| 039 | Merge Two Sorted Arrays | Parallel Patterns | Intermediate | [Open](examples/039_merge-two-sorted-arrays/README.md) |
| 040 | Top K Selection | Parallel Patterns | Intermediate | [Open](examples/040_top-k-selection/README.md) |
| 041 | Matrix Vector Multiply | Linear Algebra | Intermediate | [Open](examples/041_matrix-vector-multiply/README.md) |
| 042 | Naive Matrix Multiply | Linear Algebra | Intermediate | [Open](examples/042_naive-matrix-multiply/README.md) |
| 043 | Tiled Matrix Multiply | Linear Algebra | Intermediate | [Open](examples/043_tiled-matrix-multiply/README.md) |
| 044 | Batched Matrix Multiply | Linear Algebra | Intermediate | [Open](examples/044_batched-matrix-multiply/README.md) |
| 045 | Convolution 1D | Linear Algebra | Intermediate | [Open](examples/045_convolution-1d/README.md) |
| 046 | Convolution 2D | Linear Algebra | Intermediate | [Open](examples/046_convolution-2d/README.md) |
| 047 | Separable Convolution | Linear Algebra | Intermediate | [Open](examples/047_separable-convolution/README.md) |
| 048 | Sobel Edge Detection | Linear Algebra | Intermediate | [Open](examples/048_sobel-edge-detection/README.md) |
| 049 | Gaussian Blur | Linear Algebra | Intermediate | [Open](examples/049_gaussian-blur/README.md) |
| 050 | Median Filter | Linear Algebra | Intermediate | [Open](examples/050_median-filter/README.md) |
| 051 | Box Filter With Shared Memory | Linear Algebra | Advanced | [Open](examples/051_box-filter-with-shared-memory/README.md) |
| 052 | Sparse Matrix Vector Multiply CSR | Linear Algebra | Advanced | [Open](examples/052_sparse-matrix-vector-multiply-csr/README.md) |
| 053 | Sparse Matrix Dense Vector Multiply | Linear Algebra | Advanced | [Open](examples/053_sparse-matrix-dense-vector-multiply/README.md) |
| 054 | Jacobi Iteration | Linear Algebra | Advanced | [Open](examples/054_jacobi-iteration/README.md) |
| 055 | Red Black Relaxation | Linear Algebra | Advanced | [Open](examples/055_red-black-relaxation/README.md) |
| 056 | Power Iteration | Linear Algebra | Advanced | [Open](examples/056_power-iteration/README.md) |
| 057 | LU Factorization Sketch | Linear Algebra | Advanced | [Open](examples/057_lu-factorization-sketch/README.md) |
| 058 | Cholesky Factorization | Linear Algebra | Advanced | [Open](examples/058_cholesky-factorization/README.md) |
| 059 | QR Factorization Sketch | Linear Algebra | Advanced | [Open](examples/059_qr-factorization-sketch/README.md) |
| 060 | FFT Based Convolution | Linear Algebra | Advanced | [Open](examples/060_fft-based-convolution/README.md) |
| 061 | Image Resize Nearest Neighbor | Image and Signal | Advanced | [Open](examples/061_image-resize-nearest-neighbor/README.md) |
| 062 | Image Resize Bilinear | Image and Signal | Advanced | [Open](examples/062_image-resize-bilinear/README.md) |
| 063 | Template Matching | Image and Signal | Advanced | [Open](examples/063_template-matching/README.md) |
| 064 | Non Maximum Suppression | Image and Signal | Advanced | [Open](examples/064_non-maximum-suppression/README.md) |
| 065 | Integral Image | Image and Signal | Advanced | [Open](examples/065_integral-image/README.md) |
| 066 | Canny Pipeline Stages | Image and Signal | Advanced | [Open](examples/066_canny-pipeline-stages/README.md) |
| 067 | Audio Gain And Mixing | Image and Signal | Advanced | [Open](examples/067_audio-gain-and-mixing/README.md) |
| 068 | FIR Filter | Image and Signal | Advanced | [Open](examples/068_fir-filter/README.md) |
| 069 | IIR Filter Sections | Image and Signal | Advanced | [Open](examples/069_iir-filter-sections/README.md) |
| 070 | Spectrogram With FFT | Image and Signal | Advanced | [Open](examples/070_spectrogram-with-fft/README.md) |
| 071 | Peak Detection | Image and Signal | Advanced | [Open](examples/071_peak-detection/README.md) |
| 072 | Delta Encoding | Image and Signal | Advanced | [Open](examples/072_delta-encoding/README.md) |
| 073 | Run Length Encoding | Image and Signal | Advanced | [Open](examples/073_run-length-encoding/README.md) |
| 074 | Parallel Base64 Or Hex Encode | Image and Signal | Advanced | [Open](examples/074_parallel-base64-or-hex-encode/README.md) |
| 075 | Block CRC Checksum | Image and Signal | Advanced | [Open](examples/075_block-crc-checksum/README.md) |
| 076 | Monte Carlo Pi | Simulation | Advanced | [Open](examples/076_monte-carlo-pi/README.md) |
| 077 | Monte Carlo Option Pricing | Simulation | Advanced | [Open](examples/077_monte-carlo-option-pricing/README.md) |
| 078 | Random Walk Simulation | Simulation | Advanced | [Open](examples/078_random-walk-simulation/README.md) |
| 079 | N Body Naive | Simulation | Advanced | [Open](examples/079_n-body-naive/README.md) |
| 080 | N Body Tiled | Simulation | Advanced | [Open](examples/080_n-body-tiled/README.md) |
| 081 | Lennard Jones Forces | Simulation | Advanced | [Open](examples/081_lennard-jones-forces/README.md) |
| 082 | Heat Diffusion Grid | Simulation | Advanced | [Open](examples/082_heat-diffusion-grid/README.md) |
| 083 | Wave Equation Solver | Simulation | Advanced | [Open](examples/083_wave-equation-solver/README.md) |
| 084 | Lattice Boltzmann Step | Simulation | Advanced | [Open](examples/084_lattice-boltzmann-step/README.md) |
| 085 | Game Of Life | Simulation | Advanced | [Open](examples/085_game-of-life/README.md) |
| 086 | Boids Flocking | Simulation | Advanced | [Open](examples/086_boids-flocking/README.md) |
| 087 | Mandelbrot Renderer | Simulation | Advanced | [Open](examples/087_mandelbrot-renderer/README.md) |
| 088 | Julia Renderer | Simulation | Advanced | [Open](examples/088_julia-renderer/README.md) |
| 089 | Ray Sphere Tracer | Simulation | Advanced | [Open](examples/089_ray-sphere-tracer/README.md) |
| 090 | Path Tracing Diffuse Scene | Simulation | Advanced | [Open](examples/090_path-tracing-diffuse-scene/README.md) |
| 091 | Parallel BFS | Graph and ML | Advanced | [Open](examples/091_parallel-bfs/README.md) |
| 092 | Single Source Shortest Path | Graph and ML | Advanced | [Open](examples/092_single-source-shortest-path/README.md) |
| 093 | PageRank | Graph and ML | Advanced | [Open](examples/093_pagerank/README.md) |
| 094 | Connected Components | Graph and ML | Advanced | [Open](examples/094_connected-components/README.md) |
| 095 | Union Find | Graph and ML | Advanced | [Open](examples/095_union-find/README.md) |
| 096 | K Means Clustering | Graph and ML | Advanced | [Open](examples/096_k-means-clustering/README.md) |
| 097 | DBSCAN Acceleration | Graph and ML | Advanced | [Open](examples/097_dbscan-acceleration/README.md) |
| 098 | Neural Network Forward Pass | Graph and ML | Advanced | [Open](examples/098_neural-network-forward-pass/README.md) |
| 099 | MLP Backpropagation | Graph and ML | Advanced | [Open](examples/099_mlp-backpropagation/README.md) |
| 100 | Multi GPU All Reduce Study | Graph and ML | Advanced | [Open](examples/100_multi-gpu-all-reduce-study/README.md) |
