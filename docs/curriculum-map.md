# Curriculum Map

`pmpp-cuda-study` is organized as a PMPP-style CUDA curriculum, not a loose collection of kernels. The plan below extends the repo from a strong `001-100` base into a broader `001-140` long-form study track.

## Module Map

| Module | Range | Title | Purpose | Typical Concepts |
|---|---:|---|---|---|
| 01 | 001-012 | Foundations | Learn the CUDA execution model with the smallest complete kernels. | kernel launches, indexing, memory copies, validation |
| 02 | 013-022 | Memory and Execution Basics | Build intuition about execution hierarchy and memory behavior. | grid-stride loops, coalescing, divergence, shared memory |
| 03 | 023-036 | Reductions / Scans / Compaction | Learn the core collective PMPP patterns. | reductions, scans, segmented scan, compaction |
| 04 | 037-048 | Histograms / Sorting / Irregular Primitives | Study atomics, privatization, gather/scatter, and sorting stages. | histograms, radix pass, selection, irregular writes |
| 05 | 049-070 | Matrix / Stencil / Convolution | Cover dense numeric kernels and neighborhood-based computation. | matmul, transpose, stencils, convolution, interpolation |
| 06 | 071-086 | Sparse / Graph Workloads | Move into irregular sparse and graph-oriented workloads. | COO/CSR/ELL, BFS, SSSP, PageRank, connectivity |
| 07 | 087-104 | ML / Tensor Kernels | Connect CUDA fundamentals to modern ML inference/training primitives. | activations, norms, quantization, attention, MLPs |
| 08 | 105-120 | Simulation / Rendering / Scientific Computing | Show CUDA for Monte Carlo, particles, PDEs, and rendering. | random sampling, n-body, fluids, ray tracing |
| 09 | 121-132 | Performance Engineering | Turn performance into an explicit study subject. | occupancy, launch tuning, bandwidth, branch efficiency |
| 10 | 133-140 | Capstones / Optimization Ladders | Compare naive and optimized versions of the same workload family. | optimization ladders, benchmarking, tradeoff analysis |

## Full Numbered Example Catalog

### 01. Foundations

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 001 | `001_hello-world-kernel` | Hello World Kernel | beginner | Launch a minimal kernel and verify that device execution works at all. |
| 002 | `002_vector-add` | Vector Addition | beginner | Add two vectors elementwise as the canonical CUDA hello-world workflow. |
| 003 | `003_vector-subtract` | Vector Subtraction | beginner | Subtract vectors to reinforce 1D indexing and CPU/GPU validation. |
| 004 | `004_vector-multiply` | Vector Multiply | beginner | Multiply vectors elementwise to practice simple arithmetic kernels. |
| 005 | `005_scalar-vector-multiply` | Scalar Vector Multiply | beginner | Scale a vector by a scalar to reinforce parameter passing into kernels. |
| 006 | `006_copy-kernel` | Copy Kernel | beginner | Copy one array to another to isolate indexing and memory movement. |
| 007 | `007_reverse-array` | Reverse Array | beginner | Reverse an array to practice index transforms. |
| 008 | `008_clamp-values` | Clamp Values | beginner | Clamp an array into a range to introduce simple branching. |
| 009 | `009_threshold-mask` | Threshold Mask | beginner | Convert values into a binary mask using a threshold. |
| 010 | `010_saxpy` | SAXPY | beginner | Compute `a*x + y` as a classic BLAS-style warmup kernel. |
| 011 | `011_abs-and-square` | Abs And Square | beginner | Apply absolute value and square transforms to the same input. |
| 012 | `012_rgb-to-grayscale` | RGB To Grayscale | beginner | Convert RGB pixels to grayscale to introduce 2D image mapping. |

### 02. Memory and Execution Basics

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 013 | `013_grid-stride-loop` | Grid-Stride Loop | beginner | Rewrite a simple vector kernel using a grid-stride loop. |
| 014 | `014_coalesced-access-demo` | Coalesced Access Demo | intermediate | Compare contiguous memory access against a deliberately strided pattern. |
| 015 | `015_uncoalesced-access-demo` | Uncoalesced Access Demo | intermediate | Show bandwidth loss from poor global-memory access. |
| 016 | `016_warp-divergence-demo` | Warp Divergence Demo | intermediate | Demonstrate how branch-heavy code splits warp execution. |
| 017 | `017_shared-memory-intro` | Shared Memory Intro | beginner | Stage values in shared memory to illustrate block-local cooperation. |
| 018 | `018_bank-conflict-demo` | Bank Conflict Demo | advanced | Show how shared-memory access patterns can serialize through bank conflicts. |
| 019 | `019_constant-memory-demo` | Constant Memory Demo | intermediate | Use constant memory for small read-only parameters. |
| 020 | `020_texture-like-sampling-demo` | Texture-Like Sampling Demo | intermediate | Illustrate read-mostly sampling patterns for image-like data. |
| 021 | `021_pinned-memory-copy` | Pinned Memory Copy | intermediate | Compare pageable and pinned host-memory transfer behavior. |
| 022 | `022_async-copy-streams` | Async Copy With Streams | advanced | Overlap transfers and kernels using streams. |

### 03. Reductions / Scans / Compaction

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 023 | `023_sum-reduction` | Sum Reduction | intermediate | Reduce a large array to one sum using block partials. |
| 024 | `024_max-reduction` | Max Reduction | intermediate | Compute a maximum using a shared-memory reduction tree. |
| 025 | `025_min-reduction` | Min Reduction | intermediate | Compute a minimum using the same reduction structure. |
| 026 | `026_argmax-reduction` | Argmax Reduction | advanced | Compute both the maximum value and its index. |
| 027 | `027_argmin-reduction` | Argmin Reduction | advanced | Compute both the minimum value and its index. |
| 028 | `028_dot-product-reduction` | Dot Product Reduction | intermediate | Combine elementwise multiply with reduction into a dot product. |
| 029 | `029_l2-norm` | L2 Norm | intermediate | Compute an L2 norm with square-and-sum reduction. |
| 030 | `030_naive-scan` | Naive Scan | intermediate | Implement a conceptual prefix sum baseline. |
| 031 | `031_work-efficient-scan` | Work-Efficient Scan | advanced | Implement a Blelloch-style prefix sum. |
| 032 | `032_inclusive-scan` | Inclusive Scan | intermediate | Study inclusive scan semantics and validation. |
| 033 | `033_exclusive-scan` | Exclusive Scan | intermediate | Study exclusive scan semantics and validation. |
| 034 | `034_segmented-scan` | Segmented Scan | advanced | Compute prefix sums independently inside flagged segments. |
| 035 | `035_stream-compaction` | Stream Compaction | advanced | Filter an array into a dense output using scan-like structure. |
| 036 | `036_predicate-count` | Predicate Count | intermediate | Count how many values satisfy a condition. |

### 04. Histograms / Sorting / Irregular Primitives

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 037 | `037_histogram-global-atomics` | Histogram With Global Atomics | intermediate | Build a histogram directly with global atomics as a baseline. |
| 038 | `038_histogram-privatized` | Privatized Histogram | advanced | Use per-block privatization to reduce atomic contention. |
| 039 | `039_histogram-shared-memory` | Shared-Memory Histogram | advanced | Accumulate bins in shared memory before a global flush. |
| 040 | `040_gather` | Gather | intermediate | Collect values from arbitrary indices into a dense output. |
| 041 | `041_scatter` | Scatter | intermediate | Write values into arbitrary target indices to study irregular writes. |
| 042 | `042_naive-matrix-multiply` | Naive Matrix Multiply | intermediate | Use dense GEMM as a baseline before optimization. |
| 043 | `043_tiled-matrix-multiply` | Tiled Matrix Multiply | intermediate | Use shared-memory tiling to improve dense GEMM reuse. |
| 044 | `044_even-odd-sort` | Even-Odd Sort | intermediate | Sort a small array with compare-exchange passes. |
| 045 | `045_bitonic-sort` | Bitonic Sort | advanced | Implement a classic GPU sorting network. |
| 046 | `046_odd-even-merge-sort` | Odd-Even Merge Sort | advanced | Study another sorting-network style approach. |
| 047 | `047_radix-sort-pass` | Radix Sort Pass | advanced | Implement one digit pass of radix sort. |
| 048 | `048_top-k-selection` | Top-K Selection | advanced | Select top-k values from a small input as a study primitive. |

### 05. Matrix / Stencil / Convolution

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 049 | `049_gaussian-blur` | Gaussian Blur | intermediate | Apply a weighted blur to image data. |
| 050 | `050_box-blur` | Box Blur | intermediate | Apply a uniform blur as a simpler neighborhood filter. |
| 051 | `051_sobel-filter` | Sobel Filter | intermediate | Compute image gradients for edge detection. |
| 052 | `052_laplacian-filter` | Laplacian Filter | intermediate | Apply a Laplacian kernel to study second-derivative filters. |
| 053 | `053_sharpen-filter` | Sharpen Filter | intermediate | Sharpen an image using a stencil-style kernel. |
| 054 | `054_median-filter` | Median Filter | advanced | Apply a nonlinear neighborhood filter to image data. |
| 055 | `055_matrix-add` | Matrix Add | beginner | Add two matrices elementwise with 2D indexing. |
| 056 | `056_matrix-transpose-naive` | Matrix Transpose Naive | intermediate | Transpose a matrix using a straightforward global-memory kernel. |
| 057 | `057_matrix-transpose-tiled` | Matrix Transpose Tiled | intermediate | Transpose a matrix using shared memory for better locality. |
| 058 | `058_matrix-vector-multiply` | Matrix Vector Multiply | intermediate | Multiply a dense matrix by a vector. |
| 059 | `059_rectangular-matmul` | Rectangular Matrix Multiply | intermediate | Multiply non-square matrices with general row/column mapping. |
| 060 | `060_batched-matmul` | Batched Matrix Multiply | advanced | Compute many small GEMMs in one launch. |
| 061 | `061_1d-stencil` | 1D Stencil | intermediate | Apply a neighborhood stencil to a 1D signal. |
| 062 | `062_2d-stencil` | 2D Stencil | intermediate | Apply a 2D stencil with boundary handling. |
| 063 | `063_3d-stencil` | 3D Stencil | advanced | Apply a 3D stencil to a volume. |
| 064 | `064_convolution-1d` | 1D Convolution | intermediate | Convolve a signal with a short filter. |
| 065 | `065_convolution-2d` | 2D Convolution | intermediate | Convolve an image with a direct 2D kernel. |
| 066 | `066_separable-convolution` | Separable Convolution | advanced | Split blur into row and column passes for better efficiency. |
| 067 | `067_integral-image` | Integral Image | advanced | Build a summed-area table for fast region queries. |
| 068 | `068_image-resize-nearest` | Image Resize Nearest | intermediate | Resize an image with nearest-neighbor sampling. |
| 069 | `069_image-resize-bilinear` | Image Resize Bilinear | intermediate | Resize an image with bilinear interpolation. |
| 070 | `070_template-matching` | Template Matching | advanced | Slide a template and compute similarity scores. |

### 06. Sparse / Graph Workloads

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 071 | `071_sparse-vector-gather` | Sparse Vector Gather | intermediate | Gather sparse vector entries from an index list. |
| 072 | `072_spmv-coo` | SpMV COO | advanced | Multiply a COO sparse matrix by a dense vector. |
| 073 | `073_spmv-csr` | SpMV CSR | advanced | Multiply a CSR sparse matrix by a dense vector. |
| 074 | `074_spmv-ell` | SpMV ELL | advanced | Multiply an ELLPACK sparse matrix by a dense vector. |
| 075 | `075_spmm-csr-dense` | SpMM CSR-Dense | advanced | Multiply a CSR sparse matrix by a dense matrix. |
| 076 | `076_jacobi-iteration` | Jacobi Iteration | advanced | Solve a linear system iteratively with Jacobi updates. |
| 077 | `077_red-black-relaxation` | Red-Black Relaxation | advanced | Alternate updates on checkerboard partitions of a grid. |
| 078 | `078_power-iteration` | Power Iteration | advanced | Estimate a dominant eigenvector iteratively. |
| 079 | `079_bfs-frontier-expansion` | BFS Frontier Expansion | advanced | Expand one BFS frontier at a time using CSR edges. |
| 080 | `080_bfs-level-synchronous` | BFS Level Synchronous | advanced | Run BFS with host-controlled per-level launches. |
| 081 | `081_sssp-bellman-ford` | SSSP Bellman-Ford | advanced | Relax weighted graph edges repeatedly from one source. |
| 082 | `082_pagerank-iteration` | PageRank Iteration | advanced | Perform PageRank iterations on a small directed graph. |
| 083 | `083_connected-components-label-prop` | Connected Components Label Propagation | advanced | Use label propagation to find connected components. |
| 084 | `084_union-find` | Union Find | advanced | Build a GPU study version of disjoint-set unions. |
| 085 | `085_triangle-counting-baseline` | Triangle Counting Baseline | advanced | Count graph triangles with a simple baseline. |
| 086 | `086_graph-coloring-baseline` | Graph Coloring Baseline | advanced | Assign graph colors with a simple parallel heuristic. |

### 07. ML / Tensor Kernels

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 087 | `087_relu` | ReLU | beginner | Apply the ReLU activation elementwise. |
| 088 | `088_leaky-relu` | Leaky ReLU | beginner | Apply leaky ReLU to compare activation behavior. |
| 089 | `089_sigmoid` | Sigmoid | beginner | Apply the sigmoid activation to a vector. |
| 090 | `090_tanh` | Tanh | beginner | Apply tanh to a vector. |
| 091 | `091_softmax-rowwise` | Softmax Rowwise | advanced | Compute rowwise softmax for a small matrix. |
| 092 | `092_gelu` | GELU | intermediate | Apply the GELU activation with an approximate formula. |
| 093 | `093_batch-norm-inference` | Batch Norm Inference | advanced | Apply batch normalization in inference mode. |
| 094 | `094_layer-norm` | Layer Norm | advanced | Normalize one vector or row at a time using mean and variance. |
| 095 | `095_rmsnorm` | RMSNorm | advanced | Apply RMSNorm to introduce norm-only normalization. |
| 096 | `096_embedding-lookup` | Embedding Lookup | intermediate | Gather embedding rows from an index tensor. |
| 097 | `097_quantize-int8` | Quantize Int8 | advanced | Quantize floating-point values into int8 with scales. |
| 098 | `098_dequantize-int8` | Dequantize Int8 | advanced | Reconstruct floating-point values from int8 plus scale. |
| 099 | `099_mlp-forward` | MLP Forward Pass | advanced | Run a tiny MLP forward pass. |
| 100 | `100_mlp-output-backprop` | MLP Output Backprop | advanced | Compute output-layer gradients for a tiny MLP. |
| 101 | `101_attention-score-kernel` | Attention Score Kernel | advanced | Compute scaled dot-product attention scores `Q*K^T`. |
| 102 | `102_attention-softmax-apply` | Attention Softmax Apply | advanced | Apply softmax-normalized attention weights to values. |
| 103 | `103_batched-gemm-for-ml` | Batched GEMM For ML | advanced | Use many small matrix multiplies in an ML-style workload. |
| 104 | `104_conv2d-im2col-study` | Conv2D Im2Col Study | advanced | Show the im2col view of convolution for learning tensor lowering. |

### 08. Simulation / Rendering / Scientific Computing

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 105 | `105_monte-carlo-pi` | Monte Carlo Pi | intermediate | Estimate pi using random point samples. |
| 106 | `106_option-pricing-black-scholes-mc` | Option Pricing Monte Carlo | advanced | Price an option with one Monte Carlo path per thread. |
| 107 | `107_random-walk` | Random Walk | intermediate | Simulate many independent 1D random walks. |
| 108 | `108_heat-diffusion-2d` | Heat Diffusion 2D | intermediate | Perform explicit diffusion updates on a temperature grid. |
| 109 | `109_wave-equation-1d` | Wave Equation 1D | advanced | Advance a 1D wave equation with multi-buffer state. |
| 110 | `110_nbody-naive` | N-Body Naive | advanced | Compute all-pairs particle interactions naively. |
| 111 | `111_nbody-tiled` | N-Body Tiled | advanced | Reuse particle tiles in shared memory for locality. |
| 112 | `112_lennard-jones` | Lennard-Jones Forces | advanced | Compute molecular-style pairwise forces on a small system. |
| 113 | `113_game-of-life` | Game Of Life | intermediate | Update a cellular automaton grid in parallel. |
| 114 | `114_boids` | Boids | advanced | Compute one flocking update from local boid rules. |
| 115 | `115_mandelbrot` | Mandelbrot Renderer | intermediate | Render a Mandelbrot image with one thread per pixel. |
| 116 | `116_julia-set` | Julia Set Renderer | intermediate | Render a Julia set for a fixed complex constant. |
| 117 | `117_ray-sphere-tracer` | Ray-Sphere Tracer | advanced | Trace primary rays against a sphere with simple shading. |
| 118 | `118_path-tracing-one-bounce` | Path Tracing One Bounce | advanced | Estimate diffuse radiance with stochastic one-bounce sampling. |
| 119 | `119_lattice-boltzmann-step` | Lattice Boltzmann Step | advanced | Run one collide-and-stream step for a tiny fluid grid. |
| 120 | `120_fir-filter` | FIR Filter | intermediate | Apply a finite impulse response filter to a signal. |

### 09. Performance Engineering

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 121 | `121_iir-filter-channels` | IIR Filter Across Channels | advanced | Parallelize recursive filters across independent channels. |
| 122 | `122_warp-shuffle-reduction` | Warp Shuffle Reduction | advanced | Replace shared-memory traffic with warp shuffle operations. |
| 123 | `123_cooperative-groups-reduction` | Cooperative Groups Reduction | advanced | Organize collective reduction logic with cooperative groups. |
| 124 | `124_launch-configuration-study` | Launch Configuration Study | intermediate | Sweep block sizes for a fixed kernel and record performance. |
| 125 | `125_occupancy-study` | Occupancy Study | advanced | Explore how launch shape and resource use affect occupancy. |
| 126 | `126_instruction-throughput-study` | Instruction Throughput Study | advanced | Compare arithmetic-heavy kernels with different instruction mixes. |
| 127 | `127_memory-bandwidth-study` | Memory Bandwidth Study | advanced | Measure effective bandwidth across several access patterns. |
| 128 | `128_shared-memory-vs-global-study` | Shared Memory Vs Global Study | advanced | Compare kernels that reuse data with and without shared memory. |
| 129 | `129_fused-kernel-study` | Fused Kernel Study | advanced | Compare multiple small kernels against a fused variant. |
| 130 | `130_stream-overlap-study` | Stream Overlap Study | advanced | Measure overlap between transfers and kernel execution. |
| 131 | `131_register-pressure-demo` | Register Pressure Demo | advanced | Show how more temporaries can affect occupancy and speed. |
| 132 | `132_branch-efficiency-study` | Branch Efficiency Study | advanced | Measure the performance cost of branch-heavy control flow. |

### 10. Capstones / Optimization Ladders

| # | Slug | Title | Difficulty | Description |
|---:|---|---|---|---|
| 133 | `133_vector-add-optimization-ladder` | Vector Add Optimization Ladder | intermediate | Compare baseline, grid-stride, and transfer-aware vector add versions. |
| 134 | `134_reduction-optimization-ladder` | Reduction Optimization Ladder | advanced | Compare naive, shared-memory, warp-shuffle, and multi-pass reductions. |
| 135 | `135_histogram-optimization-ladder` | Histogram Optimization Ladder | advanced | Compare global-atomic, privatized, and shared-memory histogram variants. |
| 136 | `136_matmul-optimization-ladder` | Matmul Optimization Ladder | advanced | Compare naive, tiled, rectangular, and batched GEMM variants. |
| 137 | `137_convolution-optimization-ladder` | Convolution Optimization Ladder | advanced | Compare direct, separable, and cached convolution approaches. |
| 138 | `138_spmv-format-comparison` | SpMV Format Comparison | advanced | Compare COO, CSR, and ELL SpMV on the same sparse input. |
| 139 | `139_attention-kernel-ladder` | Attention Kernel Ladder | advanced | Compare naive attention-score computation against improved variants. |
| 140 | `140_end-to-end-mini-pmpp-capstone` | End-To-End Mini PMPP Capstone | advanced | Combine data movement, compute kernels, validation, and benchmarking in one mini project. |

## Recommended Initial Expansion Batch

High-leverage next implementations:

1. `013_grid-stride-loop`
2. `014_coalesced-access-demo`
3. `016_warp-divergence-demo`
4. `018_bank-conflict-demo`
5. `026_argmax-reduction`
6. `031_work-efficient-scan`
7. `034_segmented-scan`
8. `038_histogram-privatized`
9. `047_radix-sort-pass`
10. `057_matrix-transpose-tiled`
11. `059_rectangular-matmul`
12. `063_3d-stencil`
13. `066_separable-convolution`
14. `073_spmv-csr`
15. `079_bfs-frontier-expansion`
16. `082_pagerank-iteration`
17. `091_softmax-rowwise`
18. `094_layer-norm`
19. `097_quantize-int8`
20. `134_reduction-optimization-ladder`
