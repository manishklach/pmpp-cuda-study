#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "043_tiled-matrix-multiply";
constexpr int kTile = 16;

__global__ void matmul_tiled_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];

  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float sum = 0.0f;

  for (int tile_idx = 0; tile_idx < (k + kTile - 1) / kTile; ++tile_idx) {
    int a_col = tile_idx * kTile + threadIdx.x;
    int b_row = tile_idx * kTile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int e = 0; e < kTile; ++e)
      sum += tile_a[threadIdx.y][e] * tile_b[e][threadIdx.x];
    __syncthreads();
  }

  if (row < m && col < n)
    c[row * n + col] = sum;
}

std::vector<float> cpu_reference(const std::vector<float> &a, const std::vector<float> &b, int size) {
  std::vector<float> c(size * size, 0.0f);
  for (int row = 0; row < size; ++row)
    for (int col = 0; col < size; ++col)
      for (int e = 0; e < size; ++e)
        c[row * size + col] += a[row * size + e] * b[e * size + col];
  return c;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int size = options.size;
  const std::size_t matrix_bytes = static_cast<std::size_t>(size) * size * sizeof(float);
  std::vector<float> a = pmpp::make_uniform_floats(size * size, options.seed, -2.0f, 2.0f);
  std::vector<float> b = pmpp::make_uniform_floats(size * size, options.seed + 1, -2.0f, 2.0f);
  std::vector<float> gpu(size * size, 0.0f);
  std::vector<float> cpu = cpu_reference(a, b, size);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_c, matrix_bytes));

  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((size + kTile - 1) / kTile, (size + kTile - 1) / kTile);
  matmul_tiled_kernel<<<blocks, threads>>>(device_a, device_b, device_c, size, size, size);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_c, matrix_bytes, cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_c));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-4f);
  summary.notes = "Shared-memory tiling reduces repeated global-memory reads compared with the naive baseline.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int size = options.size;
  const std::size_t matrix_bytes = static_cast<std::size_t>(size) * size * sizeof(float);
  std::vector<float> a = pmpp::make_uniform_floats(size * size, options.seed, -2.0f, 2.0f);
  std::vector<float> b = pmpp::make_uniform_floats(size * size, options.seed + 1, -2.0f, 2.0f);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_c, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((size + kTile - 1) / kTile, (size + kTile - 1) / kTile);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    matmul_tiled_kernel<<<blocks, threads>>>(device_a, device_b, device_c, size, size, size);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(matrix_bytes * 3, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(static_cast<std::size_t>(size) * size, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_c));
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);

  if (options.check) {
    pmpp::ValidationSummary summary = run_check(options);
    pmpp::print_validation_report(kExampleName, summary);
    if (!summary.ok)
      return EXIT_FAILURE;
  }

  if (options.bench) {
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::BenchmarkStats stats = run_bench(options);
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Output elements/sec");
  }

  return EXIT_SUCCESS;
}
