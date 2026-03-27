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

constexpr const char *kExampleName = "042_naive-matrix-multiply";
constexpr int kTile = 16;

__global__ void matmul_naive_kernel(const float *a, const float *b, float *c, int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= size || col >= size)
    return;

  float sum = 0.0f;
  // One thread computes one output element. That mapping is easy to reason about, but each
  // thread rereads an entire row of A and an entire column of B directly from global memory.
  for (int k = 0; k < size; ++k)
    sum += a[row * size + k] * b[k * size + col];

  c[row * size + col] = sum;
}

std::vector<float> cpu_reference(const std::vector<float> &a, const std::vector<float> &b, int size) {
  std::vector<float> c(size * size, 0.0f);
  for (int row = 0; row < size; ++row)
    for (int col = 0; col < size; ++col)
      for (int k = 0; k < size; ++k)
        c[row * size + col] += a[row * size + k] * b[k * size + col];
  return c;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int size = options.size;
  const std::size_t matrix_bytes = static_cast<std::size_t>(size) * size * sizeof(float);
  const std::vector<float> a = pmpp::make_uniform_floats(size * size, options.seed, -2.0f, 2.0f);
  const std::vector<float> b =
      pmpp::make_uniform_floats(size * size, options.seed + 1, -2.0f, 2.0f);
  const std::vector<float> cpu = cpu_reference(a, b, size);
  std::vector<float> gpu(size * size, 0.0f);

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
  matmul_naive_kernel<<<blocks, threads>>>(device_a, device_b, device_c, size);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_c, matrix_bytes, cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_c));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-4f);
  summary.notes =
      "This baseline favors clarity: 2D output mapping, no shared memory, and heavy global reads.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int size = options.size;
  const std::size_t matrix_bytes = static_cast<std::size_t>(size) * size * sizeof(float);
  const std::vector<float> a = pmpp::make_uniform_floats(size * size, options.seed, -2.0f, 2.0f);
  const std::vector<float> b =
      pmpp::make_uniform_floats(size * size, options.seed + 1, -2.0f, 2.0f);

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
    matmul_naive_kernel<<<blocks, threads>>>(device_a, device_b, device_c, size);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(matrix_bytes * 3, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(static_cast<std::size_t>(size) * size, stats.avg_ms);
  stats.problem_label = "Matrix dimension";
  stats.problem_size = static_cast<std::size_t>(size);

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
