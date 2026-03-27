#include <cuda_runtime.h>

#include <algorithm>
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

constexpr const char *kExampleName = "116_bank-conflict-study";
constexpr int kTile = 32;

__global__ void transpose_conflict_kernel(const float *input, float *output, int width, int height) {
  __shared__ float tile[kTile][kTile];
  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  if (x < width && y < height)
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
  __syncthreads();
  int tx = blockIdx.y * kTile + threadIdx.x;
  int ty = blockIdx.x * kTile + threadIdx.y;
  if (tx < height && ty < width)
    output[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_padded_kernel(const float *input, float *output, int width, int height) {
  __shared__ float tile[kTile][kTile + 1];
  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  if (x < width && y < height)
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
  __syncthreads();
  int tx = blockIdx.y * kTile + threadIdx.x;
  int ty = blockIdx.x * kTile + threadIdx.y;
  if (tx < height && ty < width)
    output[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
}

std::vector<float> cpu_reference(const std::vector<float> &input, int width, int height) {
  std::vector<float> output(input.size(), 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      output[x * height + y] = input[y * width + x];
  return output;
}

std::vector<float> run_kernel(const std::vector<float> &input, int width, int height, bool padded) {
  std::vector<float> gpu(input.size(), 0.0f);
  const std::size_t bytes = input.size() * sizeof(float);
  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  if (padded)
    transpose_padded_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
  else
    transpose_conflict_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, bytes, cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  return gpu;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int width = std::max(kTile, options.size);
  const int height = std::max(kTile, options.size - options.size / 8);
  std::vector<float> input = pmpp::make_uniform_floats(width * height, options.seed, -2.0f, 2.0f);
  std::vector<float> cpu = cpu_reference(input, width, height);
  std::vector<float> conflict = run_kernel(input, width, height, false);
  std::vector<float> padded = run_kernel(input, width, height, true);

  pmpp::ValidationSummary conflict_summary = pmpp::compare_vectors(cpu, conflict, 1.0e-6f);
  pmpp::ValidationSummary padded_summary = pmpp::compare_vectors(cpu, padded, 1.0e-6f);
  pmpp::ValidationSummary summary = padded_summary;
  summary.ok = conflict_summary.ok && padded_summary.ok;
  summary.mismatch_count = conflict_summary.mismatch_count + padded_summary.mismatch_count;
  summary.max_abs_error = std::max(conflict_summary.max_abs_error, padded_summary.max_abs_error);
  summary.notes = "Both kernels compute the same transpose; the padded tile exists to avoid the common shared-memory bank-conflict pattern.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int width = std::max(kTile, options.size);
  const int height = std::max(kTile, options.size - options.size / 8);
  std::vector<float> input = pmpp::make_uniform_floats(width * height, options.seed, -2.0f, 2.0f);
  const std::size_t bytes = input.size() * sizeof(float);
  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    transpose_padded_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes * 2, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(static_cast<std::size_t>(width) * height, stats.avg_ms);
  stats.problem_label = "Matrix elements";
  stats.problem_size = static_cast<std::size_t>(width) * height;

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
  }
  return EXIT_SUCCESS;
}
