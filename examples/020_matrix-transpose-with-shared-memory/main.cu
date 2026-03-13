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

constexpr const char *kExampleName = "020_matrix-transpose-with-shared-memory";
constexpr int kTileDim = 16;

__global__ void transpose_tiled_kernel(const float *input, float *output, int width, int height) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  if (x < width && y < height)
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
  __syncthreads();

  int tx = blockIdx.y * kTileDim + threadIdx.x;
  int ty = blockIdx.x * kTileDim + threadIdx.y;
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

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int width = options.size;
  const int height = std::max(2, options.size - options.size / 4);
  const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
  std::vector<float> input = pmpp::make_uniform_floats(width * height, options.seed, -3.0f, 3.0f);
  std::vector<float> gpu(width * height, 0.0f);
  std::vector<float> cpu = cpu_reference(input, width, height);

  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTileDim, kTileDim);
  dim3 blocks((width + kTileDim - 1) / kTileDim, (height + kTileDim - 1) / kTileDim);
  transpose_tiled_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, bytes, cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-6f);
  summary.notes = "The padded shared-memory tile avoids common bank-conflict patterns during transpose.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int width = options.size;
  const int height = std::max(2, options.size - options.size / 4);
  const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
  std::vector<float> input = pmpp::make_uniform_floats(width * height, options.seed, -3.0f, 3.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTileDim, kTileDim);
  dim3 blocks((width + kTileDim - 1) / kTileDim, (height + kTileDim - 1) / kTileDim);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    transpose_tiled_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes * 2, stats.avg_ms);
  stats.throughput =
      pmpp::elements_per_second(static_cast<std::size_t>(width) * height, stats.avg_ms);

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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Elements/sec");
  }

  return EXIT_SUCCESS;
}
