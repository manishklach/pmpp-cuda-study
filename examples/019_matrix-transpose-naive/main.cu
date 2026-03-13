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

constexpr const char *kExampleName = "019_matrix-transpose-naive";

__global__ void transpose_naive_kernel(const float *input, float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
    output[x * height + y] = input[y * width + x];
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

  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  transpose_naive_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-6f);
  summary.notes = "This baseline is intentionally simple and serves as the comparison point for the tiled transpose.";
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

  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    transpose_naive_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
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
