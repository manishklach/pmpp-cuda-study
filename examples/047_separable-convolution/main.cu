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

constexpr const char *kExampleName = "047_separable-convolution";
constexpr int kRadius = 1;

__global__ void conv_horizontal_kernel(const float *input, const float *kernel, float *output,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float sum = 0.0f;
    for (int k = -kRadius; k <= kRadius; ++k) {
      int sx = min(max(x + k, 0), width - 1);
      sum += input[y * width + sx] * kernel[k + kRadius];
    }
    output[y * width + x] = sum;
  }
}

__global__ void conv_vertical_kernel(const float *input, const float *kernel, float *output,
                                     int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float sum = 0.0f;
    for (int k = -kRadius; k <= kRadius; ++k) {
      int sy = min(max(y + k, 0), height - 1);
      sum += input[sy * width + x] * kernel[k + kRadius];
    }
    output[y * width + x] = sum;
  }
}

std::vector<float> cpu_reference(const std::vector<float> &input, int width, int height,
                                 const std::vector<float> &kernel) {
  std::vector<float> temp(input.size(), 0.0f);
  std::vector<float> output(input.size(), 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int k = -kRadius; k <= kRadius; ++k) {
        int sx = std::clamp(x + k, 0, width - 1);
        temp[y * width + x] += input[y * width + sx] * kernel[k + kRadius];
      }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int k = -kRadius; k <= kRadius; ++k) {
        int sy = std::clamp(y + k, 0, height - 1);
        output[y * width + x] += temp[sy * width + x] * kernel[k + kRadius];
      }
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int width = options.size;
  const int height = options.size;
  const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
  std::vector<float> input = pmpp::make_uniform_floats(width * height, options.seed, 0.0f, 1.0f);
  std::vector<float> kernel = {0.25f, 0.5f, 0.25f};
  std::vector<float> cpu = cpu_reference(input, width, height, kernel);
  std::vector<float> gpu(width * height, 0.0f);

  float *device_input = nullptr;
  float *device_kernel = nullptr;
  float *device_temp = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_kernel, kernel.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_temp, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_kernel, kernel.data(), kernel.size() * sizeof(float),
                             cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  conv_horizontal_kernel<<<blocks, threads>>>(device_input, device_kernel, device_temp, width, height);
  conv_vertical_kernel<<<blocks, threads>>>(device_temp, device_kernel, device_output, width, height);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_kernel));
  PMPP_CUDA_CHECK(cudaFree(device_temp));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
  summary.notes = "Separable convolution replaces a 2D filter with two 1D passes.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int width = options.size;
  const int height = options.size;
  const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
  std::vector<float> input = pmpp::make_uniform_floats(width * height, options.seed, 0.0f, 1.0f);
  std::vector<float> kernel = {0.25f, 0.5f, 0.25f};

  float *device_input = nullptr;
  float *device_kernel = nullptr;
  float *device_temp = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_kernel, kernel.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_temp, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_kernel, kernel.data(), kernel.size() * sizeof(float),
                             cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    conv_horizontal_kernel<<<blocks, threads>>>(device_input, device_kernel, device_temp, width, height);
    conv_vertical_kernel<<<blocks, threads>>>(device_temp, device_kernel, device_output, width, height);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes * 3, stats.avg_ms);
  stats.throughput =
      pmpp::elements_per_second(static_cast<std::size_t>(width) * height, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_kernel));
  PMPP_CUDA_CHECK(cudaFree(device_temp));
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
                                 "Pixels/sec");
  }
  return EXIT_SUCCESS;
}
