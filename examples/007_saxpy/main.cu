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

constexpr const char *kExampleName = "007_saxpy";
constexpr float kAlpha = 1.75f;

__global__ void saxpy_kernel(float alpha, const float *x, const float *y, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = alpha * x[idx] + y[idx];
}

std::vector<float> cpu_reference(const std::vector<float> &x, const std::vector<float> &y) {
  std::vector<float> output(x.size(), 0.0f);
  for (std::size_t i = 0; i < x.size(); ++i)
    output[i] = kAlpha * x[i] + y[i];
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = options.size;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  std::vector<float> x = pmpp::make_uniform_floats(n, options.seed, -2.0f, 2.0f);
  std::vector<float> y = pmpp::make_uniform_floats(n, options.seed + 7, -5.0f, 5.0f);
  std::vector<float> gpu_output(n, 0.0f);
  std::vector<float> cpu_output = cpu_reference(x, y);

  float *device_x = nullptr;
  float *device_y = nullptr;
  float *device_out = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_x, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_y, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_out, bytes));

  PMPP_CUDA_CHECK(cudaMemcpy(device_x, x.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_y, y.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (n + threads - 1) / threads;
  saxpy_kernel<<<blocks, threads>>>(kAlpha, device_x, device_y, device_out, n);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_out, bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_x));
  PMPP_CUDA_CHECK(cudaFree(device_y));
  PMPP_CUDA_CHECK(cudaFree(device_out));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu_output, gpu_output, 1.0e-5f);
  summary.notes = "This example validates a BLAS-style axpy pattern against a scalar CPU loop.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = options.size;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  std::vector<float> x = pmpp::make_uniform_floats(n, options.seed, -2.0f, 2.0f);
  std::vector<float> y = pmpp::make_uniform_floats(n, options.seed + 7, -5.0f, 5.0f);

  float *device_x = nullptr;
  float *device_y = nullptr;
  float *device_out = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_x, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_y, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_out, bytes));

  PMPP_CUDA_CHECK(cudaMemcpy(device_x, x.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_y, y.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (n + threads - 1) / threads;
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    saxpy_kernel<<<blocks, threads>>>(kAlpha, device_x, device_y, device_out, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes * 3, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_x));
  PMPP_CUDA_CHECK(cudaFree(device_y));
  PMPP_CUDA_CHECK(cudaFree(device_out));
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
