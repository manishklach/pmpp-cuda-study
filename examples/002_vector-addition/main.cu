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

constexpr const char *kExampleName = "002_vector-addition";

__global__ void vector_add_kernel(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = a[idx] + b[idx];
}

std::vector<float> cpu_reference(const std::vector<float> &a, const std::vector<float> &b) {
  std::vector<float> output(a.size(), 0.0f);
  for (std::size_t i = 0; i < a.size(); ++i)
    output[i] = a[i] + b[i];
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = options.size;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  std::vector<float> a = pmpp::make_uniform_floats(n, options.seed, -4.0f, 4.0f);
  std::vector<float> b = pmpp::make_uniform_floats(n, options.seed + 1, -3.0f, 3.0f);
  std::vector<float> gpu_output(n, 0.0f);
  std::vector<float> cpu_output = cpu_reference(a, b);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_out = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_out, bytes));

  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (n + threads - 1) / threads;
  vector_add_kernel<<<blocks, threads>>>(device_a, device_b, device_out, n);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_out, bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_out));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu_output, gpu_output, 1.0e-5f);
  summary.notes = "CPU reference compares elementwise sums for deterministic random inputs.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = options.size;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  std::vector<float> a = pmpp::make_uniform_floats(n, options.seed, -4.0f, 4.0f);
  std::vector<float> b = pmpp::make_uniform_floats(n, options.seed + 1, -3.0f, 3.0f);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_out = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_out, bytes));

  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (n + threads - 1) / threads;

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    vector_add_kernel<<<blocks, threads>>>(device_a, device_b, device_out, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes * 3, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
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
