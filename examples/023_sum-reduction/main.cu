#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "023_sum-reduction";
constexpr int kMaxThreads = 256;

__global__ void sum_partials_kernel(const float *input, float *partials, int n) {
  __shared__ float scratch[kMaxThreads];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;

  scratch[local] = global < n ? input[global] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride)
      scratch[local] += scratch[local + stride];
    __syncthreads();
  }

  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

double cpu_reference(const std::vector<float> &input) {
  return std::accumulate(input.begin(), input.end(), 0.0);
}

double run_gpu_once(const std::vector<float> &input, int block_size) {
  const int n = static_cast<int>(input.size());
  const int blocks = (n + block_size - 1) / block_size;
  const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

  std::vector<float> partials(blocks, 0.0f);
  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  sum_partials_kernel<<<blocks, block_size>>>(device_input, device_partials, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(
      cudaMemcpy(partials.data(), device_partials, partial_bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
  return std::accumulate(partials.begin(), partials.end(), 0.0);
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  std::vector<float> input = pmpp::make_uniform_floats(options.size, options.seed, -1.0f, 1.0f);
  const double cpu_sum = cpu_reference(input);
  const double gpu_sum = run_gpu_once(input, options.block_size);

  pmpp::ValidationSummary summary = pmpp::compare_scalars(cpu_sum, gpu_sum, 1.0e-3);
  summary.notes = "GPU reduction computes block partials on device and final aggregation on host.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = options.size;
  const int blocks = (n + options.block_size - 1) / options.block_size;
  const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);
  std::vector<float> input = pmpp::make_uniform_floats(n, options.seed, -1.0f, 1.0f);

  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    sum_partials_kernel<<<blocks, options.block_size>>>(device_input, device_partials, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(input_bytes + partial_bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  if (options.block_size > kMaxThreads)
    options.block_size = kMaxThreads;

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
