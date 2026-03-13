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

constexpr const char *kExampleName = "027_prefix-sum-work-efficient-scan";
constexpr int kMaxN = 128;

__global__ void blelloch_inclusive_kernel(const int *input, int *output, int n) {
  __shared__ int temp[kMaxN];
  int tid = threadIdx.x;
  temp[tid] = tid < n ? input[tid] : 0;
  __syncthreads();

  for (int stride = 1; stride < n; stride <<= 1) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n)
      temp[idx] += temp[idx - stride];
    __syncthreads();
  }

  if (tid == 0)
    temp[n - 1] = 0;
  __syncthreads();

  for (int stride = n >> 1; stride > 0; stride >>= 1) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) {
      int prior = temp[idx - stride];
      temp[idx - stride] = temp[idx];
      temp[idx] += prior;
    }
    __syncthreads();
  }

  if (tid < n)
    output[tid] = temp[tid] + input[tid];
}

std::vector<int> cpu_reference(const std::vector<int> &input) {
  std::vector<int> output(input.size(), 0);
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = input[i] + (i == 0 ? 0 : output[i - 1]);
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  int n = std::min(kMaxN, std::max(2, options.size));
  int power_of_two = 1;
  while (power_of_two < n)
    power_of_two <<= 1;
  n = std::min(kMaxN, power_of_two);

  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, 0, 5);
  std::vector<int> cpu = cpu_reference(input);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  blelloch_inclusive_kernel<<<1, kMaxN>>>(device_input, device_output, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, n * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes = "This work-efficient scan uses an up-sweep and down-sweep in one block.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  int n = std::min(kMaxN, std::max(2, options.size));
  int power_of_two = 1;
  while (power_of_two < n)
    power_of_two <<= 1;
  n = std::min(kMaxN, power_of_two);

  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, 0, 5);
  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    blelloch_inclusive_kernel<<<1, kMaxN>>>(device_input, device_output, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int) * 2,
                                              stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

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
