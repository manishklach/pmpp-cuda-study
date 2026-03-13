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

constexpr const char *kExampleName = "026_prefix-sum-naive-scan";
constexpr int kMaxN = 128;

__global__ void hillis_steele_kernel(const int *input, int *output, int n) {
  __shared__ int data[kMaxN];
  int tid = threadIdx.x;
  data[tid] = tid < n ? input[tid] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset <<= 1) {
    int add = tid >= offset ? data[tid - offset] : 0;
    __syncthreads();
    if (tid < n)
      data[tid] += add;
    __syncthreads();
  }

  if (tid < n)
    output[tid] = data[tid];
}

std::vector<int> cpu_reference(const std::vector<int> &input) {
  std::vector<int> output(input.size(), 0);
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = input[i] + (i == 0 ? 0 : output[i - 1]);
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = std::min(kMaxN, std::max(2, options.size));
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, 0, 5);
  std::vector<int> cpu = cpu_reference(input);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  hillis_steele_kernel<<<1, kMaxN>>>(device_input, device_output, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, n * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes = "This example keeps the scan in one block so the naive Hillis-Steele structure is easy to inspect.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = std::min(kMaxN, std::max(2, options.size));
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, 0, 5);
  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    hillis_steele_kernel<<<1, kMaxN>>>(device_input, device_output, n);
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
