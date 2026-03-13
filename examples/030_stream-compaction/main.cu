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

constexpr const char *kExampleName = "030_stream-compaction";
constexpr int kMaxN = 128;

__global__ void compact_positive_kernel(const int *input, int *output, int *count, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] > 0) {
    int slot = atomicAdd(count, 1);
    output[slot] = input[idx];
  }
}

std::vector<int> cpu_reference(const std::vector<int> &input) {
  std::vector<int> output;
  for (int value : input)
    if (value > 0)
      output.push_back(value);
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = std::min(kMaxN, std::max(2, options.size));
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, -4, 4);
  std::vector<int> cpu = cpu_reference(input);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  int *device_count = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));

  compact_positive_kernel<<<1, kMaxN>>>(device_input, device_output, device_count, n);
  PMPP_CUDA_KERNEL_CHECK();

  int count = 0;
  PMPP_CUDA_CHECK(cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost));
  gpu.resize(count);
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, count * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  PMPP_CUDA_CHECK(cudaFree(device_count));

  std::sort(cpu.begin(), cpu.end());
  std::sort(gpu.begin(), gpu.end());
  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes = "This compaction baseline validates the kept set, not stable ordering, because atomic output order is not guaranteed.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = std::min(kMaxN, std::max(2, options.size));
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, -4, 4);
  int *device_input = nullptr;
  int *device_output = nullptr;
  int *device_count = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));
    compact_positive_kernel<<<1, kMaxN>>>(device_input, device_output, device_count, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int) * 2,
                                              stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  PMPP_CUDA_CHECK(cudaFree(device_count));
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
