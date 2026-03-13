#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/report.cuh"

namespace {
constexpr const char *kExampleName = "040_top-k-selection";
__global__ void bitonic_step_kernel(int *data, int j, int k) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = i ^ j;
  if (ixj > i) {
    bool ascending = (i & k) == 0;
    if ((ascending && data[i] > data[ixj]) || (!ascending && data[i] < data[ixj])) {
      int t = data[i];
      data[i] = data[ixj];
      data[ixj] = t;
    }
  }
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  const int n = 32;
  const int k_top = 5;
  std::vector<int> input = {12, 99, 3, 47, 18, 76, 5, 65, 23, 88, 14, 54, 67, 31, 42, 90,
                            1, 72, 8, 60, 27, 81, 36, 95, 11, 58, 69, 20, 84, 7, 52, 40};
  auto cpu = input;
  std::sort(cpu.begin(), cpu.end(), std::greater<int>());
  std::vector<int> cpu_top(cpu.begin(), cpu.begin() + k_top);

  if (options.check) {
    int *d = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    for (int k = 2; k <= n; k <<= 1)
      for (int j = k >> 1; j > 0; j >>= 1) {
        bitonic_step_kernel<<<1, 128>>>(d, j, k);
        PMPP_CUDA_CHECK(cudaGetLastError());
      }
    PMPP_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> sorted(n);
    PMPP_CUDA_CHECK(cudaMemcpy(sorted.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::reverse(sorted.begin(), sorted.end());
    std::vector<int> gpu_top(sorted.begin(), sorted.begin() + k_top);
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu_top, gpu_top);
    summary.notes = "This example sorts the full array first, then slices the top-k values.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(d));
    if (!summary.ok) return EXIT_FAILURE;
  }
  if (options.bench) {
    int *d = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      PMPP_CUDA_CHECK(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
      for (int k = 2; k <= n; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
          bitonic_step_kernel<<<1, 128>>>(d, j, k);
          PMPP_CUDA_CHECK(cudaGetLastError());
        }
      PMPP_CUDA_CHECK(cudaDeviceSynchronize());
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps(2ULL * n * sizeof(int), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
    if (!options.verify) std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(d));
  }
  return EXIT_SUCCESS;
}
