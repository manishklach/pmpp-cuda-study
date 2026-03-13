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
constexpr const char *kExampleName = "034_find-first-match";
__global__ void find_first_kernel(const int *input, int target, int *first_index, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] == target)
    atomicMin(first_index, idx);
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  const int target = 42;
  std::vector<int> input(options.size, 7);
  if (options.size > 5) {
    input[options.size / 3] = target;
    input[(options.size * 2) / 3] = target;
  }
  int cpu = static_cast<int>(std::find(input.begin(), input.end(), target) - input.begin());

  if (options.check) {
    int *di = nullptr, *df = nullptr, gpu = options.size;
    PMPP_CUDA_CHECK(cudaMalloc(&di, options.size * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&df, sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), options.size * sizeof(int), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(df, &gpu, sizeof(int), cudaMemcpyHostToDevice));
    find_first_kernel<<<(options.size + options.block_size - 1) / options.block_size, options.block_size>>>(di, target, df, options.size);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(&gpu, df, sizeof(int), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_scalars(cpu, gpu, 0.0);
    summary.notes = "Atomic min collapses multiple matching threads down to the earliest index.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(df));
    if (!summary.ok)
      return EXIT_FAILURE;
  }
  if (options.bench) {
    int *di = nullptr, *df = nullptr, init = options.size;
    PMPP_CUDA_CHECK(cudaMalloc(&di, options.size * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&df, sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), options.size * sizeof(int), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      PMPP_CUDA_CHECK(cudaMemcpy(df, &init, sizeof(int), cudaMemcpyHostToDevice));
      find_first_kernel<<<(options.size + options.block_size - 1) / options.block_size, options.block_size>>>(di, target, df, options.size);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps =
        pmpp::bandwidth_gbps((static_cast<std::size_t>(options.size) * sizeof(int)) + sizeof(int), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(options.size, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(df));
  }
  return EXIT_SUCCESS;
}
