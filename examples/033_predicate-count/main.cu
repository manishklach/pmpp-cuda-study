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
constexpr const char *kExampleName = "033_predicate-count";
__global__ void count_positive_kernel(const int *input, int *count, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] > 0)
    atomicAdd(count, 1);
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  std::vector<int> input = pmpp::make_uniform_ints(options.size, options.seed, -5, 5);
  int cpu = 0;
  for (int value : input)
    if (value > 0)
      ++cpu;

  if (options.check) {
    int *di = nullptr, *dc = nullptr, gpu = 0;
    PMPP_CUDA_CHECK(cudaMalloc(&di, options.size * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&dc, sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), options.size * sizeof(int), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemset(dc, 0, sizeof(int)));
    count_positive_kernel<<<(options.size + options.block_size - 1) / options.block_size, options.block_size>>>(di, dc, options.size);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(&gpu, dc, sizeof(int), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_scalars(cpu, gpu, 0.0);
    summary.notes = "Predicate count is the scalar summary version of a filtering predicate.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dc));
    if (!summary.ok)
      return EXIT_FAILURE;
  }
  if (options.bench) {
    int *di = nullptr, *dc = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, options.size * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&dc, sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), options.size * sizeof(int), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      PMPP_CUDA_CHECK(cudaMemset(dc, 0, sizeof(int)));
      count_positive_kernel<<<(options.size + options.block_size - 1) / options.block_size, options.block_size>>>(di, dc, options.size);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps =
        pmpp::bandwidth_gbps((static_cast<std::size_t>(options.size) * sizeof(int)) + sizeof(int), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(options.size, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dc));
  }
  return EXIT_SUCCESS;
}
