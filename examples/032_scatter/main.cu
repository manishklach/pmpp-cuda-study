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
constexpr const char *kExampleName = "032_scatter";
__global__ void scatter_kernel(const float *input, const int *destinations, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    output[destinations[idx]] = input[idx];
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int n = options.size;
  std::vector<float> input = pmpp::make_uniform_floats(n, options.seed, -5.0f, 5.0f);
  std::vector<int> destinations(n, 0);
  for (int i = 0; i < n; ++i)
    destinations[i] = (i * 5) % n;
  std::vector<float> cpu(n, 0.0f), gpu(n, 0.0f);
  for (int i = 0; i < n; ++i)
    cpu[destinations[i]] = input[i];

  if (options.check) {
    float *di = nullptr, *dout = nullptr;
    int *dd = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, n * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dd, n * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(dd, destinations.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
    scatter_kernel<<<(n + options.block_size - 1) / options.block_size, options.block_size>>>(di, dd, dout, n);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
    summary.notes = "Scatter writes to irregular destinations; this example keeps destinations unique.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dd));
    PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok)
      return EXIT_FAILURE;
  }
  if (options.bench) {
    float *di = nullptr, *dout = nullptr;
    int *dd = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, n * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dd, n * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(dd, destinations.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      PMPP_CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
      scatter_kernel<<<(n + options.block_size - 1) / options.block_size, options.block_size>>>(di, dd, dout, n);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps((3ULL * n) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dd));
    PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
