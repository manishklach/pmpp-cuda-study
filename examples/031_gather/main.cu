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
constexpr const char *kExampleName = "031_gather";
__global__ void gather_kernel(const float *source, const int *indices, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    output[idx] = source[indices[idx]];
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int n = options.size;
  std::vector<float> source = pmpp::make_uniform_floats(n * 2, options.seed, -5.0f, 5.0f);
  std::vector<int> indices(n, 0);
  for (int i = 0; i < n; ++i)
    indices[i] = (i * 3) % static_cast<int>(source.size());
  std::vector<float> cpu(n, 0.0f), gpu(n, 0.0f);
  for (int i = 0; i < n; ++i)
    cpu[i] = source[indices[i]];

  if (options.check) {
    float *ds = nullptr, *dout = nullptr;
    int *di = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&ds, source.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&di, n * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(ds, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(di, indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    gather_kernel<<<(n + options.block_size - 1) / options.block_size, options.block_size>>>(ds, di, dout, n);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
    summary.notes = "Gather reads from irregular source indices and writes densely.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(ds));
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok)
      return EXIT_FAILURE;
  }

  if (options.bench) {
    float *ds = nullptr, *dout = nullptr;
    int *di = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&ds, source.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&di, n * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(ds, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(di, indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      gather_kernel<<<(n + options.block_size - 1) / options.block_size, options.block_size>>>(ds, di, dout, n);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps =
        pmpp::bandwidth_gbps((source.size() + 2ULL * n) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(ds));
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
