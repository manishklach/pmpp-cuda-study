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
constexpr const char *kExampleName = "021_dot-product";
constexpr int kMaxThreads = 256;

__global__ void dot_partials_kernel(const float *a, const float *b, float *partials, int n) {
  __shared__ float scratch[kMaxThreads];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  scratch[local] = global < n ? a[global] * b[global] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride)
      scratch[local] += scratch[local + stride];
    __syncthreads();
  }
  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

double run_gpu_once(const std::vector<float> &a, const std::vector<float> &b, int block_size) {
  int n = static_cast<int>(a.size());
  int blocks = (n + block_size - 1) / block_size;
  std::vector<float> partials(blocks, 0.0f);
  float *da = nullptr, *db = nullptr, *dp = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&dp, blocks * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  dot_partials_kernel<<<blocks, block_size>>>(da, db, dp, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(da));
  PMPP_CUDA_CHECK(cudaFree(db));
  PMPP_CUDA_CHECK(cudaFree(dp));
  return std::accumulate(partials.begin(), partials.end(), 0.0);
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  if (options.block_size > kMaxThreads)
    options.block_size = kMaxThreads;

  std::vector<float> a = pmpp::make_uniform_floats(options.size, options.seed, -2.0f, 2.0f);
  std::vector<float> b = pmpp::make_uniform_floats(options.size, options.seed + 1, -3.0f, 3.0f);

  if (options.check) {
    double cpu = 0.0;
    for (int i = 0; i < options.size; ++i)
      cpu += static_cast<double>(a[i]) * b[i];
    double gpu = run_gpu_once(a, b, options.block_size);
    pmpp::ValidationSummary summary = pmpp::compare_scalars(cpu, gpu, 1.0e-3);
    summary.notes = "Dot product combines elementwise multiplication with a sum reduction.";
    pmpp::print_validation_report(kExampleName, summary);
    if (!summary.ok)
      return EXIT_FAILURE;
  }

  if (options.bench) {
    float *da = nullptr, *db = nullptr, *dp = nullptr;
    int blocks = (options.size + options.block_size - 1) / options.block_size;
    PMPP_CUDA_CHECK(cudaMalloc(&da, options.size * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&db, options.size * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dp, blocks * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(da, a.data(), options.size * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(db, b.data(), options.size * sizeof(float), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      dot_partials_kernel<<<blocks, options.block_size>>>(da, db, dp, options.size);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps =
        pmpp::bandwidth_gbps((static_cast<std::size_t>(options.size) * 2 + blocks) * sizeof(float),
                             stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(options.size, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(da));
    PMPP_CUDA_CHECK(cudaFree(db));
    PMPP_CUDA_CHECK(cudaFree(dp));
  }

  return EXIT_SUCCESS;
}
