#include <cuda_runtime.h>

#include <algorithm>
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

int sanitize_block_size(int requested) {
  return std::max(32, std::min(requested, kMaxThreads));
}

__global__ void elementwise_product_kernel(const float *a, const float *b, float *products, int n) {
  // Each thread maps to one vector element. The access pattern is perfectly coalesced because
  // neighboring threads read neighboring entries from both inputs and write neighboring products.
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    products[index] = a[index] * b[index];
}

__global__ void dot_block_reduce_kernel(const float *a, const float *b, float *partials, int n) {
  __shared__ float scratch[kMaxThreads];

  int local = threadIdx.x;
  int global = blockIdx.x * blockDim.x + local;
  int stride = blockDim.x * gridDim.x;

  float thread_sum = 0.0f;
  // Grid-stride iteration lets a modest launch cover large vectors while still preserving
  // contiguous accesses inside each warp on every pass through the loop.
  for (int index = global; index < n; index += stride)
    thread_sum += a[index] * b[index];

  scratch[local] = thread_sum;
  __syncthreads();

  // This tree reduction turns many thread-local products into one block partial.
  // Threads drop out as stride shrinks, so later stages have less parallel work.
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (local < offset)
      scratch[local] += scratch[local + offset];
    __syncthreads();
  }

  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

double cpu_reference(const std::vector<float> &a, const std::vector<float> &b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return sum;
}

double run_product_then_host_sum(const std::vector<float> &a, const std::vector<float> &b,
                                 int block_size) {
  const int n = static_cast<int>(a.size());
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  const int blocks = (n + block_size - 1) / block_size;

  std::vector<float> products(n, 0.0f);
  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_products = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_products, bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

  elementwise_product_kernel<<<blocks, block_size>>>(device_a, device_b, device_products, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(products.data(), device_products, bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_products));
  return std::accumulate(products.begin(), products.end(), 0.0);
}

double run_block_reduction(const std::vector<float> &a, const std::vector<float> &b, int block_size) {
  const int n = static_cast<int>(a.size());
  const std::size_t vector_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

  std::vector<float> partials(blocks, 0.0f);
  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, vector_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, vector_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), vector_bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), vector_bytes, cudaMemcpyHostToDevice));

  dot_block_reduce_kernel<<<blocks, block_size>>>(device_a, device_b, device_partials, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(
      cudaMemcpy(partials.data(), device_partials, partial_bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
  return std::accumulate(partials.begin(), partials.end(), 0.0);
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int block_size = sanitize_block_size(options.block_size);
  std::vector<float> a = pmpp::make_uniform_floats(options.size, options.seed, -2.0f, 2.0f);
  std::vector<float> b = pmpp::make_uniform_floats(options.size, options.seed + 1, -3.0f, 3.0f);

  const double cpu = cpu_reference(a, b);
  const double gpu_baseline = run_product_then_host_sum(a, b, block_size);
  const double gpu_reduced = run_block_reduction(a, b, block_size);

  pmpp::ValidationSummary baseline_summary = pmpp::compare_scalars(cpu, gpu_baseline, 1.0e-3);
  pmpp::ValidationSummary reduced_summary = pmpp::compare_scalars(cpu, gpu_reduced, 1.0e-3);

  pmpp::ValidationSummary summary = reduced_summary;
  summary.ok = baseline_summary.ok && reduced_summary.ok;
  summary.mismatch_count = baseline_summary.mismatch_count + reduced_summary.mismatch_count;
  summary.max_abs_error =
      std::max(baseline_summary.max_abs_error, reduced_summary.max_abs_error);
  summary.notes =
      "Validated both a map-then-host-sum baseline and a shared-memory block reduction.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int block_size = sanitize_block_size(options.block_size);
  const int n = options.size;
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  const std::size_t vector_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

  std::vector<float> a = pmpp::make_uniform_floats(n, options.seed, -2.0f, 2.0f);
  std::vector<float> b = pmpp::make_uniform_floats(n, options.seed + 1, -3.0f, 3.0f);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_a, vector_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_b, vector_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_a, a.data(), vector_bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_b, b.data(), vector_bytes, cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    dot_block_reduce_kernel<<<blocks, block_size>>>(device_a, device_b, device_partials, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(vector_bytes * 2 + partial_bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Input elements";
  stats.problem_size = static_cast<std::size_t>(n);

  PMPP_CUDA_CHECK(cudaFree(device_a));
  PMPP_CUDA_CHECK(cudaFree(device_b));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
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
