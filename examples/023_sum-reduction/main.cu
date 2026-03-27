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

constexpr const char *kExampleName = "023_sum-reduction";
constexpr int kMaxThreads = 256;

int sanitize_block_size(int requested) {
  int block_size = std::max(32, std::min(requested, kMaxThreads));
  int power_of_two = 1;
  while ((power_of_two << 1) <= block_size)
    power_of_two <<= 1;
  return power_of_two;
}

__global__ void reduce_interleaved_kernel(const float *input, float *partials, int n) {
  __shared__ float scratch[kMaxThreads];
  int local = threadIdx.x;
  int global = blockIdx.x * blockDim.x + local;

  // Each block reads one contiguous chunk, so the initial global-memory load is coalesced.
  scratch[local] = global < n ? input[global] : 0.0f;
  __syncthreads();

  // This classic interleaved pattern is easy to understand, but the modulo test causes
  // divergence as only some lanes participate at each stride.
  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    if ((local % (stride << 1)) == 0)
      scratch[local] += scratch[local + stride];
    __syncthreads();
  }

  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

__global__ void reduce_sequential_kernel(const float *input, float *partials, int n) {
  __shared__ float scratch[kMaxThreads];
  int local = threadIdx.x;
  int global = blockIdx.x * blockDim.x + local;

  scratch[local] = global < n ? input[global] : 0.0f;
  __syncthreads();

  // Sequential addressing keeps the active threads contiguous, which reduces divergence
  // and improves how the block walks shared memory compared with the interleaved baseline.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride)
      scratch[local] += scratch[local + stride];
    __syncthreads();
  }

  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

double cpu_reference(const std::vector<float> &input) {
  return std::accumulate(input.begin(), input.end(), 0.0);
}

double run_interleaved_once(const std::vector<float> &input, int block_size) {
  const int n = static_cast<int>(input.size());
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

  std::vector<float> partials(blocks, 0.0f);
  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  reduce_interleaved_kernel<<<blocks, block_size>>>(device_input, device_partials, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(
      cudaMemcpy(partials.data(), device_partials, partial_bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
  return std::accumulate(partials.begin(), partials.end(), 0.0);
}

double run_sequential_once(const std::vector<float> &input, int block_size) {
  const int n = static_cast<int>(input.size());
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

  std::vector<float> partials(blocks, 0.0f);
  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  reduce_sequential_kernel<<<blocks, block_size>>>(device_input, device_partials, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(
      cudaMemcpy(partials.data(), device_partials, partial_bytes, cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
  return std::accumulate(partials.begin(), partials.end(), 0.0);
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int block_size = sanitize_block_size(options.block_size);
  std::vector<float> input = pmpp::make_uniform_floats(options.size, options.seed, -1.0f, 1.0f);

  const double cpu_sum = cpu_reference(input);
  const double gpu_interleaved = run_interleaved_once(input, block_size);
  const double gpu_sequential = run_sequential_once(input, block_size);

  pmpp::ValidationSummary interleaved_summary =
      pmpp::compare_scalars(cpu_sum, gpu_interleaved, 1.0e-3);
  pmpp::ValidationSummary sequential_summary =
      pmpp::compare_scalars(cpu_sum, gpu_sequential, 1.0e-3);

  pmpp::ValidationSummary summary = sequential_summary;
  summary.ok = interleaved_summary.ok && sequential_summary.ok;
  summary.mismatch_count =
      interleaved_summary.mismatch_count + sequential_summary.mismatch_count;
  summary.max_abs_error =
      std::max(interleaved_summary.max_abs_error, sequential_summary.max_abs_error);
  summary.notes =
      "Validated both an interleaved baseline and a lower-divergence sequential-addressing "
      "reduction.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int block_size = sanitize_block_size(options.block_size);
  const int n = options.size;
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);
  std::vector<float> input = pmpp::make_uniform_floats(n, options.seed, -1.0f, 1.0f);

  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    reduce_sequential_kernel<<<blocks, block_size>>>(device_input, device_partials, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(input_bytes + partial_bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Input elements";
  stats.problem_size = static_cast<std::size_t>(n);

  PMPP_CUDA_CHECK(cudaFree(device_input));
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
