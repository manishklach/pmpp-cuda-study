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

constexpr const char *kExampleName = "041_matrix-vector-multiply";
constexpr int kMaxThreads = 256;

int sanitize_block_size(int requested) {
  return std::max(32, std::min(requested, kMaxThreads));
}

__global__ void matvec_naive_kernel(const float *matrix, const float *vector, float *output, int rows,
                                    int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;

  float sum = 0.0f;
  // One thread computes one output row. Matrix reads are row-major within the thread, but
  // neighboring threads walk different rows, so cross-thread accesses are strided by `cols`.
  for (int col = 0; col < cols; ++col)
    sum += matrix[row * cols + col] * vector[col];

  output[row] = sum;
}

__global__ void matvec_cached_vector_kernel(const float *matrix, const float *vector, float *output,
                                            int rows, int cols) {
  __shared__ float vector_tile[kMaxThreads];
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int tile_start = 0; tile_start < cols; tile_start += blockDim.x) {
    int vector_index = tile_start + threadIdx.x;
    vector_tile[threadIdx.x] = vector_index < cols ? vector[vector_index] : 0.0f;
    __syncthreads();

    int tile_width = std::min(blockDim.x, cols - tile_start);
    if (row < rows) {
      for (int offset = 0; offset < tile_width; ++offset)
        sum += matrix[row * cols + tile_start + offset] * vector_tile[offset];
    }

    // This barrier keeps the next tile load from clobbering shared memory before every row
    // has finished consuming the current vector tile.
    __syncthreads();
  }

  if (row < rows)
    output[row] = sum;
}

std::vector<float> cpu_reference(const std::vector<float> &matrix, const std::vector<float> &vector,
                                 int rows, int cols) {
  std::vector<float> output(rows, 0.0f);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      output[row] += matrix[row * cols + col] * vector[col];
  return output;
}

std::vector<float> run_kernel(const std::vector<float> &matrix, const std::vector<float> &vector,
                              int rows, int cols, int block_size, bool use_cached_vector) {
  const std::size_t matrix_bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);
  const std::size_t vector_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * sizeof(float);
  std::vector<float> gpu(rows, 0.0f);

  float *device_matrix = nullptr;
  float *device_vector = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_matrix, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_vector, vector_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_matrix, matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_vector, vector.data(), vector_bytes, cudaMemcpyHostToDevice));

  const int blocks = std::max(1, (rows + block_size - 1) / block_size);
  if (use_cached_vector)
    matvec_cached_vector_kernel<<<blocks, block_size>>>(device_matrix, device_vector, device_output,
                                                        rows, cols);
  else
    matvec_naive_kernel<<<blocks, block_size>>>(device_matrix, device_vector, device_output, rows,
                                                cols);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, output_bytes, cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_matrix));
  PMPP_CUDA_CHECK(cudaFree(device_vector));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  return gpu;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int rows = options.size;
  const int cols = std::max(8, options.size / 2);
  const int block_size = sanitize_block_size(options.block_size);
  const std::vector<float> matrix =
      pmpp::make_uniform_floats(rows * cols, options.seed, -2.0f, 2.0f);
  const std::vector<float> vector =
      pmpp::make_uniform_floats(cols, options.seed + 1, -2.0f, 2.0f);
  const std::vector<float> cpu = cpu_reference(matrix, vector, rows, cols);
  const std::vector<float> gpu_naive = run_kernel(matrix, vector, rows, cols, block_size, false);
  const std::vector<float> gpu_cached = run_kernel(matrix, vector, rows, cols, block_size, true);

  pmpp::ValidationSummary naive_summary = pmpp::compare_vectors(cpu, gpu_naive, 1.0e-5f);
  pmpp::ValidationSummary cached_summary = pmpp::compare_vectors(cpu, gpu_cached, 1.0e-5f);

  pmpp::ValidationSummary summary = cached_summary;
  summary.ok = naive_summary.ok && cached_summary.ok;
  summary.mismatch_count = naive_summary.mismatch_count + cached_summary.mismatch_count;
  summary.max_abs_error = std::max(naive_summary.max_abs_error, cached_summary.max_abs_error);
  summary.notes =
      "Validated both the one-thread-per-row baseline and a cached-vector shared-memory variant.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int rows = options.size;
  const int cols = std::max(8, options.size / 2);
  const int block_size = sanitize_block_size(options.block_size);
  const int blocks = std::max(1, (rows + block_size - 1) / block_size);
  const std::size_t matrix_bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);
  const std::size_t vector_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * sizeof(float);

  const std::vector<float> matrix =
      pmpp::make_uniform_floats(rows * cols, options.seed, -2.0f, 2.0f);
  const std::vector<float> vector =
      pmpp::make_uniform_floats(cols, options.seed + 1, -2.0f, 2.0f);

  float *device_matrix = nullptr;
  float *device_vector = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_matrix, matrix_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_vector, vector_bytes));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
  PMPP_CUDA_CHECK(cudaMemcpy(device_matrix, matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_vector, vector.data(), vector_bytes, cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    matvec_cached_vector_kernel<<<blocks, block_size>>>(device_matrix, device_vector, device_output,
                                                        rows, cols);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(matrix_bytes + vector_bytes + output_bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(rows, stats.avg_ms);
  stats.problem_label = "Matrix elements";
  stats.problem_size = static_cast<std::size_t>(rows) * cols;

  PMPP_CUDA_CHECK(cudaFree(device_matrix));
  PMPP_CUDA_CHECK(cudaFree(device_vector));
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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Rows/sec");
  }

  return EXIT_SUCCESS;
}
