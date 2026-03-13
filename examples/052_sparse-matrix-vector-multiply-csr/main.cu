#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "052_sparse-matrix-vector-multiply-csr";

struct CsrProblem {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
  std::vector<float> x;
};

__global__ void spmv_csr_kernel(const int *row_ptr, const int *col_idx, const float *values,
                                const float *x, float *y, int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
      sum += values[jj] * x[col_idx[jj]];
    y[row] = sum;
  }
}

CsrProblem make_problem(int rows, unsigned int seed) {
  (void)seed;
  CsrProblem problem;
  problem.rows = rows;
  problem.cols = rows;
  problem.row_ptr.reserve(rows + 1);
  problem.row_ptr.push_back(0);
  for (int row = 0; row < rows; ++row) {
    if (row > 0) {
      problem.col_idx.push_back(row - 1);
      problem.values.push_back(-1.0f);
    }
    problem.col_idx.push_back(row);
    problem.values.push_back(4.0f + static_cast<float>(row % 5));
    if (row + 1 < rows) {
      problem.col_idx.push_back(row + 1);
      problem.values.push_back(-1.0f);
    }
    problem.row_ptr.push_back(static_cast<int>(problem.col_idx.size()));
  }
  problem.x.resize(rows, 0.0f);
  for (int i = 0; i < rows; ++i)
    problem.x[i] = 1.0f + static_cast<float>(i % 7) * 0.25f;
  return problem;
}

std::vector<float> cpu_reference(const CsrProblem &problem) {
  std::vector<float> y(problem.rows, 0.0f);
  for (int row = 0; row < problem.rows; ++row)
    for (int jj = problem.row_ptr[row]; jj < problem.row_ptr[row + 1]; ++jj)
      y[row] += problem.values[jj] * problem.x[problem.col_idx[jj]];
  return y;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  CsrProblem problem = make_problem(std::max(4, options.size), options.seed);
  std::vector<float> cpu = cpu_reference(problem);
  std::vector<float> gpu(problem.rows, 0.0f);

  int *device_row_ptr = nullptr;
  int *device_col_idx = nullptr;
  float *device_values = nullptr;
  float *device_x = nullptr;
  float *device_y = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_row_ptr, problem.row_ptr.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_col_idx, problem.col_idx.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_values, problem.values.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_x, problem.x.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_y, gpu.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_row_ptr, problem.row_ptr.data(),
                             problem.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_col_idx, problem.col_idx.data(),
                             problem.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_values, problem.values.data(),
                             problem.values.size() * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(
      cudaMemcpy(device_x, problem.x.data(), problem.x.size() * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (problem.rows + threads - 1) / threads;
  spmv_csr_kernel<<<blocks, threads>>>(device_row_ptr, device_col_idx, device_values, device_x,
                                       device_y, problem.rows);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(
      cudaMemcpy(gpu.data(), device_y, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_row_ptr));
  PMPP_CUDA_CHECK(cudaFree(device_col_idx));
  PMPP_CUDA_CHECK(cudaFree(device_values));
  PMPP_CUDA_CHECK(cudaFree(device_x));
  PMPP_CUDA_CHECK(cudaFree(device_y));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
  summary.notes = "One row is mapped to one thread, which keeps the CSR traversal straightforward for study.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  CsrProblem problem = make_problem(std::max(128, options.size), options.seed);

  int *device_row_ptr = nullptr;
  int *device_col_idx = nullptr;
  float *device_values = nullptr;
  float *device_x = nullptr;
  float *device_y = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_row_ptr, problem.row_ptr.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_col_idx, problem.col_idx.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_values, problem.values.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_x, problem.x.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_y, problem.rows * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_row_ptr, problem.row_ptr.data(),
                             problem.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_col_idx, problem.col_idx.data(),
                             problem.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_values, problem.values.data(),
                             problem.values.size() * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(
      cudaMemcpy(device_x, problem.x.data(), problem.x.size() * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (problem.rows + threads - 1) / threads;
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    spmv_csr_kernel<<<blocks, threads>>>(device_row_ptr, device_col_idx, device_values, device_x,
                                         device_y, problem.rows);
    PMPP_CUDA_KERNEL_CHECK();
  });
  std::size_t bytes = problem.row_ptr.size() * sizeof(int) + problem.col_idx.size() * sizeof(int) +
                      problem.values.size() * sizeof(float) + problem.x.size() * sizeof(float) +
                      static_cast<std::size_t>(problem.rows) * sizeof(float);
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(problem.rows, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_row_ptr));
  PMPP_CUDA_CHECK(cudaFree(device_col_idx));
  PMPP_CUDA_CHECK(cudaFree(device_values));
  PMPP_CUDA_CHECK(cudaFree(device_x));
  PMPP_CUDA_CHECK(cudaFree(device_y));
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
                                 "Rows/sec");
  }

  return EXIT_SUCCESS;
}
