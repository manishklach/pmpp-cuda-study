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
constexpr const char *kExampleName = "041_matrix-vector-multiply";
__global__ void matvec_kernel(const float *matrix, const float *vector, float *output, int rows,
                              int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col)
      sum += matrix[row * cols + col] * vector[col];
    output[row] = sum;
  }
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int rows = options.size;
  int cols = std::max(4, options.size / 2);
  std::vector<float> matrix = pmpp::make_uniform_floats(rows * cols, options.seed, -2.0f, 2.0f);
  std::vector<float> vector = pmpp::make_uniform_floats(cols, options.seed + 1, -2.0f, 2.0f);
  std::vector<float> cpu(rows, 0.0f), gpu(rows, 0.0f);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      cpu[r] += matrix[r * cols + c] * vector[c];

  if (options.check) {
    float *dm = nullptr, *dv = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&dm, matrix.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dv, vector.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(dm, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(dv, vector.data(), vector.size() * sizeof(float), cudaMemcpyHostToDevice));
    matvec_kernel<<<(rows + options.block_size - 1) / options.block_size, options.block_size>>>(dm, dv, dout, rows, cols);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
    summary.notes = "Matrix-vector multiply maps one row to one thread in this baseline implementation.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(dm));
    PMPP_CUDA_CHECK(cudaFree(dv));
    PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok)
      return EXIT_FAILURE;
  }
  if (options.bench) {
    float *dm = nullptr, *dv = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&dm, matrix.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dv, vector.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(dm, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(dv, vector.data(), vector.size() * sizeof(float), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      matvec_kernel<<<(rows + options.block_size - 1) / options.block_size, options.block_size>>>(dm, dv, dout, rows, cols);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps =
        pmpp::bandwidth_gbps((matrix.size() + vector.size() + gpu.size()) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(rows, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Rows/sec");
    PMPP_CUDA_CHECK(cudaFree(dm));
    PMPP_CUDA_CHECK(cudaFree(dv));
    PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
