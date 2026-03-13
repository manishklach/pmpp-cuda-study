#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/report.cuh"

namespace {
constexpr const char *kExampleName = "039_merge-two-sorted-arrays";
__global__ void merge_kernel(const int *a, int na, const int *b, int nb, int *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = na + nb;
  if (idx < total) {
    int lo = max(0, idx - nb);
    int hi = min(idx, na);
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (a[mid - 1] > b[idx - mid])
        hi = mid - 1;
      else
        lo = mid;
    }
    int i = lo;
    int j = idx - i;
    int a_val = i < na ? a[i] : INT_MAX;
    int b_val = j < nb ? b[j] : INT_MAX;
    out[idx] = min(a_val, b_val);
  }
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int na = std::max(8, options.size / 2);
  int nb = std::max(8, options.size - na);
  std::vector<int> a(na), b(nb), cpu(na + nb), gpu(na + nb);
  for (int i = 0; i < na; ++i) a[i] = i * 2;
  for (int i = 0; i < nb; ++i) b[i] = i * 2 + 1;
  std::merge(a.begin(), a.end(), b.begin(), b.end(), cpu.begin());

  if (options.check) {
    int *da = nullptr, *db = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&da, na * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&db, nb * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, cpu.size() * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(da, a.data(), na * sizeof(int), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(db, b.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
    merge_kernel<<<(static_cast<int>(cpu.size()) + options.block_size - 1) / options.block_size, options.block_size>>>(da, na, db, nb, dout);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, cpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
    summary.notes = "This merge uses diagonal partitioning to assign output positions in parallel.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(da)); PMPP_CUDA_CHECK(cudaFree(db)); PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok) return EXIT_FAILURE;
  }

  if (options.bench) {
    int *da = nullptr, *db = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&da, na * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&db, nb * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, cpu.size() * sizeof(int)));
    PMPP_CUDA_CHECK(cudaMemcpy(da, a.data(), na * sizeof(int), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(db, b.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      merge_kernel<<<(static_cast<int>(cpu.size()) + options.block_size - 1) / options.block_size, options.block_size>>>(da, na, db, nb, dout);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps((a.size() + b.size() + cpu.size()) * sizeof(int), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(cpu.size(), stats.avg_ms);
    if (!options.verify) std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(da)); PMPP_CUDA_CHECK(cudaFree(db)); PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
