#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "093_pagerank";
constexpr float kDamping = 0.85f;
constexpr int kIterations = 8;

struct Graph {
  int nodes = 0;
  std::vector<int> src;
  std::vector<int> dst;
  std::vector<int> out_degree;
};

__global__ void pagerank_contrib_kernel(const int *src, const int *dst, const int *out_degree,
                                        const float *rank, float *next_rank, int edges) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= edges)
    return;
  int u = src[idx];
  int v = dst[idx];
  if (out_degree[u] > 0)
    atomicAdd(&next_rank[v], rank[u] / out_degree[u]);
}

__global__ void pagerank_finalize_kernel(float *next_rank, int nodes, float damping) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nodes)
    next_rank[idx] = (1.0f - damping) / nodes + damping * next_rank[idx];
}

Graph make_graph(int nodes) {
  nodes = std::max(5, nodes);
  Graph g;
  g.nodes = nodes;
  g.out_degree.assign(nodes, 0);
  for (int i = 0; i < nodes; ++i) {
    int a = (i + 1) % nodes;
    int b = (i + 2) % nodes;
    g.src.push_back(i);
    g.dst.push_back(a);
    ++g.out_degree[i];
    if (i % 2 == 0) {
      g.src.push_back(i);
      g.dst.push_back(b);
      ++g.out_degree[i];
    }
  }
  return g;
}

std::vector<float> cpu_reference(const Graph &g) {
  std::vector<float> rank(g.nodes, 1.0f / g.nodes);
  std::vector<float> next(g.nodes, 0.0f);
  for (int iter = 0; iter < kIterations; ++iter) {
    std::fill(next.begin(), next.end(), 0.0f);
    for (std::size_t e = 0; e < g.src.size(); ++e)
      next[g.dst[e]] += rank[g.src[e]] / g.out_degree[g.src[e]];
    for (int i = 0; i < g.nodes; ++i)
      next[i] = (1.0f - kDamping) / g.nodes + kDamping * next[i];
    rank.swap(next);
  }
  return rank;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  Graph g = make_graph(options.size);
  std::vector<float> cpu = cpu_reference(g);
  std::vector<float> rank(g.nodes, 1.0f / g.nodes);

  int *d_src = nullptr, *d_dst = nullptr, *d_out = nullptr;
  float *d_rank = nullptr, *d_next = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&d_src, g.src.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_dst, g.dst.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_out, g.nodes * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_rank, g.nodes * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_next, g.nodes * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(d_src, g.src.data(), g.src.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_dst, g.dst.data(), g.dst.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_out, g.out_degree.data(), g.nodes * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_rank, rank.data(), g.nodes * sizeof(float), cudaMemcpyHostToDevice));

  const int edge_blocks = (static_cast<int>(g.src.size()) + 255) / 256;
  const int node_blocks = (g.nodes + 255) / 256;
  for (int iter = 0; iter < kIterations; ++iter) {
    PMPP_CUDA_CHECK(cudaMemset(d_next, 0, g.nodes * sizeof(float)));
    pagerank_contrib_kernel<<<edge_blocks, 256>>>(d_src, d_dst, d_out, d_rank, d_next,
                                                  static_cast<int>(g.src.size()));
    pagerank_finalize_kernel<<<node_blocks, 256>>>(d_next, g.nodes, kDamping);
    PMPP_CUDA_KERNEL_CHECK();
    std::swap(d_rank, d_next);
  }

  PMPP_CUDA_CHECK(cudaMemcpy(rank.data(), d_rank, g.nodes * sizeof(float), cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(d_src));
  PMPP_CUDA_CHECK(cudaFree(d_dst));
  PMPP_CUDA_CHECK(cudaFree(d_out));
  PMPP_CUDA_CHECK(cudaFree(d_rank));
  PMPP_CUDA_CHECK(cudaFree(d_next));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, rank, 1.0e-5f);
  summary.notes = "This PageRank study uses an edge-parallel contribution pass plus a finalize pass.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  Graph g = make_graph(std::max(64, options.size));
  std::vector<float> rank(g.nodes, 1.0f / g.nodes);

  int *d_src = nullptr, *d_dst = nullptr, *d_out = nullptr;
  float *d_rank = nullptr, *d_next = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&d_src, g.src.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_dst, g.dst.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_out, g.nodes * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_rank, g.nodes * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_next, g.nodes * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(d_src, g.src.data(), g.src.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_dst, g.dst.data(), g.dst.size() * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_out, g.out_degree.data(), g.nodes * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_rank, rank.data(), g.nodes * sizeof(float), cudaMemcpyHostToDevice));

  const int edge_blocks = (static_cast<int>(g.src.size()) + 255) / 256;
  const int node_blocks = (g.nodes + 255) / 256;
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    for (int iter = 0; iter < kIterations; ++iter) {
      PMPP_CUDA_CHECK(cudaMemset(d_next, 0, g.nodes * sizeof(float)));
      pagerank_contrib_kernel<<<edge_blocks, 256>>>(d_src, d_dst, d_out, d_rank, d_next,
                                                    static_cast<int>(g.src.size()));
      pagerank_finalize_kernel<<<node_blocks, 256>>>(d_next, g.nodes, kDamping);
      PMPP_CUDA_KERNEL_CHECK();
      std::swap(d_rank, d_next);
    }
  });
  std::size_t bytes = g.src.size() * sizeof(int) + g.dst.size() * sizeof(int) +
                      g.out_degree.size() * sizeof(int) + 2 * g.nodes * sizeof(float);
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(g.nodes, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(d_src));
  PMPP_CUDA_CHECK(cudaFree(d_dst));
  PMPP_CUDA_CHECK(cudaFree(d_out));
  PMPP_CUDA_CHECK(cudaFree(d_rank));
  PMPP_CUDA_CHECK(cudaFree(d_next));
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
                                 "Nodes/sec");
  }
  return EXIT_SUCCESS;
}
