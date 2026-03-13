// Example 093: PageRank
// Track: Graph and ML
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

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

int main() {
  std::vector<int> src = {0, 0, 1, 2, 2, 3, 4};
  std::vector<int> dst = {1, 2, 2, 0, 3, 2, 3};
  const int nodes = 5;
  const int edges = static_cast<int>(src.size());
  const float damping = 0.85f;
  const int iterations = 5;
  std::vector<int> out_degree(nodes, 0);
  for (int u : src)
    ++out_degree[u];

  std::vector<float> cpu(nodes, 1.0f / nodes), cpu_next(nodes, 0.0f);
  for (int iter = 0; iter < iterations; ++iter) {
    std::fill(cpu_next.begin(), cpu_next.end(), 0.0f);
    for (int e = 0; e < edges; ++e)
      cpu_next[dst[e]] += cpu[src[e]] / out_degree[src[e]];
    for (int i = 0; i < nodes; ++i)
      cpu_next[i] = (1.0f - damping) / nodes + damping * cpu_next[i];
    cpu.swap(cpu_next);
  }

  int *d_src = nullptr, *d_dst = nullptr, *d_out = nullptr;
  float *d_rank = nullptr, *d_next = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_dst, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_out, nodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_rank, nodes * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_next, nodes * sizeof(float)));
  std::vector<float> rank(nodes, 1.0f / nodes);
  CHECK_CUDA(cudaMemcpy(d_src, src.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dst, dst.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_out, out_degree.data(), nodes * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_rank, rank.data(), nodes * sizeof(float), cudaMemcpyHostToDevice));

  for (int iter = 0; iter < iterations; ++iter) {
    CHECK_CUDA(cudaMemset(d_next, 0, nodes * sizeof(float)));
    pagerank_contrib_kernel<<<(edges + 255) / 256, 256>>>(d_src, d_dst, d_out, d_rank, d_next,
                                                          edges);
    pagerank_finalize_kernel<<<1, 256>>>(d_next, nodes, damping);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::swap(d_rank, d_next);
  }
  rank.assign(nodes, 0.0f);
  CHECK_CUDA(cudaMemcpy(rank.data(), d_rank, nodes * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < nodes; ++i)
    if (std::fabs(cpu[i] - rank[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Ranks:";
  for (float value : rank)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_rank));
  CHECK_CUDA(cudaFree(d_next));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
