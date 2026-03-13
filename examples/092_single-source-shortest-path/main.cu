// Example 092: Single Source Shortest Path
// Track: Graph and ML
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__global__ void relax_edges_kernel(const int *src, const int *dst, const float *weight, int edges,
                                   float *distance) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= edges)
    return;
  int u = src[idx];
  int v = dst[idx];
  float candidate = distance[u] + weight[idx];
  if (distance[u] < 1.0e19f)
    atomicMin(reinterpret_cast<int *>(&distance[v]), __float_as_int(candidate));
}

int main() {
  std::vector<int> src = {0, 0, 1, 1, 2, 3, 4};
  std::vector<int> dst = {1, 2, 2, 3, 4, 4, 5};
  std::vector<float> weight = {1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 1.0f, 3.0f};
  const int nodes = 6;
  const int edges = static_cast<int>(src.size());
  std::vector<float> cpu(nodes, 1.0e20f), gpu(nodes, 1.0e20f);
  cpu[0] = 0.0f;
  for (int iter = 0; iter < nodes - 1; ++iter)
    for (int e = 0; e < edges; ++e)
      cpu[dst[e]] = std::min(cpu[dst[e]], cpu[src[e]] + weight[e]);

  int *d_src = nullptr, *d_dst = nullptr;
  float *d_weight = nullptr, *d_distance = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_dst, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_weight, edges * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_distance, nodes * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_src, src.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dst, dst.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), edges * sizeof(float), cudaMemcpyHostToDevice));
  gpu[0] = 0.0f;
  CHECK_CUDA(cudaMemcpy(d_distance, gpu.data(), nodes * sizeof(float), cudaMemcpyHostToDevice));
  for (int iter = 0; iter < nodes - 1; ++iter) {
    relax_edges_kernel<<<(edges + 255) / 256, 256>>>(d_src, d_dst, d_weight, edges, d_distance);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_distance, nodes * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < nodes; ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Distances:";
  for (float value : gpu)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_distance));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
