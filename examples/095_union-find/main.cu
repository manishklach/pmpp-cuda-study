// Example 095: Union Find
// Track: Graph and ML
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
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

__device__ int find_root_device(int *parent, int x) {
  while (parent[x] != x)
    x = parent[x];
  return x;
}

__global__ void union_edges_kernel(const int *u, const int *v, int edges, int *parent) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= edges)
    return;
  int ru = find_root_device(parent, u[idx]);
  int rv = find_root_device(parent, v[idx]);
  if (ru != rv) {
    int high = max(ru, rv);
    int low = min(ru, rv);
    atomicMin(&parent[high], low);
  }
}

__global__ void compress_paths_kernel(int *parent, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    while (parent[idx] != parent[parent[idx]])
      parent[idx] = parent[parent[idx]];
}

int main() {
  std::vector<int> u = {0, 1, 3, 4};
  std::vector<int> v = {1, 2, 4, 5};
  const int nodes = 6;
  const int edges = static_cast<int>(u.size());
  std::vector<int> cpu(nodes), gpu(nodes);
  for (int i = 0; i < nodes; ++i) {
    cpu[i] = i;
    gpu[i] = i;
  }
  auto find_cpu = [&](int x) {
    while (cpu[x] != x)
      x = cpu[x];
    return x;
  };
  for (int iter = 0; iter < 4; ++iter) {
    for (int e = 0; e < edges; ++e) {
      int ru = find_cpu(u[e]);
      int rv = find_cpu(v[e]);
      if (ru != rv)
        cpu[std::max(ru, rv)] = std::min(ru, rv);
    }
    for (int i = 0; i < nodes; ++i)
      while (cpu[i] != cpu[cpu[i]])
        cpu[i] = cpu[cpu[i]];
  }

  int *d_u = nullptr, *d_v = nullptr, *d_parent = nullptr;
  CHECK_CUDA(cudaMalloc(&d_u, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_v, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_parent, nodes * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_u, u.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, v.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_parent, gpu.data(), nodes * sizeof(int), cudaMemcpyHostToDevice));
  for (int iter = 0; iter < 4; ++iter) {
    union_edges_kernel<<<(edges + 255) / 256, 256>>>(d_u, d_v, edges, d_parent);
    compress_paths_kernel<<<1, 256>>>(d_parent, nodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_parent, nodes * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = cpu == gpu;
  std::cout << "Parents:";
  for (int value : gpu)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_u));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_parent));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
