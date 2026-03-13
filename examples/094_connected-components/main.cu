// Example 094: Connected Components
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

__global__ void propagate_labels_kernel(const int *u, const int *v, int edges, int *labels,
                                        int *changed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= edges)
    return;
  int a = u[idx];
  int b = v[idx];
  int min_label = min(labels[a], labels[b]);
  if (min_label < labels[a]) {
    atomicMin(&labels[a], min_label);
    *changed = 1;
  }
  if (min_label < labels[b]) {
    atomicMin(&labels[b], min_label);
    *changed = 1;
  }
}

int main() {
  std::vector<int> u = {0, 1, 3, 4, 5};
  std::vector<int> v = {1, 2, 4, 5, 3};
  const int nodes = 6;
  const int edges = static_cast<int>(u.size());
  std::vector<int> cpu(nodes), gpu(nodes);
  for (int i = 0; i < nodes; ++i)
    cpu[i] = i;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int e = 0; e < edges; ++e) {
      int min_label = std::min(cpu[u[e]], cpu[v[e]]);
      if (min_label < cpu[u[e]]) {
        cpu[u[e]] = min_label;
        changed = true;
      }
      if (min_label < cpu[v[e]]) {
        cpu[v[e]] = min_label;
        changed = true;
      }
    }
  }

  int *d_u = nullptr, *d_v = nullptr, *d_labels = nullptr, *d_changed = nullptr;
  CHECK_CUDA(cudaMalloc(&d_u, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_v, edges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_labels, nodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_changed, sizeof(int)));
  for (int i = 0; i < nodes; ++i)
    gpu[i] = i;
  CHECK_CUDA(cudaMemcpy(d_u, u.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, v.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_labels, gpu.data(), nodes * sizeof(int), cudaMemcpyHostToDevice));
  int host_changed = 1;
  while (host_changed) {
    host_changed = 0;
    CHECK_CUDA(cudaMemcpy(d_changed, &host_changed, sizeof(int), cudaMemcpyHostToDevice));
    propagate_labels_kernel<<<(edges + 255) / 256, 256>>>(d_u, d_v, edges, d_labels, d_changed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
  }
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_labels, nodes * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = cpu == gpu;
  std::cout << "Labels:";
  for (int value : gpu)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_u));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_labels));
  CHECK_CUDA(cudaFree(d_changed));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
