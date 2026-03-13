// Example 091: Parallel BFS
// Track: Graph and ML
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__global__ void bfs_expand_kernel(const int *offsets, const int *edges, const int *frontier,
                                  int frontier_size, int *next_frontier, int *next_size,
                                  int *visited, int *distance, int level) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= frontier_size)
    return;
  int node = frontier[idx];
  for (int e = offsets[node]; e < offsets[node + 1]; ++e) {
    int neighbor = edges[e];
    if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
      distance[neighbor] = level + 1;
      int slot = atomicAdd(next_size, 1);
      next_frontier[slot] = neighbor;
    }
  }
}

int main() {
  std::vector<int> offsets = {0, 2, 4, 6, 7, 8, 8};
  std::vector<int> edges = {1, 2, 0, 3, 0, 4, 5, 5};
  const int nodes = static_cast<int>(offsets.size()) - 1;
  std::vector<int> cpu_distance(nodes, -1), gpu_distance(nodes, -1);
  std::queue<int> q;
  q.push(0);
  cpu_distance[0] = 0;
  while (!q.empty()) {
    int node = q.front();
    q.pop();
    for (int e = offsets[node]; e < offsets[node + 1]; ++e) {
      int neighbor = edges[e];
      if (cpu_distance[neighbor] == -1) {
        cpu_distance[neighbor] = cpu_distance[node] + 1;
        q.push(neighbor);
      }
    }
  }

  int *d_offsets = nullptr, *d_edges = nullptr, *d_frontier = nullptr, *d_next_frontier = nullptr;
  int *d_next_size = nullptr, *d_visited = nullptr, *d_distance = nullptr;
  CHECK_CUDA(cudaMalloc(&d_offsets, offsets.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_edges, edges.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontier, nodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, nodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_next_size, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_visited, nodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_distance, nodes * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_edges, edges.data(), edges.size() * sizeof(int), cudaMemcpyHostToDevice));
  std::vector<int> visited(nodes, 0);
  visited[0] = 1;
  gpu_distance[0] = 0;
  CHECK_CUDA(cudaMemcpy(d_visited, visited.data(), nodes * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_distance, gpu_distance.data(), nodes * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<int> frontier = {0}, next_frontier(nodes, 0);
  int level = 0;
  while (!frontier.empty()) {
    int frontier_size = static_cast<int>(frontier.size());
    int next_size = 0;
    CHECK_CUDA(cudaMemcpy(d_frontier, frontier.data(), frontier_size * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_next_size, 0, sizeof(int)));
    bfs_expand_kernel<<<(frontier_size + 255) / 256, 256>>>(
        d_offsets, d_edges, d_frontier, frontier_size, d_next_frontier, d_next_size, d_visited,
        d_distance, level);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&next_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
    frontier.assign(next_size, 0);
    if (next_size > 0)
      CHECK_CUDA(cudaMemcpy(frontier.data(), d_next_frontier, next_size * sizeof(int),
                            cudaMemcpyDeviceToHost));
    ++level;
  }
  CHECK_CUDA(
      cudaMemcpy(gpu_distance.data(), d_distance, nodes * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = cpu_distance == gpu_distance;
  std::cout << "Distances:";
  for (int value : gpu_distance)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_offsets));
  CHECK_CUDA(cudaFree(d_edges));
  CHECK_CUDA(cudaFree(d_frontier));
  CHECK_CUDA(cudaFree(d_next_frontier));
  CHECK_CUDA(cudaFree(d_next_size));
  CHECK_CUDA(cudaFree(d_visited));
  CHECK_CUDA(cudaFree(d_distance));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
