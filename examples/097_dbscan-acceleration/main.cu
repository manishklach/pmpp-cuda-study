// Example 097: DBSCAN Acceleration
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

struct Point {
  float x;
  float y;
};

__global__ void neighbor_count_kernel(const Point *points, int *counts, int n, float eps2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  int count = 0;
  for (int j = 0; j < n; ++j) {
    float dx = points[idx].x - points[j].x;
    float dy = points[idx].y - points[j].y;
    if (dx * dx + dy * dy <= eps2)
      ++count;
  }
  counts[idx] = count;
}

int main() {
  std::vector<Point> points = {{0.0f, 0.0f}, {0.1f, 0.1f}, {0.2f, 0.0f},
                               {4.8f, 4.9f}, {5.0f, 5.1f}, {9.0f, 9.0f}};
  const int n = static_cast<int>(points.size());
  const float eps = 0.35f;
  const int min_pts = 3;
  std::vector<int> cpu_counts(n, 0), gpu_counts(n, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      float dx = points[i].x - points[j].x;
      float dy = points[i].y - points[j].y;
      if (dx * dx + dy * dy <= eps * eps)
        ++cpu_counts[i];
    }

  Point *d_points = nullptr;
  int *d_counts = nullptr;
  CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(Point)));
  CHECK_CUDA(cudaMalloc(&d_counts, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
  neighbor_count_kernel<<<1, 256>>>(d_points, d_counts, n, eps * eps);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu_counts.data(), d_counts, n * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = cpu_counts == gpu_counts;
  std::cout << "Core-point flags:";
  for (int value : gpu_counts)
    std::cout << " " << (value >= min_pts ? 1 : 0);
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_points));
  CHECK_CUDA(cudaFree(d_counts));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
