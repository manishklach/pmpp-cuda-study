// Example 096: K Means Clustering
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

__global__ void assign_clusters_kernel(const Point *points, const Point *centroids, int *labels,
                                       int n, int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  float best_dist = 1.0e30f;
  int best = 0;
  for (int c = 0; c < k; ++c) {
    float dx = points[idx].x - centroids[c].x;
    float dy = points[idx].y - centroids[c].y;
    float dist = dx * dx + dy * dy;
    if (dist < best_dist) {
      best_dist = dist;
      best = c;
    }
  }
  labels[idx] = best;
}

__global__ void accumulate_centroids_kernel(const Point *points, const int *labels, float *sum_x,
                                            float *sum_y, int *counts, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  int label = labels[idx];
  atomicAdd(&sum_x[label], points[idx].x);
  atomicAdd(&sum_y[label], points[idx].y);
  atomicAdd(&counts[label], 1);
}

int main() {
  std::vector<Point> points = {{0.0f, 0.0f}, {0.2f, 0.1f},  {4.8f, 5.0f},
                               {5.2f, 4.9f}, {0.1f, -0.1f}, {5.1f, 5.3f}};
  const int n = static_cast<int>(points.size());
  const int k = 2;
  const int iterations = 3;
  std::vector<Point> cpu_centroids = {{0.0f, 0.0f}, {5.0f, 5.0f}}, gpu_centroids = cpu_centroids;
  std::vector<int> cpu_labels(n, 0), gpu_labels(n, 0);
  for (int iter = 0; iter < iterations; ++iter) {
    std::vector<float> sum_x(k, 0.0f), sum_y(k, 0.0f);
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
      float d0 = (points[i].x - cpu_centroids[0].x) * (points[i].x - cpu_centroids[0].x) +
                 (points[i].y - cpu_centroids[0].y) * (points[i].y - cpu_centroids[0].y);
      float d1 = (points[i].x - cpu_centroids[1].x) * (points[i].x - cpu_centroids[1].x) +
                 (points[i].y - cpu_centroids[1].y) * (points[i].y - cpu_centroids[1].y);
      cpu_labels[i] = d0 < d1 ? 0 : 1;
      sum_x[cpu_labels[i]] += points[i].x;
      sum_y[cpu_labels[i]] += points[i].y;
      counts[cpu_labels[i]] += 1;
    }
    for (int c = 0; c < k; ++c)
      if (counts[c] > 0) {
        cpu_centroids[c].x = sum_x[c] / counts[c];
        cpu_centroids[c].y = sum_y[c] / counts[c];
      }
  }

  Point *d_points = nullptr, *d_centroids = nullptr;
  int *d_labels = nullptr, *d_counts = nullptr;
  float *d_sum_x = nullptr, *d_sum_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(Point)));
  CHECK_CUDA(cudaMalloc(&d_centroids, k * sizeof(Point)));
  CHECK_CUDA(cudaMalloc(&d_labels, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_counts, k * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_sum_x, k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sum_y, k * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
  for (int iter = 0; iter < iterations; ++iter) {
    CHECK_CUDA(
        cudaMemcpy(d_centroids, gpu_centroids.data(), k * sizeof(Point), cudaMemcpyHostToDevice));
    assign_clusters_kernel<<<1, 256>>>(d_points, d_centroids, d_labels, n, k);
    CHECK_CUDA(cudaMemset(d_sum_x, 0, k * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sum_y, 0, k * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_counts, 0, k * sizeof(int)));
    accumulate_centroids_kernel<<<1, 256>>>(d_points, d_labels, d_sum_x, d_sum_y, d_counts, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> sum_x(k, 0.0f), sum_y(k, 0.0f);
    std::vector<int> counts(k, 0);
    CHECK_CUDA(cudaMemcpy(sum_x.data(), d_sum_x, k * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sum_y.data(), d_sum_y, k * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(counts.data(), d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));
    for (int c = 0; c < k; ++c)
      if (counts[c] > 0) {
        gpu_centroids[c].x = sum_x[c] / counts[c];
        gpu_centroids[c].y = sum_y[c] / counts[c];
      }
  }
  CHECK_CUDA(cudaMemcpy(gpu_labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (cpu_labels[i] != gpu_labels[i])
      ok = false;
  for (int c = 0; c < k; ++c)
    if (std::fabs(cpu_centroids[c].x - gpu_centroids[c].x) > 1.0e-5f ||
        std::fabs(cpu_centroids[c].y - gpu_centroids[c].y) > 1.0e-5f)
      ok = false;
  std::cout << "Centroids:";
  for (const auto &p : gpu_centroids)
    std::cout << " (" << p.x << ", " << p.y << ")";
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_points));
  CHECK_CUDA(cudaFree(d_centroids));
  CHECK_CUDA(cudaFree(d_labels));
  CHECK_CUDA(cudaFree(d_counts));
  CHECK_CUDA(cudaFree(d_sum_x));
  CHECK_CUDA(cudaFree(d_sum_y));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
