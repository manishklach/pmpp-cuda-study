// Example 081: Lennard Jones Forces
// Track: Simulation
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

struct Vec2 {
  float x;
  float y;
};

__global__ void lennard_jones_kernel(const Vec2 *positions, Vec2 *forces, int n, float epsilon,
                                     float sigma) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Vec2 force = {0.0f, 0.0f};
  for (int j = 0; j < n; ++j) {
    if (i == j)
      continue;
    float dx = positions[j].x - positions[i].x;
    float dy = positions[j].y - positions[i].y;
    float r2 = dx * dx + dy * dy + 1.0e-4f;
    float inv_r2 = 1.0f / r2;
    float sr2 = sigma * sigma * inv_r2;
    float sr6 = sr2 * sr2 * sr2;
    float sr12 = sr6 * sr6;
    float magnitude = 24.0f * epsilon * (2.0f * sr12 - sr6) * inv_r2;
    force.x += magnitude * dx;
    force.y += magnitude * dy;
  }
  forces[i] = force;
}

int main() {
  const int n = 8;
  const float epsilon = 0.5f;
  const float sigma = 1.0f;
  std::vector<Vec2> positions(n), cpu(n), gpu(n);
  for (int i = 0; i < n; ++i)
    positions[i] = {0.6f * i, 0.3f * (i % 3)};
  for (int i = 0; i < n; ++i) {
    Vec2 force = {0.0f, 0.0f};
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      float dx = positions[j].x - positions[i].x;
      float dy = positions[j].y - positions[i].y;
      float r2 = dx * dx + dy * dy + 1.0e-4f;
      float inv_r2 = 1.0f / r2;
      float sr2 = sigma * sigma * inv_r2;
      float sr6 = sr2 * sr2 * sr2;
      float sr12 = sr6 * sr6;
      float magnitude = 24.0f * epsilon * (2.0f * sr12 - sr6) * inv_r2;
      force.x += magnitude * dx;
      force.y += magnitude * dy;
    }
    cpu[i] = force;
  }

  Vec2 *d_pos = nullptr, *d_force = nullptr;
  CHECK_CUDA(cudaMalloc(&d_pos, n * sizeof(Vec2)));
  CHECK_CUDA(cudaMalloc(&d_force, n * sizeof(Vec2)));
  CHECK_CUDA(cudaMemcpy(d_pos, positions.data(), n * sizeof(Vec2), cudaMemcpyHostToDevice));
  lennard_jones_kernel<<<1, 64>>>(d_pos, d_force, n, epsilon, sigma);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_force, n * sizeof(Vec2), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu[i].x - gpu[i].x) > 1.0e-3f || std::fabs(cpu[i].y - gpu[i].y) > 1.0e-3f)
      ok = false;
  std::cout << "Particle 0 force: (" << gpu[0].x << ", " << gpu[0].y << ")" << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_pos));
  CHECK_CUDA(cudaFree(d_force));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
