// Example 079: N Body Naive
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

struct Vec3 {
  float x;
  float y;
  float z;
};

__global__ void nbody_naive_kernel(const Vec3 *positions, Vec3 *accelerations, int n,
                                   float softening) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Vec3 acc = {0.0f, 0.0f, 0.0f};
  Vec3 pi = positions[i];
  for (int j = 0; j < n; ++j) {
    if (i == j)
      continue;
    Vec3 pj = positions[j];
    float dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
    float dist2 = dx * dx + dy * dy + dz * dz + softening;
    float inv = rsqrtf(dist2);
    float inv3 = inv * inv * inv;
    acc.x += dx * inv3;
    acc.y += dy * inv3;
    acc.z += dz * inv3;
  }
  accelerations[i] = acc;
}

int main() {
  const int n = 16;
  const float softening = 1.0e-3f;
  std::vector<Vec3> positions(n), cpu(n), gpu(n);
  for (int i = 0; i < n; ++i)
    positions[i] = {0.1f * i, 0.05f * (i % 5), 0.08f * (i % 3)};
  for (int i = 0; i < n; ++i) {
    Vec3 acc = {0.0f, 0.0f, 0.0f};
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      float dx = positions[j].x - positions[i].x, dy = positions[j].y - positions[i].y,
            dz = positions[j].z - positions[i].z;
      float dist2 = dx * dx + dy * dy + dz * dz + softening;
      float inv = 1.0f / std::sqrt(dist2);
      float inv3 = inv * inv * inv;
      acc.x += dx * inv3;
      acc.y += dy * inv3;
      acc.z += dz * inv3;
    }
    cpu[i] = acc;
  }

  Vec3 *d_positions = nullptr, *d_acc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_positions, n * sizeof(Vec3)));
  CHECK_CUDA(cudaMalloc(&d_acc, n * sizeof(Vec3)));
  CHECK_CUDA(cudaMemcpy(d_positions, positions.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
  nbody_naive_kernel<<<1, 64>>>(d_positions, d_acc, n, softening);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_acc, n * sizeof(Vec3), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu[i].x - gpu[i].x) > 1.0e-3f || std::fabs(cpu[i].y - gpu[i].y) > 1.0e-3f ||
        std::fabs(cpu[i].z - gpu[i].z) > 1.0e-3f)
      ok = false;
  std::cout << "Particle 0 acceleration: (" << gpu[0].x << ", " << gpu[0].y << ", " << gpu[0].z
            << ")" << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_positions));
  CHECK_CUDA(cudaFree(d_acc));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
