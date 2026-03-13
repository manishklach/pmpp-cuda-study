// Example 080: N Body Tiled
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

__global__ void nbody_tiled_kernel(const Vec3 *positions, Vec3 *accelerations, int n,
                                   float softening) {
  __shared__ Vec3 tile[64];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Vec3 pi = positions[i];
  Vec3 acc = {0.0f, 0.0f, 0.0f};
  int tiles = (n + blockDim.x - 1) / blockDim.x;
  for (int tile_idx = 0; tile_idx < tiles; ++tile_idx) {
    int j = tile_idx * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = j < n ? positions[j] : Vec3{0.0f, 0.0f, 0.0f};
    __syncthreads();
    int limit = min(blockDim.x, n - tile_idx * blockDim.x);
    for (int k = 0; k < limit; ++k) {
      int global_j = tile_idx * blockDim.x + k;
      if (global_j == i)
        continue;
      float dx = tile[k].x - pi.x, dy = tile[k].y - pi.y, dz = tile[k].z - pi.z;
      float dist2 = dx * dx + dy * dy + dz * dz + softening;
      float inv = rsqrtf(dist2);
      float inv3 = inv * inv * inv;
      acc.x += dx * inv3;
      acc.y += dy * inv3;
      acc.z += dz * inv3;
    }
    __syncthreads();
  }
  accelerations[i] = acc;
}

int main() {
  const int n = 32;
  const float softening = 1.0e-3f;
  std::vector<Vec3> positions(n), cpu(n), gpu(n);
  for (int i = 0; i < n; ++i)
    positions[i] = {0.03f * i, 0.07f * (i % 7), 0.05f * (i % 4)};
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
  nbody_tiled_kernel<<<1, 32>>>(d_positions, d_acc, n, softening);
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
