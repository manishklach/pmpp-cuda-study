// Example 086: Boids Flocking
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

__device__ float length_sq(Vec2 v) {
  return v.x * v.x + v.y * v.y;
}

__global__ void boids_kernel(const Vec2 *positions, const Vec2 *velocities, Vec2 *next_positions,
                             Vec2 *next_velocities, int n, float radius) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Vec2 align = {0.0f, 0.0f}, cohesion = {0.0f, 0.0f}, separation = {0.0f, 0.0f};
  int neighbors = 0;
  for (int j = 0; j < n; ++j) {
    if (i == j)
      continue;
    Vec2 delta = {positions[j].x - positions[i].x, positions[j].y - positions[i].y};
    if (length_sq(delta) <= radius * radius) {
      align.x += velocities[j].x;
      align.y += velocities[j].y;
      cohesion.x += positions[j].x;
      cohesion.y += positions[j].y;
      separation.x -= delta.x;
      separation.y -= delta.y;
      ++neighbors;
    }
  }
  Vec2 velocity = velocities[i];
  if (neighbors > 0) {
    float inv = 1.0f / neighbors;
    align.x = align.x * inv - velocity.x;
    align.y = align.y * inv - velocity.y;
    cohesion.x = cohesion.x * inv - positions[i].x;
    cohesion.y = cohesion.y * inv - positions[i].y;
    velocity.x += 0.05f * align.x + 0.02f * cohesion.x + 0.03f * separation.x;
    velocity.y += 0.05f * align.y + 0.02f * cohesion.y + 0.03f * separation.y;
  }
  next_velocities[i] = velocity;
  next_positions[i] = {positions[i].x + velocity.x, positions[i].y + velocity.y};
}

int main() {
  const int n = 6;
  const float radius = 2.0f;
  std::vector<Vec2> positions = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}};
  std::vector<Vec2> velocities = {{0.1f, 0.0f}, {0.1f, 0.0f}, {0.05f, 0.02f},
                                  {0.0f, 0.1f}, {0.0f, 0.1f}, {-0.02f, 0.08f}};
  std::vector<Vec2> cpu_pos(n), cpu_vel(n), gpu_pos(n), gpu_vel(n);
  for (int i = 0; i < n; ++i) {
    Vec2 align = {0.0f, 0.0f}, cohesion = {0.0f, 0.0f}, separation = {0.0f, 0.0f};
    int neighbors = 0;
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      Vec2 delta = {positions[j].x - positions[i].x, positions[j].y - positions[i].y};
      if (delta.x * delta.x + delta.y * delta.y <= radius * radius) {
        align.x += velocities[j].x;
        align.y += velocities[j].y;
        cohesion.x += positions[j].x;
        cohesion.y += positions[j].y;
        separation.x -= delta.x;
        separation.y -= delta.y;
        ++neighbors;
      }
    }
    Vec2 velocity = velocities[i];
    if (neighbors > 0) {
      float inv = 1.0f / neighbors;
      align.x = align.x * inv - velocity.x;
      align.y = align.y * inv - velocity.y;
      cohesion.x = cohesion.x * inv - positions[i].x;
      cohesion.y = cohesion.y * inv - positions[i].y;
      velocity.x += 0.05f * align.x + 0.02f * cohesion.x + 0.03f * separation.x;
      velocity.y += 0.05f * align.y + 0.02f * cohesion.y + 0.03f * separation.y;
    }
    cpu_vel[i] = velocity;
    cpu_pos[i] = {positions[i].x + velocity.x, positions[i].y + velocity.y};
  }

  Vec2 *d_pos = nullptr, *d_vel = nullptr, *d_next_pos = nullptr, *d_next_vel = nullptr;
  CHECK_CUDA(cudaMalloc(&d_pos, n * sizeof(Vec2)));
  CHECK_CUDA(cudaMalloc(&d_vel, n * sizeof(Vec2)));
  CHECK_CUDA(cudaMalloc(&d_next_pos, n * sizeof(Vec2)));
  CHECK_CUDA(cudaMalloc(&d_next_vel, n * sizeof(Vec2)));
  CHECK_CUDA(cudaMemcpy(d_pos, positions.data(), n * sizeof(Vec2), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_vel, velocities.data(), n * sizeof(Vec2), cudaMemcpyHostToDevice));
  boids_kernel<<<1, 64>>>(d_pos, d_vel, d_next_pos, d_next_vel, n, radius);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu_pos.data(), d_next_pos, n * sizeof(Vec2), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(gpu_vel.data(), d_next_vel, n * sizeof(Vec2), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu_pos[i].x - gpu_pos[i].x) > 1.0e-5f ||
        std::fabs(cpu_pos[i].y - gpu_pos[i].y) > 1.0e-5f ||
        std::fabs(cpu_vel[i].x - gpu_vel[i].x) > 1.0e-5f ||
        std::fabs(cpu_vel[i].y - gpu_vel[i].y) > 1.0e-5f)
      ok = false;
  std::cout << "Boid 0 next position: (" << gpu_pos[0].x << ", " << gpu_pos[0].y << ")"
            << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_pos));
  CHECK_CUDA(cudaFree(d_vel));
  CHECK_CUDA(cudaFree(d_next_pos));
  CHECK_CUDA(cudaFree(d_next_vel));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
