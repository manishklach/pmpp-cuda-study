// Example 078: Random Walk Simulation
// Track: Simulation
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__device__ unsigned int lcg_walk(unsigned int &state) {
  state = 1664525u * state + 1013904223u;
  return state;
}

__global__ void random_walk_kernel(int *positions, int walkers, int steps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= walkers)
    return;
  unsigned int state = 13579u + 17u * static_cast<unsigned int>(idx);
  int position = 0;
  for (int step = 0; step < steps; ++step)
    position += (lcg_walk(state) & 1u) ? 1 : -1;
  positions[idx] = position;
}

int main() {
  const int walkers = 4096, steps = 256;
  std::vector<int> positions(walkers, 0);
  int *d_positions = nullptr;
  CHECK_CUDA(cudaMalloc(&d_positions, walkers * sizeof(int)));
  random_walk_kernel<<<(walkers + 255) / 256, 256>>>(d_positions, walkers, steps);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(positions.data(), d_positions, walkers * sizeof(int), cudaMemcpyDeviceToHost));

  double mean = std::accumulate(positions.begin(), positions.end(), 0.0) / walkers;
  bool ok = std::fabs(mean) < 8.0;
  std::cout << "Mean final position: " << mean << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "CHECK RNG SAMPLE SIZE") << std::endl;
  CHECK_CUDA(cudaFree(d_positions));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
