// Example 076: Monte Carlo Pi
// Track: Simulation
// Difficulty: Intermediate
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

__device__ unsigned int lcg_next(unsigned int &state) {
  state = 1664525u * state + 1013904223u;
  return state;
}

__device__ float uniform01(unsigned int &state) {
  return (lcg_next(state) & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

__global__ void monte_carlo_pi_kernel(int trials_per_thread, int *hits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int state = 1234567u + 747796405u * static_cast<unsigned int>(idx);
  int local_hits = 0;
  for (int i = 0; i < trials_per_thread; ++i) {
    float x = uniform01(state);
    float y = uniform01(state);
    if (x * x + y * y <= 1.0f)
      ++local_hits;
  }
  hits[idx] = local_hits;
}

int main() {
  const int threads = 256, blocks = 32, total_threads = threads * blocks, trials_per_thread = 1024;
  std::vector<int> hits(total_threads, 0);
  int *d_hits = nullptr;
  CHECK_CUDA(cudaMalloc(&d_hits, total_threads * sizeof(int)));
  monte_carlo_pi_kernel<<<blocks, threads>>>(trials_per_thread, d_hits);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(hits.data(), d_hits, total_threads * sizeof(int), cudaMemcpyDeviceToHost));

  long long total_hits = 0;
  for (int value : hits)
    total_hits += value;
  long long total_trials = static_cast<long long>(total_threads) * trials_per_thread;
  double estimate = 4.0 * static_cast<double>(total_hits) / static_cast<double>(total_trials);
  bool ok = std::fabs(estimate - 3.141592653589793) < 0.05;
  std::cout << "Pi estimate: " << estimate << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "CHECK SAMPLE COUNT") << std::endl;
  CHECK_CUDA(cudaFree(d_hits));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
