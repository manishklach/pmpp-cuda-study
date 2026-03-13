// Example 083: Wave Equation Solver
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

__global__ void wave_step_kernel(const float *previous, const float *current, float *next, int n,
                                 float c2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  if (idx == 0 || idx == n - 1) {
    next[idx] = 0.0f;
    return;
  }
  next[idx] = 2.0f * current[idx] - previous[idx] +
              c2 * (current[idx - 1] - 2.0f * current[idx] + current[idx + 1]);
}

int main() {
  const int n = 32;
  const float c2 = 0.15f;
  std::vector<float> previous(n, 0.0f), current(n, 0.0f), cpu(n, 0.0f), gpu(n, 0.0f);
  current[n / 2] = 1.0f;
  previous[n / 2] = 1.0f;
  for (int i = 1; i < n - 1; ++i)
    cpu[i] = 2.0f * current[i] - previous[i] +
             c2 * (current[i - 1] - 2.0f * current[i] + current[i + 1]);

  float *d_prev = nullptr, *d_cur = nullptr, *d_next = nullptr;
  CHECK_CUDA(cudaMalloc(&d_prev, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_cur, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_next, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_prev, previous.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cur, current.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  wave_step_kernel<<<1, 128>>>(d_prev, d_cur, d_next, n, c2);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_next, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-6f)
      ok = false;
  std::cout << "Center after step: " << gpu[n / 2] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_prev));
  CHECK_CUDA(cudaFree(d_cur));
  CHECK_CUDA(cudaFree(d_next));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
