// Example 056: Power Iteration
// Difficulty: Advanced

// Track: Linear Algebra
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                       \
  do {                                                                                         \
    cudaError_t status__ = (call);                                                             \
    if (status__ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                      \
      std::exit(EXIT_FAILURE);                                                                 \
    }                                                                                          \
  } while (0)

__global__ void matvec_power_kernel(const float *matrix, const float *x, float *y, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    float sum = 0.0f;
    for (int col = 0; col < n; ++col)
      sum += matrix[row * n + col] * x[col];
    y[row] = sum;
  }
}
int main() {
  const int n = 3, iters = 8;
  std::vector<float> A = {4, 1, 1, 1, 3, 0, 1, 0, 2}, x(n, 1.0f), y(n, 0.0f), cpu_x(n, 1.0f),
                     cpu_y(n, 0.0f), gpu(n, 0.0f);
  for (int it = 0; it < iters; ++it) {
    for (int r = 0; r < n; ++r) {
      cpu_y[r] = 0.0f;
      for (int c = 0; c < n; ++c)
        cpu_y[r] += A[r * n + c] * cpu_x[c];
    }
    float norm = 0.0f;
    for (float v : cpu_y)
      norm += v * v;
    norm = std::sqrt(norm);
    for (int i = 0; i < n; ++i)
      cpu_x[i] = cpu_y[i] / norm;
  }
  float *dA = nullptr, *dx = nullptr, *dy = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dx, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, y.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  for (int it = 0; it < iters; ++it) {
    matvec_power_kernel<<<1, 64>>>(dA, dx, dy, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(gpu.data(), dy, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float norm = 0.0f;
    for (float v : gpu)
      norm += v * v;
    norm = std::sqrt(norm);
    for (float &v : gpu)
      v /= norm;
    CHECK_CUDA(cudaMemcpy(dx, gpu.data(), gpu.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(gpu[i] - cpu_x[i]) > 1e-3f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
