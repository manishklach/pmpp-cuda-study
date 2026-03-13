// Example 054: Jacobi Iteration
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

__global__ void jacobi_step_kernel(const float *a, const float *b, const float *x_old, float *x_new,
                                   int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    float sigma = 0.0f;
    for (int col = 0; col < n; ++col)
      if (col != row)
        sigma += a[row * n + col] * x_old[col];
    x_new[row] = (b[row] - sigma) / a[row * n + row];
  }
}
int main() {
  const int n = 4, iters = 10;
  std::vector<float> a = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8},
                     b = {6, 25, -11, 15}, x(n, 0.0f), next(n, 0.0f), cpu(n, 0.0f),
                     cpu_next(n, 0.0f), gpu(n, 0.0f);
  for (int it = 0; it < iters; ++it) {
    for (int r = 0; r < n; ++r) {
      float sigma = 0.0f;
      for (int c = 0; c < n; ++c)
        if (c != r)
          sigma += a[r * n + c] * cpu[c];
      cpu_next[r] = (b[r] - sigma) / a[r * n + r];
    }
    cpu = cpu_next;
  }
  float *da = nullptr, *db = nullptr, *dx = nullptr, *dn = nullptr;
  CHECK_CUDA(cudaMalloc(&da, a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db, b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dx, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dn, next.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  for (int it = 0; it < iters; ++it) {
    jacobi_step_kernel<<<1, 64>>>(da, db, dx, dn, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::swap(dx, dn);
  }
  CHECK_CUDA(cudaMemcpy(gpu.data(), dx, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-3f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dn));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
