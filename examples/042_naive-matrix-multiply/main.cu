// Example 042: Naive Matrix Multiply
// Difficulty: Intermediate

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

__global__ void matmul_naive_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int e = 0; e < k; ++e)
      sum += a[row * k + e] * b[e * n + col];
    c[row * n + col] = sum;
  }
}
int main() {
  const int m = 8, n = 8, k = 8;
  std::vector<float> a(m * k), b(k * n), gpu(m * n, 0.0f), cpu(m * n, 0.0f);
  for (int i = 0; i < m * k; ++i)
    a[i] = (i % 5) + 1;
  for (int i = 0; i < k * n; ++i)
    b[i] = (i % 7) + 1;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c)
      for (int e = 0; e < k; ++e)
        cpu[r * n + c] += a[r * k + e] * b[e * n + c];
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK_CUDA(cudaMalloc(&da, a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db, b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dc, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((n + t.x - 1) / t.x, (m + t.y - 1) / t.y);
  matmul_naive_kernel<<<bl, t>>>(da, db, dc, m, n, k);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), dc, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < m * n; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
