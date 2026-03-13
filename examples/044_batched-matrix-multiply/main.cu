// Example 044: Batched Matrix Multiply
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

__global__ void batched_matmul_kernel(const float *a, const float *b, float *c, int batch, int m,
                                      int n, int k) {
  int batch_id = blockIdx.z;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (batch_id < batch && row < m && col < n) {
    const float *a_batch = a + batch_id * m * k;
    const float *b_batch = b + batch_id * k * n;
    float *c_batch = c + batch_id * m * n;
    float sum = 0.0f;
    for (int e = 0; e < k; ++e)
      sum += a_batch[row * k + e] * b_batch[e * n + col];
    c_batch[row * n + col] = sum;
  }
}
int main() {
  const int batch = 3, m = 4, n = 4, k = 4;
  std::vector<float> a(batch * m * k), b(batch * k * n), gpu(batch * m * n, 0.0f),
      cpu(batch * m * n, 0.0f);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = (i % 5) + 1;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = (i % 7) + 1;
  for (int bt = 0; bt < batch; ++bt)
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < n; ++c)
        for (int e = 0; e < k; ++e)
          cpu[bt * m * n + r * n + c] += a[bt * m * k + r * k + e] * b[bt * k * n + e * n + c];
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK_CUDA(cudaMalloc(&da, a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db, b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dc, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((n + t.x - 1) / t.x, (m + t.y - 1) / t.y, batch);
  batched_matmul_kernel<<<bl, t>>>(da, db, dc, batch, m, n, k);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), dc, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
