// Example 018: Matrix Addition
// Track: Foundations
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status__ = (call);                                                                 \
    if (status__ != cudaSuccess) {                                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)
__global__ void k(const float *a, const float *b, float *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * w + x;
    out[idx] = a[idx] + b[idx];
  }
}
int main() {
  const int w = 32, h = 24, c = w * h;
  size_t bytes = (size_t)c * sizeof(float);
  std::vector<float> a(c), b(c), go(c, 0.0f), co(c, 0.0f);
  for (int i = 0; i < c; ++i) {
    a[i] = 0.5f * (i % 17);
    b[i] = (i % 9) - 4;
    co[i] = a[i] + b[i];
  }
  float *da = nullptr, *db = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&da, bytes));
  CHECK_CUDA(cudaMalloc(&db, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  k<<<bl, t>>>(da, db, do_, w, h);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  int mm = 0;
  for (int i = 0; i < c; ++i)
    if (fabs(go[i] - co[i]) > 1.0e-5f)
      ++mm;
  std::cout << "Validation: " << (mm == 0 ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(do_));
  return mm == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
