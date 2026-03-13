// Example 019: Matrix Transpose Naive
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
__global__ void k(const float *in, float *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h)
    out[x * h + y] = in[y * w + x];
}
int main() {
  const int w = 7, h = 5, c = w * h;
  size_t bytes = (size_t)c * sizeof(float);
  std::vector<float> in(c), go(c, 0.0f), co(c, 0.0f);
  for (int i = 0; i < c; ++i)
    in[i] = (float)i;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      co[x * h + y] = in[y * w + x];
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  dim3 t(16, 16), b((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  k<<<b, t>>>(di, do_, w, h);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  bool ok = go == co;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
