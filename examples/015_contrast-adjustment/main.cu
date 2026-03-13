// Example 015: Contrast Adjustment
// Track: Foundations
// Difficulty: Beginner
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
__host__ __device__ unsigned char clamp_to_byte(int v) {
  return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}
__global__ void k(const uchar3 *in, uchar3 *out, float f, int c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < c) {
    uchar3 p = in[idx];
    out[idx] = make_uchar3(clamp_to_byte((int)(((float)p.x - 128.0f) * f + 128.0f + 0.5f)),
                           clamp_to_byte((int)(((float)p.y - 128.0f) * f + 128.0f + 0.5f)),
                           clamp_to_byte((int)(((float)p.z - 128.0f) * f + 128.0f + 0.5f)));
  }
}
int main() {
  const int c = 64;
  const float f = 1.2f;
  size_t bytes = (size_t)c * sizeof(uchar3);
  std::vector<uchar3> in(c), go(c), co(c);
  for (int i = 0; i < c; ++i) {
    in[i] = make_uchar3((30 + i * 9) % 256, (70 + i * 11) % 256, (120 + i * 5) % 256);
    uchar3 p = in[i];
    co[i] = make_uchar3(clamp_to_byte((int)(((float)p.x - 128.0f) * f + 128.0f + 0.5f)),
                        clamp_to_byte((int)(((float)p.y - 128.0f) * f + 128.0f + 0.5f)),
                        clamp_to_byte((int)(((float)p.z - 128.0f) * f + 128.0f + 0.5f)));
  }
  uchar3 *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  k<<<(c + 127) / 128, 128>>>(di, do_, f, c);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < c; ++i)
    if (go[i].x != co[i].x || go[i].y != co[i].y || go[i].z != co[i].z)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
