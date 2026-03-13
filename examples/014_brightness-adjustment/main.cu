// Example 014: Brightness Adjustment
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
__global__ void k(const uchar3 *in, uchar3 *out, int d, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * w + x;
    uchar3 p = in[idx];
    out[idx] = make_uchar3(clamp_to_byte((int)p.x + d), clamp_to_byte((int)p.y + d),
                           clamp_to_byte((int)p.z + d));
  }
}
int main() {
  const int w = 8, h = 8, d = 24, c = w * h;
  size_t bytes = (size_t)c * sizeof(uchar3);
  std::vector<uchar3> in(c), go(c), co(c);
  for (int i = 0; i < c; ++i) {
    in[i] = make_uchar3((i * 13) % 256, (40 + i * 7) % 256, (90 + i * 5) % 256);
    uchar3 p = in[i];
    co[i] = make_uchar3(clamp_to_byte((int)p.x + d), clamp_to_byte((int)p.y + d),
                        clamp_to_byte((int)p.z + d));
  }
  uchar3 *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  dim3 t(16, 16), b((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  k<<<b, t>>>(di, do_, d, w, h);
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
