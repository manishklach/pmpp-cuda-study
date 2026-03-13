// Example 012: RGB To Grayscale
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
__global__ void k(const uchar3 *in, unsigned char *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * w + x;
    uchar3 p = in[idx];
    float v = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
    out[idx] = (unsigned char)(v + 0.5f);
  }
}
int main() {
  const int w = 8, h = 8, c = w * h;
  size_t ib = (size_t)c * sizeof(uchar3), ob = (size_t)c * sizeof(unsigned char);
  std::vector<uchar3> in(c);
  std::vector<unsigned char> go(c, 0), co(c, 0);
  for (int i = 0; i < c; ++i) {
    in[i] = make_uchar3((i * 13) % 256, (40 + i * 7) % 256, (90 + i * 5) % 256);
    uchar3 p = in[i];
    float v = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
    co[i] = (unsigned char)(v + 0.5f);
  }
  uchar3 *di = nullptr;
  unsigned char *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, ib));
  CHECK_CUDA(cudaMalloc(&do_, ob));
  CHECK_CUDA(cudaMemcpy(di, in.data(), ib, cudaMemcpyHostToDevice));
  dim3 t(16, 16), b((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  k<<<b, t>>>(di, do_, w, h);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, ob, cudaMemcpyDeviceToHost));
  bool ok = go == co;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
