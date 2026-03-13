// Example 020: Matrix Transpose With Shared Memory
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
constexpr int TILE_DIM = 16;
__global__ void k(const float *in, float *out, int w, int h) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (x < w && y < h)
    tile[threadIdx.y][threadIdx.x] = in[y * w + x];
  __syncthreads();
  int tx = blockIdx.y * TILE_DIM + threadIdx.x;
  int ty = blockIdx.x * TILE_DIM + threadIdx.y;
  if (tx < h && ty < w)
    out[ty * h + tx] = tile[threadIdx.x][threadIdx.y];
}
int main() {
  const int w = 32, h = 24, c = w * h;
  size_t bytes = (size_t)c * sizeof(float);
  std::vector<float> in(c), go(c, 0.0f), co(c, 0.0f);
  for (int i = 0; i < c; ++i)
    in[i] = (float)(i + 1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      co[x * h + y] = in[y * w + x];
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  dim3 t(TILE_DIM, TILE_DIM), b((w + TILE_DIM - 1) / TILE_DIM, (h + TILE_DIM - 1) / TILE_DIM);
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
