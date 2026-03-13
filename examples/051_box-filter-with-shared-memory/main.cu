// Example 051: Box Filter With Shared Memory
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

constexpr int TILE = 8;
__global__ void box_filter_shared_kernel(const float *input, float *output, int w, int h) {
  __shared__ float tile[TILE + 2][TILE + 2];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * TILE + tx, y = blockIdx.y * TILE + ty;
  int sx = min(max(x - 1, 0), w - 1), sy = min(max(y - 1, 0), h - 1);
  tile[ty][tx] = input[sy * w + sx];
  if (tx < TILE && ty < TILE && x < w && y < h) {
    int cx = min(max(x, 0), w - 1), cy = min(max(y, 0), h - 1);
    tile[ty + 1][tx + 1] = input[cy * w + cx];
  }
  __syncthreads();
  if (tx < TILE && ty < TILE && x < w && y < h) {
    float sum = 0.0f;
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = 0; kx < 3; ++kx)
        sum += tile[ty + ky][tx + kx];
    output[y * w + x] = sum / 9.0f;
  }
}
int main() {
  const int w = 8, h = 8;
  std::vector<float> in(w * h), gpu(w * h, 0.0f), cpu(w * h, 0.0f);
  for (int i = 0; i < w * h; ++i)
    in[i] = (float)((i % 7) + 1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float sum = 0.0f;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          int sx = std::min(std::max(x + kx, 0), w - 1);
          int sy = std::min(std::max(y + ky, 0), h - 1);
          sum += in[sy * w + sx];
        }
      cpu[y * w + x] = sum / 9.0f;
    }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, in.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(TILE, TILE), bl((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
  box_filter_shared_kernel<<<bl, t>>>(di, do_, w, h);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-4f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
