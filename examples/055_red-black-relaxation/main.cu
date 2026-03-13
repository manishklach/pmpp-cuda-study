// Example 055: Red Black Relaxation
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

__global__ void red_black_step_kernel(const float *input, float *output, int w, int h, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x + 1 < w && y > 0 && y + 1 < h && ((x + y) & 1) == color) {
    output[y * w + x] = 0.25f * (input[y * w + x - 1] + input[y * w + x + 1] +
                                 input[(y - 1) * w + x] + input[(y + 1) * w + x]);
  }
}
int main() {
  const int w = 8, h = 8;
  std::vector<float> grid(w * h, 0.0f), cpu(w * h, 0.0f), tmp(w * h, 0.0f), gpu(w * h, 0.0f);
  for (int x = 0; x < w; ++x) {
    grid[x] = 1.0f;
    grid[(h - 1) * w + x] = 1.0f;
    cpu[x] = 1.0f;
    cpu[(h - 1) * w + x] = 1.0f;
  }
  tmp = cpu;
  for (int color = 0; color < 2; ++color)
    for (int y = 1; y < h - 1; ++y)
      for (int x = 1; x < w - 1; ++x)
        if (((x + y) & 1) == color)
          tmp[y * w + x] = 0.25f * (cpu[y * w + x - 1] + cpu[y * w + x + 1] + cpu[(y - 1) * w + x] +
                                    cpu[(y + 1) * w + x]);
  cpu = tmp;
  float *d0 = nullptr, *d1 = nullptr;
  CHECK_CUDA(cudaMalloc(&d0, grid.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d1, grid.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d0, grid.data(), grid.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d1, grid.data(), grid.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  red_black_step_kernel<<<bl, t>>>(d0, d1, w, h, 0);
  red_black_step_kernel<<<bl, t>>>(d1, d1, w, h, 1);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d1, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-4f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d0));
  CHECK_CUDA(cudaFree(d1));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
