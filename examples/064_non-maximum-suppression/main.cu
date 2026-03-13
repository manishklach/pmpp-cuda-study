// Example 064: Non Maximum Suppression
// Track: Image and Signal
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__global__ void nms_kernel(const float *input, float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  float center = input[y * width + x];
  bool is_peak = true;
  for (int dy = -1; dy <= 1 && is_peak; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      int nx = x + dx;
      int ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height && input[ny * width + nx] > center) {
        is_peak = false;
        break;
      }
    }
  }
  output[y * width + x] = is_peak ? center : 0.0f;
}

int main() {
  const int width = 7, height = 6;
  std::vector<float> input = {0, 2, 1, 0, 1, 0, 0, 1, 5, 2, 0, 1, 6, 0, 0, 2, 3, 1, 0, 1, 0,
                              0, 1, 0, 0, 4, 2, 0, 0, 0, 1, 2, 0, 7, 0, 1, 0, 0, 1, 0, 0, 0};
  std::vector<float> cpu(input.size(), 0.0f), gpu(input.size(), 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float center = input[y * width + x];
      bool is_peak = true;
      for (int dy = -1; dy <= 1 && is_peak; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0)
            continue;
          int nx = x + dx;
          int ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height && input[ny * width + nx] > center)
            is_peak = false;
        }
      cpu[y * width + x] = is_peak ? center : 0.0f;
    }

  float *d_input = nullptr, *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, gpu.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  nms_kernel<<<blocks, threads>>>(d_input, d_output, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  int survivors = 0;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-6f)
      ok = false;
    if (gpu[i] > 0.0f)
      ++survivors;
  }
  std::cout << "Surviving peaks: " << survivors << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
