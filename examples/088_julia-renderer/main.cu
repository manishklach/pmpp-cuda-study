// Example 088: Julia Renderer
// Track: Simulation
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
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

__global__ void julia_kernel(int *output, int width, int height, int max_iters, float c_re,
                             float c_im) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  float zr = -1.5f + 3.0f * x / (width - 1);
  float zi = -1.5f + 3.0f * y / (height - 1);
  int iter = 0;
  while (zr * zr + zi * zi <= 4.0f && iter < max_iters) {
    float next_zr = zr * zr - zi * zi + c_re;
    float next_zi = 2.0f * zr * zi + c_im;
    zr = next_zr;
    zi = next_zi;
    ++iter;
  }
  output[y * width + x] = iter;
}

int main() {
  const int width = 16, height = 12, max_iters = 64;
  const float c_re = -0.8f, c_im = 0.156f;
  std::vector<int> cpu(width * height, 0), gpu(width * height, 0);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float zr = -1.5f + 3.0f * x / (width - 1);
      float zi = -1.5f + 3.0f * y / (height - 1);
      int iter = 0;
      while (zr * zr + zi * zi <= 4.0f && iter < max_iters) {
        float next_zr = zr * zr - zi * zi + c_re;
        float next_zi = 2.0f * zr * zi + c_im;
        zr = next_zr;
        zi = next_zi;
        ++iter;
      }
      cpu[y * width + x] = iter;
    }

  int *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_output, gpu.size() * sizeof(int)));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  julia_kernel<<<blocks, threads>>>(d_output, width, height, max_iters, c_re, c_im);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (cpu[i] != gpu[i])
      ok = false;
  std::cout << "Center pixel iterations: " << gpu[(height / 2) * width + width / 2] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
