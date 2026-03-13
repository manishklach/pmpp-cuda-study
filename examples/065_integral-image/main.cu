// Example 065: Integral Image
// Track: Image and Signal
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
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

__global__ void integral_image_kernel(const float *input, float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  float sum = 0.0f;
  for (int yy = 0; yy <= y; ++yy)
    for (int xx = 0; xx <= x; ++xx)
      sum += input[yy * width + xx];
  output[y * width + x] = sum;
}

int main() {
  const int width = 6, height = 5;
  std::vector<float> input(width * height), cpu(width * height, 0.0f), gpu(width * height, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      input[y * width + x] = static_cast<float>((x + y) % 4 + 1);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int yy = 0; yy <= y; ++yy)
        for (int xx = 0; xx <= x; ++xx)
          sum += input[yy * width + xx];
      cpu[y * width + x] = sum;
    }

  float *d_input = nullptr, *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, gpu.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  integral_image_kernel<<<blocks, threads>>>(d_input, d_output, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Bottom-right integral value: " << gpu.back() << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
