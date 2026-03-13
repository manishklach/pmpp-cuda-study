// Example 066: Canny Pipeline Stages
// Track: Image and Signal
// Difficulty: Advanced
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

__global__ void blur3x3_kernel(const float *input, float *blurred, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  float sum = 0.0f;
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int nx = x + dx;
      int ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        sum += input[ny * width + nx];
        ++count;
      }
    }
  blurred[y * width + x] = sum / count;
}

__global__ void gradient_magnitude_kernel(const float *input, float *magnitude, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int left = max(x - 1, 0), right = min(x + 1, width - 1);
  int up = max(y - 1, 0), down = min(y + 1, height - 1);
  float gx = input[y * width + right] - input[y * width + left];
  float gy = input[down * width + x] - input[up * width + x];
  magnitude[y * width + x] = sqrtf(gx * gx + gy * gy);
}

__global__ void threshold_kernel(const float *input, float *output, int count, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count)
    output[idx] = input[idx] >= threshold ? 1.0f : 0.0f;
}

int main() {
  const int width = 8, height = 8, count = width * height;
  const float threshold = 3.0f;
  std::vector<float> input(count, 0.0f), blur_cpu(count, 0.0f), edge_cpu(count, 0.0f),
      edge_gpu(count, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      input[y * width + x] = x >= 4 ? 10.0f : 1.0f;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      int samples = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          int nx = x + dx;
          int ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            sum += input[ny * width + nx];
            ++samples;
          }
        }
      blur_cpu[y * width + x] = sum / samples;
    }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      int left = std::max(x - 1, 0), right = std::min(x + 1, width - 1);
      int up = std::max(y - 1, 0), down = std::min(y + 1, height - 1);
      float gx = blur_cpu[y * width + right] - blur_cpu[y * width + left];
      float gy = blur_cpu[down * width + x] - blur_cpu[up * width + x];
      edge_cpu[y * width + x] = std::sqrt(gx * gx + gy * gy) >= threshold ? 1.0f : 0.0f;
    }

  float *d_input = nullptr, *d_blur = nullptr, *d_mag = nullptr, *d_edges = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, count * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_blur, count * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_mag, count * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_edges, count * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  dim3 threads2d(16, 16);
  dim3 blocks2d((width + threads2d.x - 1) / threads2d.x, (height + threads2d.y - 1) / threads2d.y);
  blur3x3_kernel<<<blocks2d, threads2d>>>(d_input, d_blur, width, height);
  gradient_magnitude_kernel<<<blocks2d, threads2d>>>(d_blur, d_mag, width, height);
  threshold_kernel<<<(count + 255) / 256, 256>>>(d_mag, d_edges, count, threshold);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(edge_gpu.data(), d_edges, count * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  int edge_pixels = 0;
  for (int i = 0; i < count; ++i) {
    if (std::fabs(edge_cpu[i] - edge_gpu[i]) > 1.0e-5f)
      ok = false;
    if (edge_gpu[i] > 0.0f)
      ++edge_pixels;
  }
  std::cout << "Detected edge pixels: " << edge_pixels << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_blur));
  CHECK_CUDA(cudaFree(d_mag));
  CHECK_CUDA(cudaFree(d_edges));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
