// Example 063: Template Matching
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

__global__ void ssd_template_kernel(const float *image, int width, int height, const float *templ,
                                    int templ_width, int templ_height, float *scores, int out_width,
                                    int out_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_width || y >= out_height)
    return;
  float ssd = 0.0f;
  for (int ty = 0; ty < templ_height; ++ty)
    for (int tx = 0; tx < templ_width; ++tx) {
      float diff = image[(y + ty) * width + (x + tx)] - templ[ty * templ_width + tx];
      ssd += diff * diff;
    }
  scores[y * out_width + x] = ssd;
}

int main() {
  const int width = 8, height = 8, templ_width = 3, templ_height = 3;
  const int out_width = width - templ_width + 1, out_height = height - templ_height + 1;
  std::vector<float> image(width * height, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      image[y * width + x] = static_cast<float>((x + 2 * y) % 9);
  std::vector<float> templ = {4.0f, 5.0f, 6.0f, 6.0f, 7.0f, 8.0f, 8.0f, 0.0f, 1.0f};
  for (int ty = 0; ty < templ_height; ++ty)
    for (int tx = 0; tx < templ_width; ++tx)
      image[(3 + ty) * width + (2 + tx)] = templ[ty * templ_width + tx];

  std::vector<float> cpu(out_width * out_height, 0.0f), gpu(out_width * out_height, 0.0f);
  for (int y = 0; y < out_height; ++y)
    for (int x = 0; x < out_width; ++x) {
      float ssd = 0.0f;
      for (int ty = 0; ty < templ_height; ++ty)
        for (int tx = 0; tx < templ_width; ++tx) {
          float diff = image[(y + ty) * width + (x + tx)] - templ[ty * templ_width + tx];
          ssd += diff * diff;
        }
      cpu[y * out_width + x] = ssd;
    }

  float *d_image = nullptr, *d_templ = nullptr, *d_scores = nullptr;
  CHECK_CUDA(cudaMalloc(&d_image, image.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_templ, templ.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scores, gpu.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_image, image.data(), image.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_templ, templ.data(), templ.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((out_width + threads.x - 1) / threads.x, (out_height + threads.y - 1) / threads.y);
  ssd_template_kernel<<<blocks, threads>>>(d_image, width, height, d_templ, templ_width,
                                           templ_height, d_scores, out_width, out_height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_scores, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  auto best_it = std::min_element(gpu.begin(), gpu.end());
  int best_index = static_cast<int>(best_it - gpu.begin());
  std::cout << "Best match at: (" << (best_index % out_width) << ", " << (best_index / out_width)
            << ")" << std::endl;
  std::cout << "Validation: " << (ok && *best_it == 0.0f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_image));
  CHECK_CUDA(cudaFree(d_templ));
  CHECK_CUDA(cudaFree(d_scores));
  return ok && *best_it == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
