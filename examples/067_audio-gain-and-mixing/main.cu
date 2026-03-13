// Example 067: Audio Gain And Mixing
// Track: Image and Signal
// Difficulty: Beginner
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

__global__ void mix_kernel(const float *a, const float *b, float *out, int n, float gain_a,
                           float gain_b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float mixed = gain_a * a[idx] + gain_b * b[idx];
    out[idx] = fminf(1.0f, fmaxf(-1.0f, mixed));
  }
}

int main() {
  const int n = 128;
  const float gain_a = 0.7f, gain_b = 0.4f;
  std::vector<float> a(n), b(n), cpu(n, 0.0f), gpu(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = 0.8f * sinf(0.1f * i);
    b[i] = 0.6f * cosf(0.07f * i);
    cpu[i] = std::max(-1.0f, std::min(1.0f, gain_a * a[i] + gain_b * b[i]));
  }
  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  mix_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_out, n, gain_a, gain_b);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-6f)
      ok = false;
  std::cout << "First mixed samples: " << gpu[0] << ", " << gpu[1] << ", " << gpu[2] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
