// Example 071: Peak Detection
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

__global__ void peak_mask_kernel(const float *signal, int *mask, int n, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  float left = idx > 0 ? signal[idx - 1] : signal[idx];
  float right = idx + 1 < n ? signal[idx + 1] : signal[idx];
  mask[idx] = (signal[idx] >= left && signal[idx] >= right && signal[idx] > threshold) ? 1 : 0;
}

int main() {
  std::vector<float> signal = {0.0f, 1.0f, 3.2f, 1.5f, 0.8f, 2.5f, 5.0f, 1.2f,
                               0.4f, 4.2f, 2.1f, 0.1f, 3.3f, 3.3f, 1.0f};
  const int n = static_cast<int>(signal.size());
  const float threshold = 2.0f;
  std::vector<int> cpu(n, 0), gpu(n, 0);
  for (int i = 0; i < n; ++i) {
    float left = i > 0 ? signal[i - 1] : signal[i];
    float right = i + 1 < n ? signal[i + 1] : signal[i];
    cpu[i] = (signal[i] >= left && signal[i] >= right && signal[i] > threshold) ? 1 : 0;
  }

  float *d_signal = nullptr;
  int *d_mask = nullptr;
  CHECK_CUDA(cudaMalloc(&d_signal, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_mask, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_signal, signal.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  peak_mask_kernel<<<1, 128>>>(d_signal, d_mask, n, threshold);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_mask, n * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = true;
  std::cout << "Peak indices:";
  for (int i = 0; i < n; ++i) {
    if (cpu[i] != gpu[i])
      ok = false;
    if (gpu[i])
      std::cout << " " << i;
  }
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_signal));
  CHECK_CUDA(cudaFree(d_mask));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
