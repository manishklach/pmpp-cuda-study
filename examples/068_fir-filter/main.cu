// Example 068: FIR Filter
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

__global__ void fir_kernel(const float *signal, const float *taps, float *output, int n,
                           int tap_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  float sum = 0.0f;
  for (int k = 0; k < tap_count; ++k) {
    int sample = idx - k;
    if (sample >= 0)
      sum += signal[sample] * taps[k];
  }
  output[idx] = sum;
}

int main() {
  const int n = 128, tap_count = 5;
  std::vector<float> signal(n), taps = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f}, cpu(n, 0.0f), gpu(n, 0.0f);
  for (int i = 0; i < n; ++i)
    signal[i] = sinf(0.15f * i) + 0.25f * cosf(0.7f * i);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < tap_count; ++k)
      if (i - k >= 0)
        cpu[i] += signal[i - k] * taps[k];

  float *d_signal = nullptr, *d_taps = nullptr, *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_signal, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_taps, tap_count * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_signal, signal.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_taps, taps.data(), tap_count * sizeof(float), cudaMemcpyHostToDevice));
  fir_kernel<<<(n + 255) / 256, 256>>>(d_signal, d_taps, d_output, n, tap_count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Filtered sample[20]: " << gpu[20] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_signal));
  CHECK_CUDA(cudaFree(d_taps));
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
