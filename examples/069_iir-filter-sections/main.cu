// Example 069: IIR Filter Sections
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

__global__ void iir_channels_kernel(const float *input, float *output, int channels, int samples,
                                    float a, float b) {
  int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels)
    return;
  float prev_y = 0.0f;
  for (int t = 0; t < samples; ++t) {
    int idx = channel * samples + t;
    float y = a * input[idx] + b * prev_y;
    output[idx] = y;
    prev_y = y;
  }
}

int main() {
  const int channels = 8, samples = 64;
  const float a = 0.35f, b = 0.65f;
  std::vector<float> input(channels * samples), cpu(input.size(), 0.0f), gpu(input.size(), 0.0f);
  for (int c = 0; c < channels; ++c)
    for (int t = 0; t < samples; ++t)
      input[c * samples + t] = sinf(0.1f * t) + 0.1f * c;
  for (int c = 0; c < channels; ++c) {
    float prev_y = 0.0f;
    for (int t = 0; t < samples; ++t) {
      int idx = c * samples + t;
      float y = a * input[idx] + b * prev_y;
      cpu[idx] = y;
      prev_y = y;
    }
  }

  float *d_input = nullptr, *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, gpu.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  iir_channels_kernel<<<1, 64>>>(d_input, d_output, channels, samples, a, b);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Channel 0 last sample: " << gpu[samples - 1] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
