// Example 077: Monte Carlo Option Pricing
// Track: Simulation
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

constexpr float kPi = 3.14159265358979323846f;

__device__ __host__ unsigned int lcg_step(unsigned int state) {
  return 1664525u * state + 1013904223u;
}

__device__ __host__ float uniform_host_device(unsigned int &state) {
  state = lcg_step(state);
  return ((state >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

__device__ __host__ float standard_normal(unsigned int &state) {
  float u1 = fmaxf(uniform_host_device(state), 1.0e-7f);
  float u2 = uniform_host_device(state);
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * kPi * u2);
}

__global__ void option_pricing_kernel(float *payoffs, int paths, float s0, float strike, float rate,
                                      float sigma, float maturity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;
  unsigned int state = 987654321u + 104729u * static_cast<unsigned int>(idx);
  float z = standard_normal(state);
  float drift = (rate - 0.5f * sigma * sigma) * maturity;
  float diffusion = sigma * sqrtf(maturity) * z;
  float st = s0 * expf(drift + diffusion);
  payoffs[idx] = fmaxf(st - strike, 0.0f);
}

int main() {
  const int paths = 16384;
  const float s0 = 100.0f, strike = 105.0f, rate = 0.03f, sigma = 0.2f, maturity = 1.0f;
  std::vector<float> gpu_payoffs(paths, 0.0f), cpu_payoffs(paths, 0.0f);
  for (int i = 0; i < paths; ++i) {
    unsigned int state = 987654321u + 104729u * static_cast<unsigned int>(i);
    float u1 = std::max(uniform_host_device(state), 1.0e-7f);
    float u2 = uniform_host_device(state);
    float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * kPi * u2);
    float drift = (rate - 0.5f * sigma * sigma) * maturity;
    float diffusion = sigma * std::sqrt(maturity) * z;
    float st = s0 * std::exp(drift + diffusion);
    cpu_payoffs[i] = std::max(st - strike, 0.0f);
  }

  float *d_payoffs = nullptr;
  CHECK_CUDA(cudaMalloc(&d_payoffs, paths * sizeof(float)));
  option_pricing_kernel<<<(paths + 255) / 256, 256>>>(d_payoffs, paths, s0, strike, rate, sigma,
                                                      maturity);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(gpu_payoffs.data(), d_payoffs, paths * sizeof(float), cudaMemcpyDeviceToHost));

  double cpu_price = std::accumulate(cpu_payoffs.begin(), cpu_payoffs.end(), 0.0) / paths;
  double gpu_price = std::accumulate(gpu_payoffs.begin(), gpu_payoffs.end(), 0.0) / paths;
  cpu_price *= std::exp(-rate * maturity);
  gpu_price *= std::exp(-rate * maturity);
  bool ok = std::fabs(cpu_price - gpu_price) < 1.0e-3;
  std::cout << "Option price estimate: " << gpu_price << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_payoffs));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
