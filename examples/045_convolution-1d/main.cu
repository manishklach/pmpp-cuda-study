// Example 045: Convolution 1D
// Track: Linear Algebra
// Difficulty: Intermediate
// Status: Guided template

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status__ = (call);                                                                 \
    if (status__ != cudaSuccess) {                                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

// - Study focus: data layout
// - Study focus: memory reuse
// - Study focus: correctness against a CPU reference

__global__ void study_kernel(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

static void fill_input(std::vector<float> &values, float scale) {
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = scale * static_cast<float>((i % 17) - 8);
  }
}

static void cpu_reference(const std::vector<float> &a, const std::vector<float> &b,
                          std::vector<float> &out) {
  for (int i = 0; i < static_cast<int>(out.size()); ++i) {
    out[i] = a[i] + b[i];
  }
}

int main() {
  std::cout << "Running 045" << std::endl;

  const int n = 1 << 12;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> host_a(n), host_b(n), host_out(n, 0.0f), host_ref(n, 0.0f);
  fill_input(host_a, 1.0f);
  fill_input(host_b, 0.5f);
  cpu_reference(host_a, host_b, host_ref);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_out = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, bytes));
  CHECK_CUDA(cudaMalloc(&device_b, bytes));
  CHECK_CUDA(cudaMalloc(&device_out, bytes));
  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  study_kernel<<<blocks, threads>>>(device_a, device_b, device_out, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_out.data(), device_out, bytes, cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(host_out[i] - host_ref[i]) > 1.0e-4f) {
      ++mismatches;
    }
  }

  std::cout << "Blocks: " << blocks << ", Threads: " << threads << std::endl;
  std::cout << "Validation: " << (mismatches == 0 ? "PASS" : "UPDATE TEMPLATE LOGIC") << std::endl;

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_out));
  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Suggested next steps:
// 1. Replace study_kernel with the actual kernel for this algorithm.
// 2. Expand cpu_reference to match the real computation.
// 3. Add any extra buffers, atomics, scans, or shared-memory tiles you need.
// 4. Test on tiny deterministic inputs first.
// 5. Compare with CUDA libraries when the topic overlaps with one.
