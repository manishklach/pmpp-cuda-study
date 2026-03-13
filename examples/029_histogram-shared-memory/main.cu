// Example 029: Histogram Shared Memory

// Track: Parallel Patterns
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
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

__global__ void histogram_shared_kernel(const unsigned int *input, unsigned int *bins, int n,
                                        int num_bins) {
  __shared__ unsigned int local_bins[16];
  int tid = threadIdx.x;
  if (tid < num_bins)
    local_bins[tid] = 0;
  __syncthreads();
  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n)
    atomicAdd(&local_bins[input[idx] % num_bins], 1u);
  __syncthreads();
  if (tid < num_bins)
    atomicAdd(&bins[tid], local_bins[tid]);
}
int main() {
  const int n = 2048, num_bins = 16;
  std::vector<unsigned int> input(n), gpu(num_bins, 0), cpu(num_bins, 0);
  for (int i = 0; i < n; ++i) {
    input[i] = ((i * 7) + (i / 5)) % num_bins;
    ++cpu[input[i]];
  }
  unsigned int *di = nullptr, *db = nullptr;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc(&db, num_bins * sizeof(unsigned int)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(db, 0, num_bins * sizeof(unsigned int)));
  histogram_shared_kernel<<<(n + 255) / 256, 256>>>(di, db, n, num_bins);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), db, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(db));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
