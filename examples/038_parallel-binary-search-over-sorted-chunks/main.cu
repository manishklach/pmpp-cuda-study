// Example 038: Parallel Binary Search Over Sorted Chunks

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

__global__ void batched_binary_search_kernel(const int *data, const int *queries, int *positions,
                                             int chunk_size, int total_size, int query_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < query_count) {
    int q = queries[idx];
    int chunk = idx % (total_size / chunk_size);
    int lo = chunk * chunk_size, hi = lo + chunk_size - 1, pos = -1;
    while (lo <= hi) {
      int mid = (lo + hi) / 2;
      int v = data[mid];
      if (v == q) {
        pos = mid;
        break;
      }
      if (v < q)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    positions[idx] = pos;
  }
}
int main() {
  const int chunk = 8, chunks = 4, total = chunk * chunks, qn = 8;
  std::vector<int> data(total), queries(qn), gpu(qn, -1), cpu(qn, -1);
  for (int c = 0; c < chunks; ++c)
    for (int i = 0; i < chunk; ++i)
      data[c * chunk + i] = c * 100 + i * 2;
  queries = {0, 6, 100, 108, 200, 214, 300, 314};
  for (int i = 0; i < qn; ++i) {
    int c = i % chunks;
    auto begin = data.begin() + c * chunk;
    auto end = begin + chunk;
    auto it = std::lower_bound(begin, end, queries[i]);
    cpu[i] = (it != end && *it == queries[i]) ? (int)(it - data.begin()) : -1;
  }
  int *dd = nullptr, *dq = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&dd, total * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dq, qn * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dp, qn * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dd, data.data(), total * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dq, queries.data(), qn * sizeof(int), cudaMemcpyHostToDevice));
  batched_binary_search_kernel<<<1, 128>>>(dd, dq, dp, chunk, total, qn);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), dp, qn * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dd));
  CHECK_CUDA(cudaFree(dq));
  CHECK_CUDA(cudaFree(dp));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
