// Example 053: Sparse Matrix Dense Vector Multiply
// Difficulty: Advanced

// Track: Linear Algebra
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                       \
  do {                                                                                         \
    cudaError_t status__ = (call);                                                             \
    if (status__ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                      \
      std::exit(EXIT_FAILURE);                                                                 \
    }                                                                                          \
  } while (0)

__global__ void sparse_dense_kernel(const int *row_ptr, const int *col_idx, const float *values,
                                    const float *dense, float *out, int rows, int dense_cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    for (int dc = 0; dc < dense_cols; ++dc) {
      float sum = 0.0f;
      for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
        sum += values[jj] * dense[col_idx[jj] * dense_cols + dc];
      out[row * dense_cols + dc] = sum;
    }
  }
}
int main() {
  const int rows = 4, dense_cols = 3;
  std::vector<int> row_ptr = {0, 2, 4, 7, 8}, col_idx = {0, 2, 1, 3, 0, 2, 3, 1};
  std::vector<float> vals = {10, 2, 3, 9, 7, 8, 7, 5},
                     dense = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, gpu(rows * dense_cols, 0.0f),
                     cpu(rows * dense_cols, 0.0f);
  for (int r = 0; r < rows; ++r)
    for (int dc = 0; dc < dense_cols; ++dc)
      for (int jj = row_ptr[r]; jj < row_ptr[r + 1]; ++jj)
        cpu[r * dense_cols + dc] += vals[jj] * dense[col_idx[jj] * dense_cols + dc];
  int *dr = nullptr, *dcidx = nullptr;
  float *dv = nullptr, *dd = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&dr, row_ptr.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dcidx, col_idx.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dv, vals.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dd, dense.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dcidx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dv, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dd, dense.data(), dense.size() * sizeof(float), cudaMemcpyHostToDevice));
  sparse_dense_kernel<<<1, 64>>>(dr, dcidx, dv, dd, do_, rows, dense_cols);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dr));
  CHECK_CUDA(cudaFree(dcidx));
  CHECK_CUDA(cudaFree(dv));
  CHECK_CUDA(cudaFree(dd));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
