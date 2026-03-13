#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace pmpp {

inline void cuda_check(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

inline void kernel_check(const char *file, int line) {
  cuda_check(cudaGetLastError(), file, line);
  cuda_check(cudaDeviceSynchronize(), file, line);
}

}  // namespace pmpp

#define PMPP_CUDA_CHECK(call) ::pmpp::cuda_check((call), __FILE__, __LINE__)
#define PMPP_CUDA_KERNEL_CHECK() ::pmpp::kernel_check(__FILE__, __LINE__)
