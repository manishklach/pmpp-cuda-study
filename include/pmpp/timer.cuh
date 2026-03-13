#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

namespace pmpp {

class CudaTimer {
 public:
  CudaTimer() {
    PMPP_CUDA_CHECK(cudaEventCreate(&start_));
    PMPP_CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { PMPP_CUDA_CHECK(cudaEventRecord(start_)); }

  float stop_ms() {
    PMPP_CUDA_CHECK(cudaEventRecord(stop_));
    PMPP_CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    PMPP_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace pmpp
