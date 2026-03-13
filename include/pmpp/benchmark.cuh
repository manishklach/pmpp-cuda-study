#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

#include "timer.cuh"
#include "types.cuh"

namespace pmpp {

inline BenchmarkStats run_benchmark_loop(int warmup, int iters,
                                         const std::function<void()> &body) {
  for (int i = 0; i < warmup; ++i)
    body();

  std::vector<float> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CudaTimer timer;
    timer.start();
    body();
    samples.push_back(timer.stop_ms());
  }

  BenchmarkStats stats{};
  stats.min_ms = std::numeric_limits<float>::max();
  for (float value : samples) {
    stats.avg_ms += value;
    stats.min_ms = std::min(stats.min_ms, value);
    stats.max_ms = std::max(stats.max_ms, value);
  }
  if (!samples.empty())
    stats.avg_ms /= static_cast<float>(samples.size());
  return stats;
}

inline double bandwidth_gbps(std::size_t bytes, float avg_ms) {
  if (avg_ms <= 0.0f)
    return 0.0;
  return static_cast<double>(bytes) / (avg_ms * 1.0e6);
}

inline double elements_per_second(std::size_t elements, float avg_ms) {
  if (avg_ms <= 0.0f)
    return 0.0;
  return static_cast<double>(elements) / (avg_ms * 1.0e-3);
}

}  // namespace pmpp
