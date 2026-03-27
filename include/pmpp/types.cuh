#pragma once

#include <cstddef>
#include <string>

namespace pmpp {

struct BenchmarkStats {
  float avg_ms = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  double bandwidth_gbps = 0.0;
  double throughput = 0.0;
  std::string problem_label;
  std::size_t problem_size = 0;
};

struct ValidationSummary {
  bool ok = true;
  double max_abs_error = 0.0;
  int mismatch_count = 0;
  std::string notes;
};

struct CommonOptions {
  bool check = true;
  bool bench = false;
  bool verify = false;
  int size = 1 << 12;
  int warmup = 5;
  int iters = 20;
  int block_size = 256;
  unsigned int seed = 1234;
};

}  // namespace pmpp
