#pragma once

#include <iostream>
#include <string>

#include "types.cuh"

namespace pmpp {

inline void print_validation_report(const std::string &example_name, const ValidationSummary &summary,
                                    const char *mode = "check") {
  std::cout << "Example: " << example_name << std::endl;
  std::cout << "Mode: " << mode << std::endl;
  std::cout << "Validation: " << (summary.ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "Max abs error: " << summary.max_abs_error << std::endl;
  std::cout << "Mismatches: " << summary.mismatch_count << std::endl;
  if (!summary.notes.empty())
    std::cout << "Notes: " << summary.notes << std::endl;
}

inline void print_benchmark_report(const std::string &example_name, const BenchmarkStats &stats,
                                   int warmup, int iters, const std::string &unit_label = "") {
  std::cout << "Example: " << example_name << std::endl;
  std::cout << "Mode: bench" << std::endl;
  if (!stats.problem_label.empty() && stats.problem_size > 0)
    std::cout << stats.problem_label << ": " << stats.problem_size << std::endl;
  std::cout << "Warmup iters: " << warmup << std::endl;
  std::cout << "Timed iters: " << iters << std::endl;
  std::cout << "Benchmark avg ms: " << stats.avg_ms << std::endl;
  std::cout << "Benchmark min ms: " << stats.min_ms << std::endl;
  std::cout << "Benchmark max ms: " << stats.max_ms << std::endl;
  if (stats.bandwidth_gbps > 0.0)
    std::cout << "Effective GB/s: " << stats.bandwidth_gbps << std::endl;
  if (stats.throughput > 0.0)
    std::cout << unit_label << ": " << stats.throughput << std::endl;
}

}  // namespace pmpp
