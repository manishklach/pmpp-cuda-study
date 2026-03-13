#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "types.cuh"

namespace pmpp {

inline ValidationSummary compare_vectors(const std::vector<float> &expected,
                                         const std::vector<float> &actual, float tolerance) {
  ValidationSummary summary{};
  if (expected.size() != actual.size()) {
    summary.ok = false;
    summary.mismatch_count = 1;
    return summary;
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double error =
        std::fabs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    if (error > summary.max_abs_error)
      summary.max_abs_error = error;
    if (error > tolerance) {
      summary.ok = false;
      ++summary.mismatch_count;
    }
  }
  return summary;
}

inline ValidationSummary compare_vectors(const std::vector<int> &expected,
                                         const std::vector<int> &actual) {
  ValidationSummary summary{};
  if (expected.size() != actual.size()) {
    summary.ok = false;
    summary.mismatch_count = 1;
    return summary;
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double error =
        std::fabs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    if (error > summary.max_abs_error)
      summary.max_abs_error = error;
    if (expected[i] != actual[i]) {
      summary.ok = false;
      ++summary.mismatch_count;
    }
  }
  return summary;
}

inline bool within_tolerance(double expected, double actual, double tolerance) {
  return std::fabs(expected - actual) <= tolerance;
}

inline ValidationSummary compare_scalars(double expected, double actual, double tolerance) {
  ValidationSummary summary{};
  summary.max_abs_error = std::fabs(expected - actual);
  summary.ok = summary.max_abs_error <= tolerance;
  summary.mismatch_count = summary.ok ? 0 : 1;
  return summary;
}

}  // namespace pmpp
