#pragma once

#include <cstring>
#include <string>

#include "types.cuh"

namespace pmpp {

inline bool has_flag(int argc, char **argv, const char *flag) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0)
      return true;
  return false;
}

inline bool parse_int_arg(int argc, char **argv, const char *flag, int &value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], flag) == 0) {
      value = std::stoi(argv[i + 1]);
      return true;
    }
  }
  return false;
}

inline bool parse_uint_arg(int argc, char **argv, const char *flag, unsigned int &value) {
  int temp = static_cast<int>(value);
  if (!parse_int_arg(argc, argv, flag, temp))
    return false;
  value = static_cast<unsigned int>(temp);
  return true;
}

inline CommonOptions parse_common_options(int argc, char **argv) {
  CommonOptions options{};
  options.bench = has_flag(argc, argv, "--bench");
  options.verify = has_flag(argc, argv, "--verify");
  options.check = has_flag(argc, argv, "--check") || !options.bench;
  parse_int_arg(argc, argv, "--size", options.size);
  parse_int_arg(argc, argv, "--warmup", options.warmup);
  parse_int_arg(argc, argv, "--iters", options.iters);
  parse_int_arg(argc, argv, "--block-size", options.block_size);
  parse_uint_arg(argc, argv, "--seed", options.seed);
  return options;
}

}  // namespace pmpp
