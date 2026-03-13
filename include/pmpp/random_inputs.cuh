#pragma once

#include <random>
#include <vector>

namespace pmpp {

inline std::vector<float> make_uniform_floats(int size, unsigned int seed, float low, float high) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> values(size);
  for (float &value : values)
    value = dist(rng);
  return values;
}

inline std::vector<int> make_uniform_ints(int size, unsigned int seed, int low, int high) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(low, high);
  std::vector<int> values(size);
  for (int &value : values)
    value = dist(rng);
  return values;
}

}  // namespace pmpp
