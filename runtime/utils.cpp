//===- utils.cpp - DerivLab Runtime Utilities --------------------*- C++ -*-===//
//
//  Utility functions for DerivLab runtime.
//===----------------------------------------------------------------------===//

#include <vector>
#include <random>

namespace derivlab {

// Generate a vector of normal random numbers
std::vector<double> generate_normal(size_t n, double mean = 0.0, double stddev = 1.0) {
    std::default_random_engine gen;
    std::normal_distribution<double> dist(mean, stddev);
    std::vector<double> result(n);
    for (auto &x : result) x = dist(gen);
    return result;
}

// Compute the mean of a vector
inline double mean(const std::vector<double> &v) {
    double sum = 0.0;
    for (double x : v) sum += x;
    return v.empty() ? 0.0 : sum / v.size();
}

} // namespace derivlab
