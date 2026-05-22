#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace schwarz {

inline double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    return dot(a.data(), b.data(), static_cast<int>(a.size()));
}

inline double norm2(const double* a, int n) {
    return std::sqrt(dot(a, a, n));
}

inline double norm2(const std::vector<double>& a) {
    return norm2(a.data(), static_cast<int>(a.size()));
}

inline void axpy(double alpha, const double* x, double* y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

inline void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    assert(x.size() == y.size());
    axpy(alpha, x.data(), y.data(), static_cast<int>(x.size()));
}

inline void scale(double alpha, double* x, int n) {
    for (int i = 0; i < n; ++i)
        x[i] *= alpha;
}

inline void scale(double alpha, std::vector<double>& x) {
    scale(alpha, x.data(), static_cast<int>(x.size()));
}

inline void copy_vec(const double* src, double* dst, int n) {
    std::copy(src, src + n, dst);
}

inline void fill_zero(double* x, int n) {
    std::fill(x, x + n, 0.0);
}

inline void fill_zero(std::vector<double>& x) {
    std::fill(x.begin(), x.end(), 0.0);
}

}  // namespace schwarz
