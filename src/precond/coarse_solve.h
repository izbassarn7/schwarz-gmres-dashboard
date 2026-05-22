#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace schwarz {

// Dense direct solver for the coarse problem A_0 * x = b.
// Uses LU factorization with partial pivoting (LAPACK-style).
// Suitable for small coarse systems (n_coarse = number of subdomains).
class CoarseSolver {
public:
    void setup(const std::vector<double>& A0, int n) {
        n_ = n;
        LU_ = A0;
        piv_.resize(n);

        // LU factorization with partial pivoting
        for (int k = 0; k < n_; ++k) {
            // Find pivot
            int max_row = k;
            double max_val = std::abs(LU_[k * n_ + k]);
            for (int i = k + 1; i < n_; ++i) {
                double v = std::abs(LU_[i * n_ + k]);
                if (v > max_val) { max_val = v; max_row = i; }
            }
            piv_[k] = max_row;

            if (max_val < 1e-30)
                throw std::runtime_error("Coarse matrix is singular");

            // Swap rows
            if (max_row != k) {
                for (int j = 0; j < n_; ++j)
                    std::swap(LU_[k * n_ + j], LU_[max_row * n_ + j]);
            }

            // Elimination
            for (int i = k + 1; i < n_; ++i) {
                LU_[i * n_ + k] /= LU_[k * n_ + k];
                for (int j = k + 1; j < n_; ++j)
                    LU_[i * n_ + j] -= LU_[i * n_ + k] * LU_[k * n_ + j];
            }
        }
    }

    // Solve A_0 * x = b, result written to x
    void solve(const double* b, double* x) const {
        std::copy(b, b + n_, x);

        // Apply pivots + forward solve (L)
        for (int k = 0; k < n_; ++k) {
            if (piv_[k] != k)
                std::swap(x[k], x[piv_[k]]);
            for (int i = k + 1; i < n_; ++i)
                x[i] -= LU_[i * n_ + k] * x[k];
        }

        // Backward solve (U)
        for (int k = n_ - 1; k >= 0; --k) {
            for (int j = k + 1; j < n_; ++j)
                x[k] -= LU_[k * n_ + j] * x[j];
            x[k] /= LU_[k * n_ + k];
        }
    }

    std::vector<double> solve(const std::vector<double>& b) const {
        std::vector<double> x(n_);
        solve(b.data(), x.data());
        return x;
    }

private:
    int n_ = 0;
    std::vector<double> LU_;
    std::vector<int> piv_;
};

}  // namespace schwarz
