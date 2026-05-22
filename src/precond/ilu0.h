#pragma once

#include "preconditioner.h"
#include "../core/sparse_matrix.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace schwarz {

class ILU0Precond : public Preconditioner {
public:
    void setup(const CSRMatrix& A) override {
        n_ = A.nrows;
        row_ptr_ = A.row_ptr;
        col_idx_ = A.col_idx;
        lu_vals_ = A.vals;

        // In-place ILU(0) factorization using the existing sparsity pattern
        std::vector<int> diag_ptr(n_);
        for (int i = 0; i < n_; ++i) {
            diag_ptr[i] = -1;
            for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
                if (col_idx_[j] == i) {
                    diag_ptr[i] = j;
                    break;
                }
            }
        }

        for (int i = 1; i < n_; ++i) {
            for (int jp = row_ptr_[i]; jp < row_ptr_[i + 1]; ++jp) {
                int k = col_idx_[jp];
                if (k >= i) break;

                int dk = diag_ptr[k];
                if (dk < 0 || std::abs(lu_vals_[dk]) < 1e-30) continue;
                double a_ik = lu_vals_[jp] / lu_vals_[dk];
                lu_vals_[jp] = a_ik;

                for (int kp = dk + 1; kp < row_ptr_[k + 1]; ++kp) {
                    int col_k = col_idx_[kp];
                    for (int ip = row_ptr_[i]; ip < row_ptr_[i + 1]; ++ip) {
                        if (col_idx_[ip] == col_k) {
                            lu_vals_[ip] -= a_ik * lu_vals_[kp];
                            break;
                        }
                    }
                }
            }
        }

        diag_ptr_ = std::move(diag_ptr);
    }

    // Forward/backward solve: L*U*y = x
    void apply(const double* x, double* y, int n) const override {
        std::vector<double> tmp(n);
        std::copy(x, x + n, tmp.data());

        // Forward solve (L): z = L^{-1} x
        for (int i = 0; i < n_; ++i) {
            for (int jp = row_ptr_[i]; jp < row_ptr_[i + 1]; ++jp) {
                int col = col_idx_[jp];
                if (col >= i) break;
                tmp[i] -= lu_vals_[jp] * tmp[col];
            }
        }

        // Backward solve (U): y = U^{-1} z
        for (int i = n_ - 1; i >= 0; --i) {
            for (int jp = diag_ptr_[i] + 1; jp < row_ptr_[i + 1]; ++jp) {
                tmp[i] -= lu_vals_[jp] * tmp[col_idx_[jp]];
            }
            tmp[i] /= lu_vals_[diag_ptr_[i]];
        }

        std::copy(tmp.data(), tmp.data() + n, y);
    }

    const std::vector<int>& row_ptr() const { return row_ptr_; }
    const std::vector<int>& col_idx() const { return col_idx_; }
    const std::vector<double>& lu_vals() const { return lu_vals_; }
    const std::vector<int>& diag_ptr() const { return diag_ptr_; }
    int n() const { return n_; }

private:
    int n_ = 0;
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<double> lu_vals_;
    std::vector<int> diag_ptr_;
};

}  // namespace schwarz
