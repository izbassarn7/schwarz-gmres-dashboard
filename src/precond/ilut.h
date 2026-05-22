#pragma once

// ILU(t,p) — Incomplete LU with dual threshold:
//   drop_tol  : relative drop tolerance (entries < drop_tol * ||row||_inf are dropped)
//   fill_factor: maximum fill per row = fill_factor * avg_nnz_per_row
//
// Algorithm: right-looking ILUT (Saad, 1994).
// Reference: Saad, "Iterative Methods for Sparse Linear Systems", §10.4.

#include "preconditioner.h"
#include "../core/sparse_matrix.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace schwarz {

class ILUTPrecond : public Preconditioner {
public:
    // fill_factor: max fill per row as multiple of original nnz/row
    // drop_tol   : relative drop tolerance
    explicit ILUTPrecond(double drop_tol = 1e-4, int fill_factor = 10)
        : drop_tol_(drop_tol), fill_factor_(fill_factor) {}

    void setup(const CSRMatrix& A) override {
        n_ = A.nrows;
        if (n_ == 0) return;

        int avg_nnz = std::max(1, A.nnz() / n_);
        int max_fill = fill_factor_ * avg_nnz;

        // Allocate output L and U in CSR form
        // We store L (unit lower tri) and U (upper tri) separately.
        l_row_ptr_.resize(n_ + 1, 0);
        u_row_ptr_.resize(n_ + 1, 0);

        std::vector<std::vector<std::pair<int,double>>> L_rows(n_), U_rows(n_);

        // Working row: dense representation for current row being eliminated
        std::vector<double> w(n_, 0.0);
        std::vector<int>    nz_idx;  // indices of non-zeros in w
        nz_idx.reserve(2 * max_fill);

        for (int i = 0; i < n_; ++i) {
            // Fill w with row i of A
            for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
                int j = A.col_idx[p];
                if (w[j] == 0.0) nz_idx.push_back(j);
                w[j] += A.vals[p];
            }

            // Row infinity norm for drop tolerance
            double row_norm = 0.0;
            for (int j : nz_idx) row_norm = std::max(row_norm, std::abs(w[j]));
            double tol_i = drop_tol_ * row_norm;

            // Forward elimination: for each k < i with w[k] != 0
            std::sort(nz_idx.begin(), nz_idx.end());
            for (int k : nz_idx) {
                if (k >= i) break;
                if (std::abs(w[k]) < tol_i) { w[k] = 0.0; continue; }

                // w[k] /= u_kk
                double u_kk = diag_u_[k];
                if (std::abs(u_kk) < 1e-30) { w[k] = 0.0; continue; }
                double l_ik = w[k] / u_kk;
                w[k] = l_ik;

                // Subtract l_ik * U_row[k] from w
                for (auto& [col, val] : U_rows[k]) {
                    if (col <= k) continue;
                    if (w[col] == 0.0) nz_idx.push_back(col);
                    w[col] -= l_ik * val;
                }
            }

            // Re-sort and drop small entries
            std::sort(nz_idx.begin(), nz_idx.end());

            // Split into L and U parts, apply dual-threshold drop
            std::vector<std::pair<int,double>> l_part, u_part;
            for (int j : nz_idx) {
                if (std::abs(w[j]) == 0.0) continue;
                if (j < i) {
                    if (std::abs(w[j]) >= tol_i)
                        l_part.emplace_back(j, w[j]);
                } else {
                    if (j == i || std::abs(w[j]) >= tol_i)
                        u_part.emplace_back(j, w[j]);
                }
            }

            // Apply fill-factor limit: keep largest max_fill entries in each part
            auto keep_largest = [&](std::vector<std::pair<int,double>>& part, int lim) {
                if (static_cast<int>(part.size()) > lim) {
                    // Partial sort by abs value descending, keep top lim
                    std::partial_sort(part.begin(), part.begin() + lim, part.end(),
                        [](const std::pair<int,double>& a, const std::pair<int,double>& b) {
                            return std::abs(a.second) > std::abs(b.second);
                        });
                    part.resize(lim);
                    // Re-sort by column index
                    std::sort(part.begin(), part.end(),
                        [](const std::pair<int,double>& a, const std::pair<int,double>& b) {
                            return a.first < b.first;
                        });
                }
            };

            keep_largest(l_part, max_fill);
            keep_largest(u_part, max_fill);

            // Diagonal U entry (must be present)
            double diag = 0.0;
            for (auto& [col, val] : u_part)
                if (col == i) { diag = val; break; }
            if (std::abs(diag) < 1e-30) diag = 1e-30;  // safeguard
            diag_u_.push_back(diag);

            L_rows[i] = std::move(l_part);
            U_rows[i] = std::move(u_part);

            // Reset w
            for (int j : nz_idx) w[j] = 0.0;
            nz_idx.clear();
        }

        // Build CSR storage for L and U
        // L: unit lower triangular (off-diagonal only; diagonal implicitly 1)
        // U: upper triangular including diagonal
        for (int i = 0; i < n_; ++i) {
            l_row_ptr_[i + 1] = l_row_ptr_[i] + static_cast<int>(L_rows[i].size());
            u_row_ptr_[i + 1] = u_row_ptr_[i] + static_cast<int>(U_rows[i].size());
        }
        l_col_.resize(l_row_ptr_[n_]);
        l_val_.resize(l_row_ptr_[n_]);
        u_col_.resize(u_row_ptr_[n_]);
        u_val_.resize(u_row_ptr_[n_]);

        for (int i = 0; i < n_; ++i) {
            int lp = l_row_ptr_[i];
            for (auto& [c, v] : L_rows[i]) { l_col_[lp] = c; l_val_[lp] = v; ++lp; }
            int up = u_row_ptr_[i];
            for (auto& [c, v] : U_rows[i]) { u_col_[up] = c; u_val_[up] = v; ++up; }
        }
    }

    // Solve (LU) y = x via forward/backward substitution
    void apply(const double* x, double* y, int n) const override {
        std::vector<double> z(n_);

        // Forward solve L z = x  (L is unit lower triangular)
        for (int i = 0; i < n_; ++i) {
            z[i] = x[i];
            for (int p = l_row_ptr_[i]; p < l_row_ptr_[i + 1]; ++p)
                z[i] -= l_val_[p] * z[l_col_[p]];
            // diagonal of L is 1 (unit)
        }

        // Backward solve U y = z
        for (int i = n_ - 1; i >= 0; --i) {
            y[i] = z[i];
            // skip diagonal (first entry in each row of U)
            int diag_p = u_row_ptr_[i];
            for (int p = diag_p + 1; p < u_row_ptr_[i + 1]; ++p)
                y[i] -= u_val_[p] * y[u_col_[p]];
            y[i] /= diag_u_[i];
        }
    }

    int n() const { return n_; }
    int l_nnz() const { return static_cast<int>(l_val_.size()); }
    int u_nnz() const { return static_cast<int>(u_val_.size()); }

private:
    int    n_           = 0;
    double drop_tol_    = 1e-4;
    int    fill_factor_ = 10;

    // L factors (off-diagonal lower part, sorted by col)
    std::vector<int>    l_row_ptr_;
    std::vector<int>    l_col_;
    std::vector<double> l_val_;

    // U factors (upper tri including diagonal)
    std::vector<int>    u_row_ptr_;
    std::vector<int>    u_col_;
    std::vector<double> u_val_;
    std::vector<double> diag_u_;  // u[i,i] per row
};

}  // namespace schwarz
