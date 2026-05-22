#pragma once

#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace schwarz {

struct CSRMatrix {
    int nrows = 0;
    int ncols = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> vals;

    int nnz() const { return static_cast<int>(vals.size()); }

    void resize(int nr, int nc, int nz) {
        nrows = nr;
        ncols = nc;
        row_ptr.resize(nr + 1);
        col_idx.resize(nz);
        vals.resize(nz);
    }

    void spmv(const double* x, double* y) const {
        for (int i = 0; i < nrows; ++i) {
            double sum = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
                sum += vals[j] * x[col_idx[j]];
            y[i] = sum;
        }
    }

    void spmv(const std::vector<double>& x, std::vector<double>& y) const {
        assert(static_cast<int>(x.size()) >= ncols);
        y.resize(nrows);
        spmv(x.data(), y.data());
    }

    double diagonal(int i) const {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_idx[j] == i)
                return vals[j];
        }
        return 0.0;
    }

    CSRMatrix extract_submatrix(const std::vector<int>& indices) const {
        int n = static_cast<int>(indices.size());
        // inv_map must cover both rows and columns (distributed matrix has ncols > nrows)
        std::vector<int> inv_map(std::max(nrows, ncols), -1);
        for (int i = 0; i < n; ++i)
            inv_map[indices[i]] = i;

        CSRMatrix sub;
        sub.nrows = n;
        sub.ncols = n;
        sub.row_ptr.resize(n + 1, 0);

        std::vector<int> ci;
        std::vector<double> cv;

        for (int li = 0; li < n; ++li) {
            int gi = indices[li];
            sub.row_ptr[li] = static_cast<int>(ci.size());
            for (int j = row_ptr[gi]; j < row_ptr[gi + 1]; ++j) {
                int lj = inv_map[col_idx[j]];
                if (lj >= 0) {
                    ci.push_back(lj);
                    cv.push_back(vals[j]);
                }
            }
        }
        sub.row_ptr[n] = static_cast<int>(ci.size());
        sub.col_idx = std::move(ci);
        sub.vals = std::move(cv);
        return sub;
    }
};

inline CSRMatrix make_identity(int n) {
    CSRMatrix I;
    I.nrows = n;
    I.ncols = n;
    I.row_ptr.resize(n + 1);
    I.col_idx.resize(n);
    I.vals.resize(n, 1.0);
    std::iota(I.row_ptr.begin(), I.row_ptr.end(), 0);
    std::iota(I.col_idx.begin(), I.col_idx.end(), 0);
    return I;
}

}  // namespace schwarz
