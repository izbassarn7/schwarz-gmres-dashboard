#pragma once

#include "preconditioner.h"
#include "../core/sparse_matrix.h"
#include <vector>
#include <cmath>

namespace schwarz {

class JacobiPrecond : public Preconditioner {
public:
    void setup(const CSRMatrix& A) override {
        inv_diag_.resize(A.nrows);
        for (int i = 0; i < A.nrows; ++i) {
            double d = A.diagonal(i);
            inv_diag_[i] = (std::abs(d) > 1e-15) ? 1.0 / d : 1.0;
        }
    }

    void apply(const double* x, double* y, int n) const override {
        for (int i = 0; i < n; ++i)
            y[i] = inv_diag_[i] * x[i];
    }

private:
    std::vector<double> inv_diag_;
};

}  // namespace schwarz
