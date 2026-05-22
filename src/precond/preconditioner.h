#pragma once

#include <vector>

namespace schwarz {

class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void setup(const struct CSRMatrix& A) = 0;

    // y = M^{-1} * x
    virtual void apply(const double* x, double* y, int n) const = 0;

    void apply(const std::vector<double>& x, std::vector<double>& y) const {
        y.resize(x.size());
        apply(x.data(), y.data(), static_cast<int>(x.size()));
    }
};

}  // namespace schwarz
